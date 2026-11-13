"""Parte 1 — Simulación del despacho con las políticas ya calculadas.

Carga los artefactos que guarda ``notebooks/02_mdp_rl.ipynb`` en ``data/processed/``:

- ``q_table.json``: Q-Table del escenario base (5,000 episodios, batería ideal) y metadatos.
  Es la que usan el agente LLM y la batería de QA.
- ``escenarios/q_{escenario}_eta{NN}.json``: Q-Tables de los escenarios de entrenamiento
  (``corto``, ``base``, ``largo``) para cada eficiencia de la grilla ``EFICIENCIAS_PCT``.
- ``politica_referencia.json``: política de Iteración de Valores y su modelo empírico, que
  permite replanificarla para cualquier eficiencia.
- ``precios_evaluacion.csv`` / ``precios_validacion.csv``: precios reales de XM de los días
  de prueba y de validación, con los umbrales de cuartil de cada hora (ventana móvil causal
  de 168 h calculada con la serie completa).
- ``curva_aprendizaje.csv``: ahorro de Q-Learning según los episodios de entrenamiento.

Recorre cada día (episodio de 24 h, batería vacía al inicio) con tres políticas: Q-Learning
(greedy), Línea Base justa (carga 01-04 h, descarga 18-21 h) e Iteración de Valores, y
devuelve el resumen económico y la traza hora a hora.

Parámetros físicos:

- **Capacidad:** se reparte en 4 escalones de SoC, así que cada carga o descarga mueve
  ``capacidad_kwh / 4`` kWh en una hora.
- **Potencia del inversor:** limita la energía de cada hora. Si es menor que
  ``capacidad_kwh / 4``, cada escalón mueve solo ``potencia_inversor_kw`` kWh. Si es mayor,
  no cambia nada, porque el modelo mueve un escalón por hora.
- **Eficiencia de ida y vuelta:** de cada kWh comprado solo se entregan ``eficiencia`` kWh
  al descargar. Se aplica al costo evitado de la descarga, no a las compras. La Iteración de
  Valores se replanifica con esa eficiencia (conoce el modelo) y Q-Learning usa la Q-Table
  entrenada con la eficiencia de la grilla más cercana.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.part1_rl.agent import QLearningAgent
from src.part1_rl.environment import Accion, BatteryEnv, State
from src.part1_rl.value_iteration import PoliticaReferencia

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESADO = RAIZ / "data" / "processed"
DIR_ESCENARIOS = DIR_PROCESADO / "escenarios"
Q_TABLE_PATH = DIR_PROCESADO / "q_table.json"
CURVA_PATH = DIR_PROCESADO / "curva_aprendizaje.csv"
POLITICA_REF_NOMBRE = "politica_referencia.json"
CAPACIDAD_MAX_KWH = 100_000.0
HORAS_POR_PASO = 1.0

#: Escenarios de entrenamiento de Q-Learning (episodios de cada uno). "base" es el que usan
#: el agente LLM y la QA.
ESCENARIOS: dict[str, int] = {"corto": 50, "base": 5_000, "largo": 50_000}
#: Eficiencias (%) para las que el notebook entrena una Q-Table por escenario.
EFICIENCIAS_PCT: tuple[int, ...] = (70, 75, 80, 85, 90, 95, 100)

COLUMNAS_TRAZA = [
    "timestamp", "precio_cop", "soc_rl", "accion_rl",
    "ahorro_acum_rl", "ahorro_acum_base", "ahorro_acum_ref",
]
COLUMNAS_UMBRALES = ["umbral_p25", "umbral_p50", "umbral_p75"]
COLUMNAS_CURVA = [
    "eficiencia_pct", "episodios", "ql_validacion", "ql_prueba",
    "base_validacion", "base_prueba", "ref_validacion", "ref_prueba",
]
PERIODOS = {
    "prueba": ("archivo_precios_evaluacion", "precios_evaluacion.csv"),
    "validacion": ("archivo_precios_validacion", "precios_validacion.csv"),
}


class SimulacionError(RuntimeError):
    """Faltan artefactos o los parámetros de la simulación no son válidos."""


@dataclass(frozen=True)
class ResultadoSimulacion:
    capacidad_kwh: float
    energia_por_nivel_kwh: float
    n_dias: int
    fecha_inicio: str
    fecha_fin: str
    fuente_datos: str
    ahorro_qlearning_cop: float
    ahorro_base_cop: float
    ahorro_referencia_cop: float
    cargas_qlearning: int
    descargas_qlearning: int
    invalidas_qlearning: int
    cargas_base: int
    descargas_base: int
    cargas_referencia: int
    descargas_referencia: int
    escenario: str = "base"
    periodo: str = "prueba"
    eficiencia: float = 1.0
    eficiencia_entrenamiento_ql: float = 1.0
    episodios_ql: int | None = None
    potencia_inversor_kw: float | None = None
    traza: pd.DataFrame = field(repr=False, compare=False, default_factory=pd.DataFrame)
    """Serie hora a hora con las columnas de ``COLUMNAS_TRAZA``.

    ``soc_rl`` es el nivel de carga (%) tras la acción de Q-Learning; ``accion_rl`` es
    ``MANTENER``/``CARGAR``/``DESCARGAR``; los ``ahorro_acum_*`` son acumulados en COP.
    """

    @property
    def diferencia_cop(self) -> float:
        """Ahorro de Q-Learning menos el de la Línea Base."""
        return self.ahorro_qlearning_cop - self.ahorro_base_cop

    @property
    def fraccion_de_referencia(self) -> float | None:
        """Ahorro de Q-Learning como fracción del de Iteración de Valores."""
        if not self.ahorro_referencia_cop:
            return None
        return self.ahorro_qlearning_cop / self.ahorro_referencia_cop

    @property
    def limitado_por_inversor(self) -> bool:
        """True si el inversor obliga a mover menos de ``capacidad / 4`` kWh por hora."""
        escalon = self.capacidad_kwh / (BatteryEnv.N_NIVELES_SOC - 1)
        return self.energia_por_nivel_kwh < escalon - 1e-9

    def vs_base(self, valor: float) -> float | None:
        """Diferencia relativa de ``valor`` frente al ahorro de la Línea Base."""
        if not self.ahorro_base_cop:
            return None
        return (valor - self.ahorro_base_cop) / abs(self.ahorro_base_cop)

    def resumen(self) -> dict[str, Any]:
        """Cifras clave ya calculadas (sin la traza), p. ej. para el resumen del LLM."""

        def pct(valor: float | None) -> float | None:
            return None if valor is None else round(100 * valor, 1)

        fraccion = self.fraccion_de_referencia
        return {
            "periodo_evaluado": (
                "prueba: días nunca usados para entrenar ni para elegir hiperparámetros"
                if self.periodo == "prueba"
                else "validación: días usados para elegir hiperparámetros"
            ),
            "fechas": f"{self.fecha_inicio} a {self.fecha_fin}",
            "dias_simulados": self.n_dias,
            "fuente_de_precios": f"{self.fuente_datos} (precio de bolsa, no tarifa del usuario)",
            "capacidad_bateria_kwh": round(self.capacidad_kwh, 1),
            "eficiencia_ida_y_vuelta_pct": round(100 * self.eficiencia, 1),
            "potencia_inversor_kw": (
                None if self.potencia_inversor_kw is None else round(self.potencia_inversor_kw, 1)
            ),
            "energia_por_hora_kwh": round(self.energia_por_nivel_kwh, 2),
            "inversor_limita_la_energia_por_hora": self.limitado_por_inversor,
            "escenario_de_entrenamiento_q_learning": (
                f"{self.escenario} ({self.episodios_ql:,} episodios)"
                if self.episodios_ql else self.escenario
            ),
            "ahorro_total_cop": {
                "q_learning": round(self.ahorro_qlearning_cop),
                "linea_base_regla_fija": round(self.ahorro_base_cop),
                "iteracion_de_valores_referencia": round(self.ahorro_referencia_cop),
            },
            "q_learning_vs_linea_base_pct": pct(self.vs_base(self.ahorro_qlearning_cop)),
            "iteracion_de_valores_vs_linea_base_pct": pct(self.vs_base(self.ahorro_referencia_cop)),
            "diferencia_q_learning_menos_linea_base_cop": round(self.diferencia_cop),
            "q_learning_como_pct_de_la_referencia": pct(fraccion),
            "ahorro_promedio_diario_cop": {
                "q_learning": round(self.ahorro_qlearning_cop / self.n_dias),
                "linea_base_regla_fija": round(self.ahorro_base_cop / self.n_dias),
            },
            "ciclos_de_carga": {
                "q_learning": self.cargas_qlearning,
                "linea_base_regla_fija": self.cargas_base,
                "iteracion_de_valores_referencia": self.cargas_referencia,
            },
        }


# --------------------------------------------------------------------------- #
# Artefactos
# --------------------------------------------------------------------------- #
def eficiencia_de_grilla(eficiencia: float) -> int:
    """Eficiencia (%) de la grilla de entrenamiento más cercana a ``eficiencia`` (0-1]."""
    return min(EFICIENCIAS_PCT, key=lambda e: abs(e - 100 * float(eficiencia)))


def ruta_escenario(escenario: str = "base", eficiencia: float = 1.0) -> Path:
    """Ruta de la Q-Table de un escenario entrenado con la eficiencia más cercana."""
    if escenario not in ESCENARIOS:
        raise SimulacionError(
            f"escenario debe ser uno de {list(ESCENARIOS)}; se recibió {escenario!r}."
        )
    pct = eficiencia_de_grilla(eficiencia)
    if escenario == "base" and pct == 100:
        return Q_TABLE_PATH
    return DIR_ESCENARIOS / f"q_{escenario}_eta{pct}.json"


def escenarios_disponibles(eficiencia: float = 1.0) -> list[str]:
    """Escenarios cuya Q-Table existe para la eficiencia dada."""
    return [e for e in ESCENARIOS if ruta_escenario(e, eficiencia).exists()]


def curva_aprendizaje(ruta: Path = CURVA_PATH) -> pd.DataFrame:
    """Ahorro de Q-Learning por episodios y eficiencia (batería de 4 kWh, periodo completo)."""
    if not Path(ruta).exists():
        raise SimulacionError(
            f"No existe la curva de aprendizaje en {ruta}. Ejecuta notebooks/02_mdp_rl.ipynb."
        )
    try:
        curva = pd.read_csv(ruta)
        curva = curva[COLUMNAS_CURVA].apply(pd.to_numeric, errors="raise")
    except (KeyError, ValueError, pd.errors.ParserError) as exc:
        raise SimulacionError(f"Curva de aprendizaje inválida ({Path(ruta).name}): {exc}") from exc
    return curva


def _artefacto(q_table_path: Path, nombre: str) -> Path:
    """Busca un artefacto compartido junto a la Q-Table o en ``data/processed``."""
    junto = Path(q_table_path).parent / nombre
    return junto if junto.exists() else DIR_PROCESADO / nombre


def _a_fecha(valor: "date | datetime | str | None", nombre: str) -> date | None:
    if valor is None or valor == "":
        return None
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    try:
        return datetime.strptime(str(valor).strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise SimulacionError(f"{nombre} debe tener formato AAAA-MM-DD; se recibió {valor!r}.") from exc


def _cargar_precios(ruta: Path) -> pd.DataFrame:
    """Lee un CSV de precios: precio y, si existen, umbrales causales por hora."""
    if not ruta.exists():
        raise SimulacionError(
            f"No existen los precios de evaluación en {ruta}. "
            "Ejecuta notebooks/02_mdp_rl.ipynb para generarlos."
        )
    try:
        df = pd.read_csv(ruta, parse_dates=["timestamp"])
        columnas = ["precio_cop_kwh"] + [c for c in COLUMNAS_UMBRALES if c in df.columns]
        datos = df[columnas].apply(pd.to_numeric, errors="raise")
    except (KeyError, ValueError, pd.errors.ParserError) as exc:
        raise SimulacionError(f"Archivo de precios inválido ({ruta.name}): {exc}") from exc
    datos.index = pd.DatetimeIndex(df["timestamp"])
    return datos


def _ruta_precios(q_table_path: Path, meta: dict, periodo: str) -> Path:
    if periodo not in PERIODOS:
        raise SimulacionError(f"periodo debe ser uno de {sorted(PERIODOS)}; se recibió {periodo!r}.")
    clave, por_defecto = PERIODOS[periodo]
    return _artefacto(q_table_path, meta.get(clave, por_defecto))


def periodo_evaluacion(
    q_table_path: Path = Q_TABLE_PATH, periodo: str = "prueba"
) -> tuple[date, date]:
    """Primer y último día disponibles para ``simular()`` (para el dashboard)."""
    try:
        meta = QLearningAgent.cargar(q_table_path).metadatos
    except (FileNotFoundError, ValueError) as exc:
        raise SimulacionError(f"{exc} Entrena el agente con notebooks/02_mdp_rl.ipynb.") from exc
    datos = _cargar_precios(_ruta_precios(q_table_path, meta, periodo))
    return datos.index.min().date(), datos.index.max().date()


def _filtrar_fechas(precios: pd.DataFrame, inicio: date | None, fin: date | None) -> pd.DataFrame:
    disponibles_ini = precios.index.min().date()
    disponibles_fin = precios.index.max().date()
    rango = f"{disponibles_ini} a {disponibles_fin}"
    if inicio and fin and inicio > fin:
        raise SimulacionError("fecha_inicio es posterior a fecha_fin.")
    for valor, nombre in ((inicio, "fecha_inicio"), (fin, "fecha_fin")):
        if valor and not (disponibles_ini <= valor <= disponibles_fin):
            raise SimulacionError(
                f"{nombre} {valor} está fuera del periodo de evaluación ({rango})."
            )
    dias = precios.index.date
    mascara = pd.Series(True, index=precios.index)
    if inicio:
        mascara &= dias >= inicio
    if fin:
        mascara &= dias <= fin
    filtrado = precios[mascara.to_numpy()]
    if filtrado.empty:
        raise SimulacionError(f"No hay días de evaluación en el rango pedido (disponible: {rango}).")
    return filtrado


def _validar_numero(valor: Any, nombre: str) -> float:
    try:
        numero = float(valor)
    except (TypeError, ValueError) as exc:
        raise SimulacionError(f"{nombre} debe ser un número; se recibió {valor!r}.") from exc
    if not math.isfinite(numero):
        raise SimulacionError(f"{nombre} debe ser un número finito.")
    return numero


# --------------------------------------------------------------------------- #
# Simulación
# --------------------------------------------------------------------------- #
def simular(
    capacidad_kwh: float,
    fecha_inicio: "date | datetime | str | None" = None,
    fecha_fin: "date | datetime | str | None" = None,
    q_table_path: Path | None = None,
    periodo: str = "prueba",
    escenario: str = "base",
    eficiencia: float = 1.0,
    potencia_inversor_kw: float | None = None,
) -> ResultadoSimulacion:
    """Simula los días de evaluación con Q-Learning, Línea Base e Iteración de Valores.

    Parameters
    ----------
    capacidad_kwh:
        Capacidad de la batería (kWh), mayor que 0.
    fecha_inicio, fecha_fin:
        Opcionales (``AAAA-MM-DD``): filtran días dentro del periodo; sin ellas se usan todos.
    q_table_path:
        Q-Table a usar. Si es ``None`` se toma la del ``escenario`` entrenado con la
        eficiencia de la grilla más cercana (ver ``ruta_escenario``).
    periodo:
        ``"prueba"`` (por defecto, nunca usados para decidir nada) o ``"validacion"`` (usados
        para elegir hiperparámetros).
    escenario:
        ``"corto"``, ``"base"`` (por defecto) o ``"largo"``: presupuesto de entrenamiento.
    eficiencia:
        Eficiencia de ida y vuelta en (0, 1]. Multiplica el costo evitado de cada descarga.
        Con 1.0 (por defecto) la batería es ideal, como en el agente LLM y la QA.
    potencia_inversor_kw:
        Opcional. Limita la energía de cada hora a ``potencia_inversor_kw · 1 h``.

    Raises
    ------
    SimulacionError
        Parámetros inválidos o artefactos ausentes/corruptos.
    """
    capacidad = _validar_numero(capacidad_kwh, "capacidad_kwh")
    if capacidad <= 0:
        raise SimulacionError("capacidad_kwh debe ser un número mayor que 0.")
    if capacidad > CAPACIDAD_MAX_KWH:
        raise SimulacionError(f"capacidad_kwh no puede superar {CAPACIDAD_MAX_KWH:,.0f} kWh.")
    eta = _validar_numero(eficiencia, "eficiencia")
    if not (0.0 < eta <= 1.0):
        raise SimulacionError("eficiencia debe estar entre 0 y 1 (por ejemplo 0.9 para 90 %).")
    potencia = None
    if potencia_inversor_kw is not None:
        potencia = _validar_numero(potencia_inversor_kw, "potencia_inversor_kw")
        if potencia <= 0:
            raise SimulacionError("potencia_inversor_kw debe ser mayor que 0.")
    inicio = _a_fecha(fecha_inicio, "fecha_inicio")
    fin = _a_fecha(fecha_fin, "fecha_fin")

    q_table_path = Path(q_table_path) if q_table_path is not None else ruta_escenario(escenario, eta)
    try:
        agente = QLearningAgent.cargar(q_table_path)
    except FileNotFoundError as exc:
        raise SimulacionError(f"{exc} Ejecuta notebooks/02_mdp_rl.ipynb para generarla.") from exc
    except ValueError as exc:
        raise SimulacionError(str(exc)) from exc

    meta = agente.metadatos
    penalizacion = float(meta.get("penalizacion_invalida", -1000.0))
    ruta_ref = _artefacto(q_table_path, meta.get("archivo_politica_referencia", POLITICA_REF_NOMBRE))
    try:
        referencia = PoliticaReferencia.cargar(ruta_ref).replanificar(eta, penalizacion)
    except FileNotFoundError as exc:
        raise SimulacionError(f"{exc} Ejecuta notebooks/02_mdp_rl.ipynb para generarla.") from exc
    except ValueError as exc:
        raise SimulacionError(str(exc)) from exc

    precios = _cargar_precios(_ruta_precios(q_table_path, meta, periodo))
    datos = _filtrar_fechas(precios, inicio, fin)
    precios = datos["precio_cop_kwh"]

    # Umbrales causales precalculados por hora (ventana móvil) o, en artefactos antiguos,
    # los cuartiles fijos del entrenamiento guardados en los metadatos.
    if set(COLUMNAS_UMBRALES) <= set(datos.columns):
        umbrales = datos[COLUMNAS_UMBRALES].to_numpy()
    elif "umbrales_precio" in meta:
        umbrales = meta["umbrales_precio"]
    else:
        raise SimulacionError(
            "Faltan los umbrales de cuartil de precio (columnas umbral_p25..p75 en los "
            "precios de evaluación). Vuelve a ejecutar notebooks/02_mdp_rl.ipynb."
        )

    energia = capacidad / (BatteryEnv.N_NIVELES_SOC - 1)
    if potencia is not None:
        energia = min(energia, potencia * HORAS_POR_PASO)
    try:
        env = BatteryEnv(
            precios.to_numpy(),
            soc_inicial=int(meta.get("soc_inicial", 0)),
            penalizacion_invalida=penalizacion,
            energia_por_nivel_kwh=energia,
            umbrales_precio=umbrales,
            aleatorio=False,
            eficiencia=eta,
        )
    except ValueError as exc:
        raise SimulacionError(f"No se pudo crear el entorno: {exc}") from exc

    def recorrer(selector: Callable[[State], int]) -> list[dict]:
        pasos: list[dict] = []
        for dia in range(env.n_dias):
            estado = env.reset(dia=dia)
            terminado = False
            while not terminado:
                estado, _, terminado, info = env.step(selector(estado))
                pasos.append(info)
        return pasos

    def resumir(pasos: list[dict]) -> tuple[pd.Series, int, int, int]:
        flujo = pd.Series([p["flujo_caja_cop"] for p in pasos], dtype=float)
        validos = [p for p in pasos if p["valida"]]
        cargas = sum(p["accion"] == Accion.CARGAR for p in validos)
        descargas = sum(p["accion"] == Accion.DESCARGAR for p in validos)
        return flujo.cumsum(), cargas, descargas, len(pasos) - len(validos)

    pasos_rl = recorrer(lambda s: agente.seleccionar_accion(s, explorar=False))
    acum_rl, cargas_rl, descargas_rl, invalidas_rl = resumir(pasos_rl)
    acum_base, cargas_base, descargas_base, _ = resumir(recorrer(BatteryEnv.naive_policy))
    acum_ref, cargas_ref, descargas_ref, _ = resumir(recorrer(referencia.accion))

    niveles = BatteryEnv.N_NIVELES_SOC - 1
    traza = pd.DataFrame({
        "timestamp": precios.index,
        "precio_cop": precios.to_numpy(),
        "soc_rl": [100.0 * p["soc"] / niveles for p in pasos_rl],
        "accion_rl": [
            Accion(p["accion"]).name if p["valida"] else "MANTENER" for p in pasos_rl
        ],
        "ahorro_acum_rl": acum_rl.to_numpy(),
        "ahorro_acum_base": acum_base.to_numpy(),
        "ahorro_acum_ref": acum_ref.to_numpy(),
    })[COLUMNAS_TRAZA]

    episodios = meta.get("episodios")
    return ResultadoSimulacion(
        capacidad_kwh=capacidad,
        energia_por_nivel_kwh=energia,
        n_dias=env.n_dias,
        fecha_inicio=str(precios.index.min().date()),
        fecha_fin=str(precios.index.max().date()),
        fuente_datos=str(meta.get("fuente_datos", "desconocida")),
        ahorro_qlearning_cop=float(acum_rl.iloc[-1]),
        ahorro_base_cop=float(acum_base.iloc[-1]),
        ahorro_referencia_cop=float(acum_ref.iloc[-1]),
        cargas_qlearning=int(cargas_rl),
        descargas_qlearning=int(descargas_rl),
        invalidas_qlearning=int(invalidas_rl),
        cargas_base=int(cargas_base),
        descargas_base=int(descargas_base),
        cargas_referencia=int(cargas_ref),
        descargas_referencia=int(descargas_ref),
        escenario=str(meta.get("escenario", escenario)),
        periodo=periodo,
        eficiencia=eta,
        eficiencia_entrenamiento_ql=float(meta.get("eficiencia_entrenamiento", 1.0)),
        episodios_ql=int(episodios) if episodios is not None else None,
        potencia_inversor_kw=potencia,
        traza=traza,
    )
