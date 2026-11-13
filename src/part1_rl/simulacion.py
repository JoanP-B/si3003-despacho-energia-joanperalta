"""Parte 1 — Simulación del despacho con las políticas ya calculadas.

Carga los artefactos que guarda ``notebooks/02_mdp_rl.ipynb`` en ``data/processed/``:

- ``q_table.json``: Q-Table del agente Q-Learning y metadatos (umbrales de cuartil, etc.).
- ``politica_referencia.json``: política de Iteración de Valores (referencia).
- ``precios_evaluacion.csv``: precios reales de XM de los días de prueba (15 % final).

Recorre cada día de evaluación (episodio de 24 h, batería vacía al inicio) con tres
políticas: Q-Learning (greedy), Línea Base justa (carga 01-04 h, descarga 18-21 h) e
Iteración de Valores. Devuelve el resumen económico y la traza hora a hora.

La capacidad se reparte en los 4 escalones de SoC (0→25→50→75→100 %), así que cada
carga o descarga mueve ``capacidad_kwh / 4`` kWh. Las políticas no dependen de la
capacidad: al escalar la energía solo se escala el flujo de caja.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from src.part1_rl.agent import QLearningAgent
from src.part1_rl.environment import Accion, BatteryEnv, State
from src.part1_rl.value_iteration import PoliticaReferencia

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESADO = RAIZ / "data" / "processed"
Q_TABLE_PATH = DIR_PROCESADO / "q_table.json"
POLITICA_REF_NOMBRE = "politica_referencia.json"
CAPACIDAD_MAX_KWH = 100_000.0
COLUMNAS_TRAZA = [
    "timestamp", "precio_cop", "soc_rl", "accion_rl",
    "ahorro_acum_rl", "ahorro_acum_base", "ahorro_acum_ref",
]


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


def _cargar_precios(ruta: Path) -> pd.Series:
    if not ruta.exists():
        raise SimulacionError(
            f"No existen los precios de evaluación en {ruta}. "
            "Ejecuta notebooks/02_mdp_rl.ipynb para generarlos."
        )
    try:
        df = pd.read_csv(ruta, parse_dates=["timestamp"])
        precios = pd.to_numeric(df["precio_cop_kwh"], errors="raise")
    except (KeyError, ValueError, pd.errors.ParserError) as exc:
        raise SimulacionError(f"Archivo de precios inválido ({ruta.name}): {exc}") from exc
    precios.index = pd.DatetimeIndex(df["timestamp"])
    return precios


def _filtrar_fechas(precios: pd.Series, inicio: date | None, fin: date | None) -> pd.Series:
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


def simular(
    capacidad_kwh: float,
    fecha_inicio: "date | datetime | str | None" = None,
    fecha_fin: "date | datetime | str | None" = None,
    q_table_path: Path = Q_TABLE_PATH,
) -> ResultadoSimulacion:
    """Simula los días de evaluación con Q-Learning, Línea Base e Iteración de Valores.

    ``fecha_inicio``/``fecha_fin`` (opcionales, ``AAAA-MM-DD``) filtran días dentro del
    periodo de evaluación; sin ellas se usan todos.

    Raises
    ------
    SimulacionError
        Capacidad o fechas inválidas, o artefactos ausentes/corruptos.
    """
    try:
        capacidad = float(capacidad_kwh)
    except (TypeError, ValueError) as exc:
        raise SimulacionError(f"capacidad_kwh debe ser un número; se recibió {capacidad_kwh!r}.") from exc
    if not math.isfinite(capacidad) or capacidad <= 0:
        raise SimulacionError("capacidad_kwh debe ser un número mayor que 0.")
    if capacidad > CAPACIDAD_MAX_KWH:
        raise SimulacionError(f"capacidad_kwh no puede superar {CAPACIDAD_MAX_KWH:,.0f} kWh.")
    inicio = _a_fecha(fecha_inicio, "fecha_inicio")
    fin = _a_fecha(fecha_fin, "fecha_fin")

    q_table_path = Path(q_table_path)
    try:
        agente = QLearningAgent.cargar(q_table_path)
    except FileNotFoundError as exc:
        raise SimulacionError(f"{exc} Entrena el agente con notebooks/02_mdp_rl.ipynb.") from exc
    except ValueError as exc:
        raise SimulacionError(str(exc)) from exc

    meta = agente.metadatos
    if "umbrales_precio" not in meta:
        raise SimulacionError(
            "La Q-Table no tiene los umbrales de precio del entrenamiento. "
            "Vuelve a ejecutar notebooks/02_mdp_rl.ipynb para regenerarla."
        )
    ruta_ref = q_table_path.parent / meta.get("archivo_politica_referencia", POLITICA_REF_NOMBRE)
    try:
        referencia = PoliticaReferencia.cargar(ruta_ref)
    except FileNotFoundError as exc:
        raise SimulacionError(f"{exc} Ejecuta notebooks/02_mdp_rl.ipynb para generarla.") from exc
    except ValueError as exc:
        raise SimulacionError(str(exc)) from exc

    precios = _cargar_precios(
        q_table_path.parent / meta.get("archivo_precios_evaluacion", "precios_evaluacion.csv")
    )
    precios = _filtrar_fechas(precios, inicio, fin)

    energia = capacidad / (BatteryEnv.N_NIVELES_SOC - 1)
    try:
        env = BatteryEnv(
            precios.to_numpy(),
            soc_inicial=int(meta.get("soc_inicial", 0)),
            penalizacion_invalida=float(meta.get("penalizacion_invalida", -1000.0)),
            energia_por_nivel_kwh=energia,
            umbrales_precio=meta["umbrales_precio"],
            aleatorio=False,
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

    def resumen(pasos: list[dict]) -> tuple[pd.Series, int, int, int]:
        flujo = pd.Series([p["flujo_caja_cop"] for p in pasos], dtype=float)
        validos = [p for p in pasos if p["valida"]]
        cargas = sum(p["accion"] == Accion.CARGAR for p in validos)
        descargas = sum(p["accion"] == Accion.DESCARGAR for p in validos)
        return flujo.cumsum(), cargas, descargas, len(pasos) - len(validos)

    pasos_rl = recorrer(lambda s: agente.seleccionar_accion(s, explorar=False))
    acum_rl, cargas_rl, descargas_rl, invalidas_rl = resumen(pasos_rl)
    acum_base, cargas_base, descargas_base, _ = resumen(recorrer(BatteryEnv.naive_policy))
    acum_ref, cargas_ref, descargas_ref, _ = resumen(recorrer(referencia.accion))

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
        traza=traza,
    )
