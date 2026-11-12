"""Parte 1 — Simulación del despacho con la política Q-Learning ya entrenada.

Carga la Q-Table (``data/processed/q_table.json``) y los precios de evaluación que guarda
``notebooks/02_mdp_rl.ipynb``, y recorre cada día de evaluación (un episodio de 24 h)
con la política greedy del agente y con la Línea Base ingenua.

La capacidad se reparte en los 4 escalones de SoC (0→25→50→75→100 %), así que cada
carga o descarga mueve ``capacidad_kwh / 4`` kWh. La política aprendida no depende de
la capacidad: al escalar la energía solo se escala el flujo de caja.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.part1_rl.agent import QLearningAgent
from src.part1_rl.environment import Accion, BatteryEnv

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESADO = RAIZ / "data" / "processed"
Q_TABLE_PATH = DIR_PROCESADO / "q_table.json"
CAPACIDAD_MAX_KWH = 100_000.0


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
    cargas_qlearning: int
    descargas_qlearning: int
    invalidas_qlearning: int

    @property
    def diferencia_cop(self) -> float:
        return self.ahorro_qlearning_cop - self.ahorro_base_cop


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
    precios.index = df["timestamp"]
    return precios


def simular(capacidad_kwh: float, q_table_path: Path = Q_TABLE_PATH) -> ResultadoSimulacion:
    """Simula los días de evaluación con Q-Learning y con la Línea Base.

    Raises
    ------
    SimulacionError
        Capacidad inválida, Q-Table ausente/corrupta o sin metadatos, o precios ausentes.
    """
    try:
        capacidad = float(capacidad_kwh)
    except (TypeError, ValueError) as exc:
        raise SimulacionError(f"capacidad_kwh debe ser un número; se recibió {capacidad_kwh!r}.") from exc
    if not math.isfinite(capacidad) or capacidad <= 0:
        raise SimulacionError("capacidad_kwh debe ser un número mayor que 0.")
    if capacidad > CAPACIDAD_MAX_KWH:
        raise SimulacionError(f"capacidad_kwh no puede superar {CAPACIDAD_MAX_KWH:,.0f} kWh.")

    try:
        agente = QLearningAgent.cargar(q_table_path)
    except FileNotFoundError as exc:
        raise SimulacionError(
            f"{exc} Entrena el agente con notebooks/02_mdp_rl.ipynb."
        ) from exc
    except ValueError as exc:
        raise SimulacionError(str(exc)) from exc

    meta = agente.metadatos
    if "umbrales_precio" not in meta:
        raise SimulacionError(
            "La Q-Table no tiene los umbrales de precio del entrenamiento. "
            "Vuelve a ejecutar notebooks/02_mdp_rl.ipynb para regenerarla."
        )
    precios = _cargar_precios(
        q_table_path.parent / meta.get("archivo_precios_evaluacion", "precios_evaluacion.csv")
    )

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

    def recorrer(selector) -> tuple[float, int, int, int]:
        total, cargas, descargas, invalidas = 0.0, 0, 0, 0
        for dia in range(env.n_dias):
            estado = env.reset(dia=dia)
            terminado = False
            while not terminado:
                estado, _, terminado, info = env.step(selector(estado))
                total += info["flujo_caja_cop"]
                if not info["valida"]:
                    invalidas += 1
                elif info["accion"] == Accion.CARGAR:
                    cargas += 1
                elif info["accion"] == Accion.DESCARGAR:
                    descargas += 1
        return total, cargas, descargas, invalidas

    ahorro_ql, cargas, descargas, invalidas = recorrer(
        lambda s: agente.seleccionar_accion(s, explorar=False)
    )
    ahorro_base, *_ = recorrer(BatteryEnv.naive_policy)

    return ResultadoSimulacion(
        capacidad_kwh=capacidad,
        energia_por_nivel_kwh=energia,
        n_dias=env.n_dias,
        fecha_inicio=str(precios.index.min().date()),
        fecha_fin=str(precios.index.max().date()),
        fuente_datos=str(meta.get("fuente_datos", "desconocida")),
        ahorro_qlearning_cop=ahorro_ql,
        ahorro_base_cop=ahorro_base,
        cargas_qlearning=cargas,
        descargas_qlearning=descargas,
        invalidas_qlearning=invalidas,
    )
