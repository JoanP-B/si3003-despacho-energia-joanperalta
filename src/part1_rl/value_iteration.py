"""Parte 1 — Política de referencia por Iteración de Valores (implementación propia).

Mismo MDP que ``BatteryEnv`` (estado ``(hora, cuartil_precio, nivel_soc)``), pero resuelto
con programación dinámica sobre un **modelo empírico** estimado solo con los días de
entrenamiento:

- ``precio_medio[h, q]``: precio medio observado en la hora ``h`` cuando su cuartil es ``q``
  (cuartil relativo a la ventana móvil causal de 168 h, el mismo que ve el agente).
  La recompensa esperada de cargar/descargar es ``∓precio_medio[h, q] · energía``.
- ``transicion[h, q, q']``: probabilidad empírica de que la hora ``h+1`` caiga en el cuartil
  ``q'`` dado que la hora ``h`` está en ``q`` (pares de horas consecutivas del mismo día).
- La hora y el SoC evolucionan de forma determinista; tras la hora 23 el episodio termina.

Bellman:
    V(h,q,s) = max_a [ R(h,q,s,a) + γ · Σ_q' T(h,q,q') · V(h+1, q', s'(s,a)) ]

Se itera hasta que ``max |V_k+1 − V_k| < tol``. Como la hora forma parte del estado, el
problema tiene horizonte 24 y converge en a lo sumo 25 barridos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.part1_rl.environment import Accion, BatteryEnv, State

N_H = BatteryEnv.N_HORAS
N_Q = BatteryEnv.N_CUARTILES
N_S = BatteryEnv.N_NIVELES_SOC
N_A = BatteryEnv.N_ACCIONES


@dataclass
class ModeloEmpirico:
    precio_medio: np.ndarray  # (24, 4)
    transicion: np.ndarray  # (23, 4, 4); fila h = P(q_{h+1} | q_h)
    conteos: np.ndarray  # (24, 4) visitas a cada (hora, cuartil)


@dataclass
class PoliticaReferencia:
    """Política greedy respecto a V*; ``accion(estado)`` es compatible con el entorno."""

    politica: np.ndarray  # (24, 4, 5) de enteros 0..2
    valores: np.ndarray  # (24, 4, 5) V*
    iteraciones: int
    gamma: float

    def accion(self, estado: State) -> int:
        h, q, s = estado
        return int(self.politica[h, q, s])

    __call__ = accion

    def guardar(self, ruta: str | Path, metadatos: dict | None = None) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        datos = {
            "gamma": self.gamma,
            "iteraciones": self.iteraciones,
            "metadatos": metadatos or {},
            "politica": self.politica.tolist(),
            "valores": np.round(self.valores, 4).tolist(),
        }
        ruta.write_text(json.dumps(datos), encoding="utf-8")
        return ruta

    @classmethod
    def cargar(cls, ruta: str | Path) -> "PoliticaReferencia":
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe la política de referencia en {ruta}.")
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            politica = np.asarray(datos["politica"], dtype=int)
            valores = np.asarray(datos["valores"], dtype=float)
            if politica.shape != (N_H, N_Q, N_S) or valores.shape != politica.shape:
                raise ValueError("forma inválida")
            if politica.min() < 0 or politica.max() >= N_A:
                raise ValueError("acción fuera de rango")
            return cls(politica, valores, int(datos["iteraciones"]), float(datos["gamma"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Archivo de política de referencia inválido: {ruta} ({exc})") from exc


def estimar_modelo(
    precios_train: "np.ndarray | Sequence[float]",
    umbrales_precio: "np.ndarray | Sequence[float] | None" = None,
) -> ModeloEmpirico:
    """Estima recompensas medias y transiciones de cuartil con los días de entrenamiento.

    ``umbrales_precio`` se interpreta igual que en ``BatteryEnv`` (3 valores fijos, una
    fila por hora, o ``None`` para la ventana móvil causal), así el cuartil de cada hora
    es exactamente el que observa el agente.
    """
    precios = np.asarray(precios_train, dtype=float).ravel()
    if precios.size == 0 or precios.size % N_H:
        raise ValueError("precios_train debe contener días completos (múltiplo de 24).")

    P = precios.reshape(-1, N_H)
    Q = BatteryEnv(precios, umbrales_precio=umbrales_precio).cuartiles_diarios  # (dias, 24)

    conteos = np.zeros((N_H, N_Q))
    suma = np.zeros((N_H, N_Q))
    for h in range(N_H):
        np.add.at(conteos[h], Q[:, h], 1)
        np.add.at(suma[h], Q[:, h], P[:, h])
    media_hora = P.mean(axis=0)
    # (h, q) nunca observado: se usa el precio medio de esa hora (el estado casi no ocurre).
    precio_medio = np.where(conteos > 0, suma / np.maximum(conteos, 1), media_hora[:, None])

    transicion = np.zeros((N_H - 1, N_Q, N_Q))
    for h in range(N_H - 1):
        np.add.at(transicion[h], (Q[:, h], Q[:, h + 1]), 1)
        marginal = np.bincount(Q[:, h + 1], minlength=N_Q).astype(float)
        marginal /= marginal.sum()
        filas = transicion[h].sum(axis=1, keepdims=True)
        transicion[h] = np.where(filas > 0, transicion[h] / np.maximum(filas, 1), marginal)
    return ModeloEmpirico(precio_medio, transicion, conteos)


def iterar_valores(
    modelo: ModeloEmpirico,
    energia_por_nivel_kwh: float = 1.0,
    gamma: float = 0.99,
    penalizacion_invalida: float = -1000.0,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> PoliticaReferencia:
    """Iteración de valores tabular sobre el modelo empírico."""
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma debe estar en [0, 1].")
    s = np.arange(N_S)
    s_cargar = np.minimum(s + 1, N_S - 1)
    s_descargar = np.maximum(s - 1, 0)
    valida_cargar = s < N_S - 1
    valida_descargar = s > 0
    monto = modelo.precio_medio * energia_por_nivel_kwh  # (24, 4)

    V = np.zeros((N_H, N_Q, N_S))
    for it in range(1, max_iter + 1):
        # Valor esperado del siguiente paso para cada (h, q, s'): Σ_q' T · V(h+1, q', s').
        futuro = np.zeros((N_H, N_Q, N_S))
        futuro[:-1] = np.einsum("hqr,hrs->hqs", modelo.transicion, V[1:])
        Qsa = np.empty((N_H, N_Q, N_S, N_A))
        Qsa[..., Accion.MANTENER] = gamma * futuro
        Qsa[..., Accion.CARGAR] = np.where(
            valida_cargar, -monto[..., None] + gamma * futuro[..., s_cargar],
            penalizacion_invalida + gamma * futuro,
        )
        Qsa[..., Accion.DESCARGAR] = np.where(
            valida_descargar, monto[..., None] + gamma * futuro[..., s_descargar],
            penalizacion_invalida + gamma * futuro,
        )
        V_nuevo = Qsa.max(axis=-1)
        delta = float(np.abs(V_nuevo - V).max())
        V = V_nuevo
        if delta < tol:
            break
    politica = Qsa.argmax(axis=-1)  # empate -> menor índice (MANTENER)
    return PoliticaReferencia(politica.astype(int), V, it, float(gamma))


def value_iteration_policy(
    precios_train: "np.ndarray | Sequence[float]",
    umbrales_precio: "np.ndarray | Sequence[float] | None" = None,
    energia_por_nivel_kwh: float = 1.0,
    gamma: float = 0.99,
    penalizacion_invalida: float = -1000.0,
    tol: float = 1e-9,
) -> PoliticaReferencia:
    """Estima el modelo empírico con los días de entrenamiento y lo resuelve con VI."""
    modelo = estimar_modelo(precios_train, umbrales_precio)
    return iterar_valores(modelo, energia_por_nivel_kwh, gamma, penalizacion_invalida, tol)
