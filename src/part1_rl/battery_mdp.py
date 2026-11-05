"""Parte 1 (técnica clásica) — MDP del despacho de batería.

IMPLEMENTACIÓN PROPIA DEL EQUIPO (requisito 2 del enunciado): el MDP y los algoritmos
de solución (iteración de valor / Q-learning) se escriben desde cero; no se invoca una
librería que ya los resuelva.

Esqueleto: define el entorno y deja los algoritmos para implementar en los días 6-8 nov.

Estado:   (hora_del_dia, nivel_de_carga_discretizado, banda_de_precio)
Acciones: CARGAR, DESCARGAR, MANTENER
Recompensa: ahorro monetario = -costo de compra de energía en esa hora
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Accion(IntEnum):
    MANTENER = 0
    CARGAR = 1
    DESCARGAR = 2


@dataclass
class ParametrosBateria:
    capacidad_kwh: float = 10.0       # energía máxima almacenable
    potencia_kw: float = 2.5          # carga/descarga por paso (1 h)
    eficiencia: float = 0.95          # rendimiento round-trip
    soc_inicial: float = 0.5          # estado de carga inicial (fracción 0..1)


class BatteryMDP:
    """Entorno MDP del despacho de batería frente al precio de bolsa.

    TODO (días 6-8 nov):
      - discretizar el estado de carga (SOC) y las bandas de precio
      - definir transición determinista según la acción y los límites físicos
      - definir recompensa = ahorro respecto a no operar la batería
      - implementar value_iteration() y q_learning()
      - comparar contra política ingenua y de referencia (métrica de ahorro)
    """

    def __init__(self, precios: list[float], params: ParametrosBateria | None = None):
        self.precios = precios
        self.params = params or ParametrosBateria()
        raise NotImplementedError("Entorno MDP por implementar (hito Vie 6 - Dom 8 nov).")

    def reset(self):
        raise NotImplementedError

    def step(self, accion: Accion):
        """Devuelve (nuevo_estado, recompensa, terminado, info)."""
        raise NotImplementedError


def value_iteration(mdp: "BatteryMDP", gamma: float = 0.99, theta: float = 1e-6):
    """Iteración de valor (implementación propia). Por completar."""
    raise NotImplementedError


def q_learning(mdp: "BatteryMDP", episodios: int = 1000, alpha: float = 0.1, gamma: float = 0.99):
    """Q-learning tabular (implementación propia). Por completar."""
    raise NotImplementedError
