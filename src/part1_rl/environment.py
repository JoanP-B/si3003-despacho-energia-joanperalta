"""Parte 1 (RL clásico) — Entorno MDP del despacho de batería (BESS).

IMPLEMENTACIÓN PROPIA (requisito del enunciado): el MDP se escribe desde cero,
sin Gymnasium ni Stable-Baselines.

El problema NO es especulación financiera, es **desplazamiento de demanda**
(load shifting): la batería absorbe energía de la red cuando el precio es bajo
y cubre demanda propia (evitando comprar a la red) cuando el precio es alto,
respetando las restricciones físicas reales del sistema BESS.

Episodio: un día real del historial (24 pasos de 1 hora).

Línea Base justa (``naive_policy``): carga en las horas 01-04 y descarga en las 18-21,
usando los 4 escalones de SoC, sin mirar el precio.

Estado (tupla discreta para evitar explosión de estados):
    (hora_del_dia, cuartil_de_precio, nivel_soc)
      - hora_del_dia:      int 0..23
      - cuartil_de_precio: int 0..3  (cuantiles 25/50/75 del historial)
      - nivel_soc:         int 0..4  (0%, 25%, 50%, 75%, 100%)

Acciones:
    0: MANTENER, 1: CARGAR, 2: DESCARGAR

Recompensa (flujo de caja neto en COP, costo evitado):
    CARGAR    -> -precio_actual * energia_por_nivel_kwh  (compra a la red)
    DESCARGAR -> +precio_actual * energia_por_nivel_kwh  (costo evitado)
    MANTENER  ->  0
    acción física inválida -> penalizacion_invalida (-1000), estado sin cambio
"""
from __future__ import annotations

from enum import IntEnum
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

# Tipo del estado discreto expuesto por el entorno.
State = Tuple[int, int, int]


class Accion(IntEnum):
    """Acciones físicas disponibles sobre la batería."""

    MANTENER = 0
    CARGAR = 1
    DESCARGAR = 2


class BatteryEnv:
    """Entorno MDP del despacho de una batería frente al precio de bolsa (XM).

    Parameters
    ----------
    precios:
        Precios horarios consecutivos (COP/kWh), empezando a las 00:00.
        La longitud debe ser múltiplo de 24 (días completos).
    soc_inicial:
        Nivel de carga discreto al inicio de cada día (0..4). Por defecto 0:
        la batería arranca vacía, así toda energía descargada fue comprada
        antes y el ahorro no se infla con energía "gratis".
    penalizacion_invalida:
        Recompensa por intentar una acción físicamente imposible
        (cargar al 100% o descargar al 0%).
    energia_por_nivel_kwh:
        Energía que representa un nivel de SoC (capacidad_kwh / 4). Con 1.0
        la recompensa es exactamente ``±precio_actual``.
    umbrales_precio:
        Umbrales (p25, p50, p75) para discretizar el precio. Si es ``None``
        se calculan de ``precios``. Para un entorno de prueba, pasar los
        umbrales del entorno de entrenamiento evita fuga de información.
    aleatorio:
        Si es ``True``, ``reset()`` elige un día al azar (entrenamiento).
        Si es ``False``, recorre los días en orden (evaluación).
    semilla:
        Semilla del generador aleatorio.
    """

    N_ACCIONES: int = 3
    N_NIVELES_SOC: int = 5  # 0..4 -> 0%, 25%, 50%, 75%, 100%
    N_CUARTILES: int = 4
    N_HORAS: int = 24

    def __init__(
        self,
        precios: "np.ndarray | pd.Series | Sequence[float]",
        soc_inicial: int = 0,
        penalizacion_invalida: float = -1000.0,
        energia_por_nivel_kwh: float = 1.0,
        umbrales_precio: "Sequence[float] | None" = None,
        aleatorio: bool = True,
        semilla: int | None = None,
    ) -> None:
        precios_arr = np.asarray(precios, dtype=float).ravel()
        if precios_arr.size == 0 or precios_arr.size % self.N_HORAS != 0:
            raise ValueError(
                "La serie de precios debe contener días completos (múltiplo de "
                f"{self.N_HORAS}); se recibieron {precios_arr.size} valores."
            )
        if not np.isfinite(precios_arr).all():
            raise ValueError("La serie de precios contiene NaN o infinitos.")
        if not (0 <= soc_inicial <= self.N_NIVELES_SOC - 1):
            raise ValueError(
                f"soc_inicial debe estar en [0, {self.N_NIVELES_SOC - 1}]; "
                f"se recibió {soc_inicial}."
            )
        if energia_por_nivel_kwh <= 0:
            raise ValueError("energia_por_nivel_kwh debe ser > 0.")

        # Matriz (n_dias, 24): cada fila es un episodio.
        self.precios_diarios: np.ndarray = precios_arr.reshape(-1, self.N_HORAS)
        self.soc_inicial: int = int(soc_inicial)
        self.penalizacion_invalida: float = float(penalizacion_invalida)
        self.energia_por_nivel_kwh: float = float(energia_por_nivel_kwh)
        self.aleatorio: bool = bool(aleatorio)
        self._rng = np.random.default_rng(semilla)

        if umbrales_precio is None:
            umbrales = np.quantile(precios_arr, [0.25, 0.50, 0.75])
        else:
            umbrales = np.asarray(umbrales_precio, dtype=float).ravel()
            if umbrales.size != self.N_CUARTILES - 1 or np.any(np.diff(umbrales) < 0):
                raise ValueError(
                    "umbrales_precio debe tener 3 valores no decrecientes (p25, p50, p75)."
                )
        self.umbrales_precio: np.ndarray = umbrales

        # Estado interno mutable.
        self._dia_siguiente: int = 0  # cursor para modo secuencial
        self.dia_actual: int = 0
        self._hora: int = 0
        self._soc: int = self.soc_inicial
        self.estado: State = (0, 0, self.soc_inicial)

    # ------------------------------------------------------------------ #
    # Propiedades y discretización
    # ------------------------------------------------------------------ #
    @property
    def n_dias(self) -> int:
        """Número de días (episodios distintos) disponibles."""
        return int(self.precios_diarios.shape[0])

    def cuartil_precio(self, precio: float) -> int:
        """Mapea un precio continuo a su cuartil discreto 0..3."""
        return int(np.searchsorted(self.umbrales_precio, precio, side="right"))

    def _precio_actual(self) -> float:
        return float(self.precios_diarios[self.dia_actual, self._hora])

    def _construir_estado(self) -> State:
        return (self._hora, self.cuartil_precio(self._precio_actual()), self._soc)

    # ------------------------------------------------------------------ #
    # API del MDP
    # ------------------------------------------------------------------ #
    def reset(self, dia: int | None = None) -> State:
        """Inicia un episodio (un día) y devuelve el estado de las 00:00.

        Si ``dia`` es ``None`` se elige al azar (modo aleatorio) o el
        siguiente día en orden (modo secuencial).
        """
        if dia is None:
            if self.aleatorio:
                dia = int(self._rng.integers(self.n_dias))
            else:
                dia = self._dia_siguiente
                self._dia_siguiente = (self._dia_siguiente + 1) % self.n_dias
        if not (0 <= dia < self.n_dias):
            raise IndexError(f"dia fuera de rango: {dia} (hay {self.n_dias} días).")

        self.dia_actual = int(dia)
        self._hora = 0
        self._soc = self.soc_inicial
        self.estado = self._construir_estado()
        return self.estado

    def step(self, accion: int) -> Tuple[State, float, bool, dict]:
        """Aplica una acción en la hora actual y avanza una hora.

        Returns
        -------
        (nuevo_estado, recompensa, terminado, info)
            ``info["flujo_caja_cop"]`` es el flujo de caja real del paso
            (sin la penalización artificial), usado para medir el ahorro.
        """
        if self._hora >= self.N_HORAS:
            raise RuntimeError("El episodio terminó; llame a reset() antes de step().")
        try:
            accion = Accion(int(accion))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Acción inválida {accion!r}; debe ser 0, 1 o 2.") from exc

        precio = self._precio_actual()
        hora = self._hora
        monto = precio * self.energia_por_nivel_kwh
        valida = True
        flujo_caja = 0.0

        if accion == Accion.CARGAR:
            if self._soc >= self.N_NIVELES_SOC - 1:  # ya está al 100%
                valida = False
            else:
                self._soc += 1
                flujo_caja = -monto  # gasto por comprar a la red
        elif accion == Accion.DESCARGAR:
            if self._soc <= 0:  # batería vacía
                valida = False
            else:
                self._soc -= 1
                flujo_caja = +monto  # costo evitado
        # MANTENER: flujo_caja = 0

        recompensa = flujo_caja if valida else self.penalizacion_invalida

        self._hora += 1
        terminado = self._hora >= self.N_HORAS
        if not terminado:
            self.estado = self._construir_estado()
        else:
            # Estado terminal: el agente ignora max Q(s') cuando terminado=True.
            self.estado = (self.N_HORAS - 1, self.estado[1], self._soc)

        info = {
            "dia": self.dia_actual,
            "hora": hora,
            "precio": precio,
            "accion": int(accion),
            "valida": valida,
            "soc": self._soc,
            "flujo_caja_cop": flujo_caja,
        }
        return self.estado, float(recompensa), bool(terminado), info

    # ------------------------------------------------------------------ #
    # Línea base (baseline) ingenua y justa
    # ------------------------------------------------------------------ #
    HORAS_CARGA_BASE: Tuple[int, ...] = (1, 2, 3, 4)
    HORAS_DESCARGA_BASE: Tuple[int, ...] = (18, 19, 20, 21)

    @staticmethod
    def naive_policy(state: State) -> int:
        """Línea Base justa (ciega al precio) que usa los 4 escalones de la batería.

        Carga en las horas 01-04 mientras ``nivel_soc < 4``, descarga en las horas 18-21
        mientras ``nivel_soc > 0`` y mantiene el resto del día. Nunca intenta una acción
        física inválida.
        """
        hora, _, soc = state
        if hora in BatteryEnv.HORAS_CARGA_BASE and soc < BatteryEnv.N_NIVELES_SOC - 1:
            return int(Accion.CARGAR)
        if hora in BatteryEnv.HORAS_DESCARGA_BASE and soc > 0:
            return int(Accion.DESCARGAR)
        return int(Accion.MANTENER)
