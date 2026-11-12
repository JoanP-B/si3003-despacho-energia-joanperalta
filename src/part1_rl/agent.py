"""Parte 1 (RL clásico) — Agente Q-Learning tabular (implementación propia).

Actualización clásica:
    Q(s,a) <- Q(s,a) + alpha * [ R + gamma * max_a' Q(s',a') - Q(s,a) ]

Exploración Epsilon-Greedy con decaimiento lineal de epsilon a lo largo del
entrenamiento. Sin librerías de alto nivel (ni Gymnasium ni Stable-Baselines).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

State = Tuple[int, int, int]


class QLearningAgent:
    """Agente Q-Learning tabular con política epsilon-greedy.

    Parameters
    ----------
    n_acciones:
        Número de acciones discretas disponibles.
    alpha:
        Tasa de aprendizaje (0, 1].
    gamma:
        Factor de descuento [0, 1].
    epsilon_inicial:
        Probabilidad de exploración al inicio del entrenamiento.
    epsilon_final:
        Probabilidad de exploración mínima tras el decaimiento.
    episodios_decaimiento:
        Número de episodios a lo largo de los cuales epsilon decae
        linealmente de ``epsilon_inicial`` a ``epsilon_final``.
    semilla:
        Semilla del generador aleatorio para reproducibilidad.
    """

    def __init__(
        self,
        n_acciones: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_inicial: float = 1.0,
        epsilon_final: float = 0.05,
        episodios_decaimiento: int = 500,
        semilla: int | None = 42,
    ) -> None:
        if n_acciones < 1:
            raise ValueError("n_acciones debe ser >= 1.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha debe estar en (0, 1].")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma debe estar en [0, 1].")
        if not (0.0 <= epsilon_final <= epsilon_inicial <= 1.0):
            raise ValueError(
                "Se requiere 0 <= epsilon_final <= epsilon_inicial <= 1."
            )
        if episodios_decaimiento < 1:
            raise ValueError("episodios_decaimiento debe ser >= 1.")

        self.n_acciones = int(n_acciones)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_inicial = float(epsilon_inicial)
        self.epsilon_final = float(epsilon_final)
        self.episodios_decaimiento = int(episodios_decaimiento)
        self.epsilon = float(epsilon_inicial)

        self._rng = np.random.default_rng(semilla)

        # Q-Table dispersa: se crea una fila de ceros al visitar un estado nuevo.
        self.q_table: Dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_acciones, dtype=float)
        )

    # ------------------------------------------------------------------ #
    # Política
    # ------------------------------------------------------------------ #
    def seleccionar_accion(self, estado: State, explorar: bool = True) -> int:
        """Devuelve una acción mediante epsilon-greedy.

        Con ``explorar=False`` actúa de forma greedy (para evaluación).
        Los empates entre valores Q máximos se rompen aleatoriamente.
        """
        if explorar and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_acciones))
        return self._greedy(estado)

    def _greedy(self, estado: State) -> int:
        q = self.q_table[estado]
        maximo = q.max()
        mejores = np.flatnonzero(q == maximo)
        return int(self._rng.choice(mejores))

    # ------------------------------------------------------------------ #
    # Aprendizaje
    # ------------------------------------------------------------------ #
    def actualizar(
        self,
        estado: State,
        accion: int,
        recompensa: float,
        estado_sig: State,
        terminado: bool,
    ) -> None:
        """Aplica la regla de actualización tabular de Q-Learning."""
        q_sa = self.q_table[estado][accion]
        max_q_sig = 0.0 if terminado else float(self.q_table[estado_sig].max())
        objetivo = recompensa + self.gamma * max_q_sig
        self.q_table[estado][accion] = q_sa + self.alpha * (objetivo - q_sa)

    def decaer_epsilon(self, episodio: int) -> None:
        """Decaimiento lineal de epsilon según el episodio actual (0-indexado)."""
        fraccion = min(1.0, episodio / self.episodios_decaimiento)
        self.epsilon = self.epsilon_inicial + fraccion * (
            self.epsilon_final - self.epsilon_inicial
        )

    # ------------------------------------------------------------------ #
    # Entrenamiento de alto nivel
    # ------------------------------------------------------------------ #
    def entrenar(self, env, episodios: int = 500) -> list[float]:
        """Entrena contra un entorno con API reset()/step() y devuelve el
        retorno (suma de recompensas) por episodio.

        El entorno debe exponer ``reset() -> estado`` y
        ``step(accion) -> (estado, recompensa, terminado, info)``.
        """
        if episodios < 1:
            raise ValueError("episodios debe ser >= 1.")

        historial_retornos: list[float] = []
        for ep in range(episodios):
            estado = env.reset()
            terminado = False
            retorno = 0.0
            while not terminado:
                accion = self.seleccionar_accion(estado, explorar=True)
                estado_sig, recompensa, terminado, _ = env.step(accion)
                self.actualizar(estado, accion, recompensa, estado_sig, terminado)
                estado = estado_sig
                retorno += recompensa
            self.decaer_epsilon(ep)
            historial_retornos.append(retorno)
        return historial_retornos

    # ------------------------------------------------------------------ #
    # Persistencia (JSON: legible y sin riesgos de deserializar pickle)
    # ------------------------------------------------------------------ #
    def guardar(self, ruta: str | Path) -> Path:
        """Guarda hiperparámetros y Q-Table en un archivo JSON."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        datos = {
            "n_acciones": self.n_acciones,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_inicial": self.epsilon_inicial,
            "epsilon_final": self.epsilon_final,
            "episodios_decaimiento": self.episodios_decaimiento,
            "epsilon": self.epsilon,
            "q_table": {
                ",".join(map(str, estado)): valores.tolist()
                for estado, valores in self.q_table.items()
            },
        }
        ruta.write_text(json.dumps(datos, indent=1), encoding="utf-8")
        return ruta

    @classmethod
    def cargar(cls, ruta: str | Path, semilla: int | None = 42) -> "QLearningAgent":
        """Reconstruye un agente entrenado desde un JSON creado con ``guardar``."""
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe la Q-Table en {ruta}.")
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            agente = cls(
                n_acciones=datos["n_acciones"],
                alpha=datos["alpha"],
                gamma=datos["gamma"],
                epsilon_inicial=datos["epsilon_inicial"],
                epsilon_final=datos["epsilon_final"],
                episodios_decaimiento=datos["episodios_decaimiento"],
                semilla=semilla,
            )
            agente.epsilon = float(datos["epsilon"])
            for clave, valores in datos["q_table"].items():
                estado = tuple(int(x) for x in clave.split(","))
                fila = np.asarray(valores, dtype=float)
                if fila.shape != (agente.n_acciones,):
                    raise ValueError(f"Fila Q con forma inválida para el estado {clave}.")
                agente.q_table[estado] = fila  # type: ignore[index]
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Archivo de Q-Table corrupto o incompleto: {ruta}") from exc
        return agente
