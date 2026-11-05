"""Parte 2 — Herramientas (tool calling) del agente LLM.

El enunciado exige al menos dos herramientas. Esqueleto de firmas; implementación en
los días 9-11 nov.
"""
from __future__ import annotations

from datetime import date


def simulate(policy: str, inicio: date, fin: date) -> dict:
    """Simula una política de despacho sobre un rango de fechas.

    Devuelve ahorro acumulado y la secuencia de decisiones. Conecta con src/part1_rl.
    TODO: implementar usando BatteryMDP y la política entrenada.
    """
    raise NotImplementedError


def search_docs(query: str, k: int = 4) -> list[dict]:
    """Recupera fragmentos del corpus regulatorio (CREG/XM) para el RAG.

    Devuelve lista de {texto, fuente}. Conecta con src/rag.
    TODO: implementar sobre el índice FAISS/Chroma.
    """
    raise NotImplementedError


def get_price(dia: date) -> float:
    """Consulta el precio de bolsa de un día. Conecta con src/data/download_xm.py."""
    raise NotImplementedError
