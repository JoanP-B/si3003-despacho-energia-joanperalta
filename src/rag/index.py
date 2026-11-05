"""Parte 2 — Ingestión del corpus regulatorio e índice vectorial (RAG).

Lee los PDFs de corpus_regulacion/, los trocea, calcula embeddings y construye un índice
FAISS/Chroma. Esqueleto; implementación en los días 9-11 nov.
"""
from __future__ import annotations

from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus_regulacion"


def construir_indice():
    """Trocea PDFs, calcula embeddings y guarda el índice. Por implementar."""
    raise NotImplementedError


def buscar(query: str, k: int = 4):
    """Recupera los k fragmentos más relevantes con su fuente. Por implementar."""
    raise NotImplementedError
