"""Parte 2 — Índice vectorial (RAG) del corpus regulatorio CREG.

Ingesta:
    - PDFs de ``corpus_regulacion/`` con ``PyPDFLoader`` (una página = un documento).
    - HTML (``.htm``/``.html``) del gestor normativo de la CREG. Se acepta porque algunos
      PDFs oficiales son escaneos sin capa de texto (p. ej. Creg098-2019.pdf) y no aportan
      contenido al índice.
Troceo:    ``RecursiveCharacterTextSplitter`` (chunk_size=1000, chunk_overlap=200).
Embeddings: ``HuggingFaceEmbeddings`` locales (por defecto ``all-MiniLM-L6-v2``, sin costo).
Índice:    FAISS guardado en ``data/processed/faiss_index``.

Construir el índice (desde la raíz del repositorio):
    python -m src.rag.vectorstore
    python -m src.rag.vectorstore --buscar "almacenamiento con baterías"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # imports pesados (torch, faiss) solo cuando se usan
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

RAIZ = Path(__file__).resolve().parents[2]
CORPUS_DIR = RAIZ / "corpus_regulacion"
INDEX_DIR = RAIZ / "data" / "processed" / "faiss_index"
META_FILE = "meta.json"

# Configurable por entorno (variable EMBEDDING_MODEL). En una prueba con 6 consultas
# en español sobre este corpus, all-MiniLM-L6-v2 recuperó 16/18 fragmentos de la fuente
# correcta y paraphrase-multilingual-MiniLM-L12-v2, 15/18.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHARS_PAGINA = 20  # por debajo se considera página sin texto (escaneo)


class RAGError(RuntimeError):
    """Error de construcción o carga del índice vectorial."""


# --------------------------------------------------------------------------- #
# Carga de documentos
# --------------------------------------------------------------------------- #
class _ExtractorTexto(HTMLParser):
    """Extrae texto visible de un HTML (ignora script/style), con saltos por bloque."""

    _BLOQUES = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.partes: list[str] = []
        self._ignorar = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._ignorar += 1
        elif tag in self._BLOQUES:
            self.partes.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._ignorar:
            self._ignorar -= 1
        elif tag in self._BLOQUES:
            self.partes.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignorar:
            self.partes.append(data)


def _leer_html(ruta: Path) -> str:
    crudo = ruta.read_bytes()
    for codificacion in ("utf-8", "cp1252", "latin-1"):
        try:
            html = crudo.decode(codificacion)
            break
        except UnicodeDecodeError:
            continue
    extractor = _ExtractorTexto()
    extractor.feed(html)
    texto = unescape("".join(extractor.partes)).replace("\xa0", " ")
    texto = re.sub(r"[ \t]+", " ", texto)
    return re.sub(r"\n\s*\n+", "\n\n", texto).strip()


def cargar_documentos(corpus_dir: Path = CORPUS_DIR) -> list["Document"]:
    """Carga PDFs y HTML del corpus. Avisa y omite archivos/páginas sin texto."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document

    if not corpus_dir.is_dir():
        raise RAGError(f"No existe la carpeta del corpus: {corpus_dir}")
    archivos = sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".htm", ".html"}
    )
    if not archivos:
        raise RAGError(f"No hay PDFs ni HTML en {corpus_dir}. Agrega resoluciones CREG.")

    documentos: list[Document] = []
    for ruta in archivos:
        try:
            if ruta.suffix.lower() == ".pdf":
                paginas = PyPDFLoader(str(ruta)).load()
                utiles = [p for p in paginas if len(p.page_content.strip()) >= MIN_CHARS_PAGINA]
                for p in utiles:
                    p.metadata = {"fuente": ruta.name, "pagina": int(p.metadata.get("page", 0)) + 1}
                if not utiles:
                    print(f"[WARN] {ruta.name}: sin texto extraíble (¿PDF escaneado?). Omitido.")
                    continue
                print(f"[OK] {ruta.name}: {len(utiles)}/{len(paginas)} páginas con texto.")
                documentos.extend(utiles)
            else:
                texto = _leer_html(ruta)
                if len(texto) < MIN_CHARS_PAGINA:
                    print(f"[WARN] {ruta.name}: HTML sin texto. Omitido.")
                    continue
                print(f"[OK] {ruta.name}: {len(texto):,} caracteres.")
                documentos.append(
                    Document(page_content=texto, metadata={"fuente": ruta.name, "pagina": None})
                )
        except Exception as exc:  # noqa: BLE001 - un archivo dañado no detiene la ingesta
            print(f"[WARN] No se pudo leer {ruta.name}: {exc}")

    if not documentos:
        raise RAGError("Ningún archivo del corpus tiene texto extraíble.")
    return documentos


def trocear(documentos: list["Document"]) -> list["Document"]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documentos)


# --------------------------------------------------------------------------- #
# Embeddings e índice
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _embeddings(modelo: str) -> "Embeddings":
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RAGError(
            "Falta langchain-huggingface/sentence-transformers. "
            "Ejecuta: python -m pip install -r requirements.txt"
        ) from exc
    return HuggingFaceEmbeddings(
        model_name=modelo,
        encode_kwargs={"normalize_embeddings": True},  # similitud coseno
    )


def construir_indice(
    corpus_dir: Path = CORPUS_DIR,
    index_dir: Path = INDEX_DIR,
    modelo: str = EMBEDDING_MODEL,
) -> int:
    """Ingiere el corpus, crea el índice FAISS y lo guarda. Devuelve el nº de fragmentos."""
    from langchain_community.vectorstores import FAISS

    fragmentos = trocear(cargar_documentos(corpus_dir))
    print(f"[..] Calculando embeddings de {len(fragmentos)} fragmentos con {modelo}…")
    indice = FAISS.from_documents(fragmentos, _embeddings(modelo))

    index_dir.mkdir(parents=True, exist_ok=True)
    indice.save_local(str(index_dir))
    meta = {
        "modelo_embeddings": modelo,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "n_fragmentos": len(fragmentos),
        "fuentes": sorted({d.metadata["fuente"] for d in fragmentos}),
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=1, ensure_ascii=False), "utf-8")
    cargar_indice.cache_clear()
    print(f"[OK] Índice guardado en {index_dir}")
    return len(fragmentos)


@lru_cache(maxsize=1)
def cargar_indice(index_dir: Path = INDEX_DIR) -> "FAISS":
    """Carga (una sola vez por proceso) el índice FAISS guardado."""
    if not (index_dir / "index.faiss").exists():
        raise RAGError(
            f"No existe el índice FAISS en {index_dir}. "
            "Constrúyelo con: python -m src.rag.vectorstore"
        )
    try:
        meta = json.loads((index_dir / META_FILE).read_text("utf-8"))
        modelo = meta["modelo_embeddings"]
    except (OSError, KeyError, json.JSONDecodeError):
        modelo = EMBEDDING_MODEL  # índice sin metadatos: se asume el modelo por defecto

    from langchain_community.vectorstores import FAISS

    try:
        # El docstore de LangChain se serializa con pickle. Es seguro solo porque el
        # índice lo genera este mismo script en local; no cargar índices de terceros.
        return FAISS.load_local(
            str(index_dir), _embeddings(modelo), allow_dangerous_deserialization=True
        )
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"No se pudo cargar el índice FAISS de {index_dir}: {exc}") from exc


def buscar(query: str, k: int = 3, index_dir: Path = INDEX_DIR) -> list[dict[str, Any]]:
    """Devuelve los k fragmentos más relevantes: {texto, fuente, pagina, score}."""
    consulta = (query or "").strip()
    if not consulta:
        raise ValueError("La consulta está vacía.")
    if k < 1:
        raise ValueError("k debe ser >= 1.")
    resultados = cargar_indice(index_dir).similarity_search_with_score(consulta, k=k)
    return [
        {
            "texto": doc.page_content,
            "fuente": doc.metadata.get("fuente", "desconocida"),
            "pagina": doc.metadata.get("pagina"),
            "score": float(distancia),  # distancia L2: menor = más relevante
        }
        for doc, distancia in resultados
    ]


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Construye o consulta el índice FAISS.")
    parser.add_argument("--buscar", metavar="CONSULTA", help="Consulta el índice existente.")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if args.buscar:
            for i, r in enumerate(buscar(args.buscar, args.k), 1):
                pagina = f", pág. {r['pagina']}" if r["pagina"] else ""
                print(f"\n[{i}] {r['fuente']}{pagina} (distancia {r['score']:.3f})")
                print(r["texto"][:400])
        else:
            construir_indice()
    except (RAGError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    raise SystemExit(_main(sys.argv[1:]))
