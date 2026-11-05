"""Descarga de datos del mercado de energía (XM) con plan B a CSV local.

Fuente principal: librería oficial pydataxm (https://github.com/EquipoAnaliticaXM/API_XM).
Plan B (requisito de las primeras 48 h): si la API falla, se cargan históricos desde
data/raw/*.csv.

Uso:
    python -m src.data.download_xm --inicio 2024-01-01 --fin 2024-03-31
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Métrica de precio de bolsa en la API de XM (ajustar según catálogo de pydataxm).
METRIC_PRECIO_BOLSA = "PrecBolsNaci"  # Precio de bolsa nacional
ENTITY_SISTEMA = "Sistema"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def descargar_precio_bolsa(inicio: date, fin: date) -> pd.DataFrame:
    """Descarga el precio de bolsa desde la API de XM usando pydataxm.

    Devuelve un DataFrame. Si la API o la librería no están disponibles, lanza
    una excepción que el llamador puede capturar para activar el plan B.
    """
    from pydataxm.pydataxm import ReadDB  # import diferido: evita fallo si no está instalada

    api = ReadDB()
    df = api.request_data(METRIC_PRECIO_BOLSA, ENTITY_SISTEMA, inicio, fin)
    return df


def cargar_plan_b(nombre: str = "precio_bolsa.csv") -> pd.DataFrame:
    """Carga históricos descargados manualmente cuando la API falla."""
    path = RAW_DIR / nombre
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el histórico de plan B en {path}. "
            "Descárgalo manualmente desde XM/Sinergox y colócalo ahí."
        )
    return pd.read_csv(path)


def obtener_precio_bolsa(inicio: date, fin: date) -> pd.DataFrame:
    """Intenta la API; si falla, cae al plan B (CSV local)."""
    try:
        df = descargar_precio_bolsa(inicio, fin)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DIR / "precio_bolsa.csv", index=False)
        print(f"[OK] Datos descargados de XM: {len(df)} filas.")
        return df
    except Exception as exc:  # noqa: BLE001 - queremos el plan B ante cualquier fallo
        print(f"[WARN] API de XM no disponible ({exc}). Usando plan B local.")
        return cargar_plan_b()


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga precio de bolsa de XM.")
    parser.add_argument("--inicio", type=_parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument("--fin", type=_parse_date, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    df = obtener_precio_bolsa(args.inicio, args.fin)
    print(df.head())


if __name__ == "__main__":
    main()
