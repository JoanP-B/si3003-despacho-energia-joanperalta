"""Descarga del precio de bolsa nacional de XM con respaldo local automático.

Fuente principal: librería oficial pydataxm (https://github.com/EquipoAnaliticaXM/API_XM),
métrica ``PrecBolsNaci`` (COP/kWh, una columna por hora del día).

Plan B: cada descarga exitosa se guarda en ``data/raw/xm_precio_bolsa.csv``. Si la API falla
(sin red, caída del servicio, certificado), se usa ese CSV filtrado al rango pedido.

Uso:
    python -m src.data.download_xm --inicio 2025-11-10 --fin 2026-11-09
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RAW_DIR = RAIZ / "data" / "raw"
PROCESSED_DIR = RAIZ / "data" / "processed"
CSV_RESPALDO = RAW_DIR / "xm_precio_bolsa.csv"

METRIC_PRECIO_BOLSA = "PrecBolsNaci"  # Precio de bolsa nacional (COP/kWh)
ENTITY_SISTEMA = "Sistema"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _usar_certificados_del_sistema() -> None:
    """En Windows, Python puede no confiar en la CA de servapibi.xm.com.co.

    ``truststore`` (opcional) hace que ``ssl`` use el almacén de certificados del sistema
    operativo. Si no está instalado, no se hace nada.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


def descargar_precio_bolsa(inicio: date, fin: date) -> pd.DataFrame:
    """Descarga ``PrecBolsNaci`` en formato ancho (``Date``, ``Values_Hour01..24``).

    Lanza una excepción si la API o la librería no están disponibles.
    """
    from pydataxm.pydataxm import ReadDB  # import diferido: evita fallo si no está instalada

    _usar_certificados_del_sistema()
    df = ReadDB().request_data(METRIC_PRECIO_BOLSA, ENTITY_SISTEMA, inicio, fin)
    if df is None or len(df) == 0:
        raise RuntimeError("La API de XM no devolvió filas para el rango pedido.")
    return df


def cargar_plan_b(ruta: Path = CSV_RESPALDO) -> pd.DataFrame:
    """Carga el respaldo local guardado en una descarga anterior."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el respaldo de XM en {ruta}. Ejecuta "
            "`python -m src.data.download_xm --inicio AAAA-MM-DD --fin AAAA-MM-DD` con conexión."
        )
    return pd.read_csv(ruta)


def a_serie_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a columnas ``(timestamp, precio_cop_kwh)`` y conserva solo días completos.

    Acepta el formato ancho de pydataxm (``Values_Hour01`` = periodo 00:00-01:00) o un CSV
    que ya tenga ``timestamp`` y ``precio_cop_kwh``.
    """
    columnas_hora = sorted(c for c in df.columns if str(c).startswith("Values_Hour"))
    if columnas_hora:
        largo = df.melt(id_vars="Date", value_vars=columnas_hora,
                        var_name="col", value_name="precio_cop_kwh")
        hora = largo["col"].str[-2:].astype(int) - 1
        largo["timestamp"] = pd.to_datetime(largo["Date"]) + pd.to_timedelta(hora, unit="h")
    elif {"timestamp", "precio_cop_kwh"} <= set(df.columns):
        largo = df.copy()
        largo["timestamp"] = pd.to_datetime(largo["timestamp"])
    else:
        raise ValueError(f"Formato de precios no reconocido: {list(df.columns)}")
    largo = largo[["timestamp", "precio_cop_kwh"]].copy()
    largo["precio_cop_kwh"] = pd.to_numeric(largo["precio_cop_kwh"], errors="coerce")
    largo = largo.dropna().sort_values("timestamp").drop_duplicates("timestamp")
    horas_por_dia = largo.groupby(largo["timestamp"].dt.date)["timestamp"].transform("size")
    return largo[horas_por_dia == 24].reset_index(drop=True)


def obtener_precio_bolsa(inicio: date, fin: date) -> tuple[pd.DataFrame, str]:
    """Intenta la API (y actualiza el respaldo); si falla, usa el CSV local.

    Returns
    -------
    (df_ancho, origen)
        ``origen`` es ``"XM API"`` o ``"XM respaldo CSV"``.
    """
    try:
        df = descargar_precio_bolsa(inicio, fin)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_RESPALDO, index=False)
        print(f"[OK] XM: {len(df)} días descargados; respaldo en {CSV_RESPALDO.name}.")
        return df, "XM API"
    except Exception as exc:  # noqa: BLE001 - cualquier fallo activa el plan B
        print(f"[WARN] API de XM no disponible ({type(exc).__name__}: {exc}). Usando respaldo local.")
        df = cargar_plan_b()
        fechas = pd.to_datetime(df["Date"]).dt.date
        return df[(fechas >= inicio) & (fechas <= fin)].reset_index(drop=True), "XM respaldo CSV"


def obtener_serie_horaria(inicio: date, fin: date) -> tuple[pd.DataFrame, str]:
    """Precio horario real ``(timestamp, precio_cop_kwh)`` entre ``inicio`` y ``fin``."""
    df, origen = obtener_precio_bolsa(inicio, fin)
    serie = a_serie_horaria(df)
    if serie.empty:
        raise RuntimeError(f"No hay días completos de precio entre {inicio} y {fin}.")
    return serie, origen


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el precio de bolsa de XM.")
    parser.add_argument("--inicio", type=_parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument("--fin", type=_parse_date, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    serie, origen = obtener_serie_horaria(args.inicio, args.fin)
    print(f"Origen: {origen} | días completos: {len(serie) // 24}")
    print(serie["precio_cop_kwh"].describe().round(1))


if __name__ == "__main__":
    main()
