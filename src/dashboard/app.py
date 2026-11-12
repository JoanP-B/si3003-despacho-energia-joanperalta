"""Dashboard (Streamlit) — esqueleto del checkpoint.

Ejecutar desde la raíz del repositorio:
    streamlit run src/dashboard/app.py

Contenido:
  - Sidebar: capacidad de batería (kWh) y rango de fechas de simulación.
  - Placeholder de resultados del despacho MDP (datos falsos por ahora).
  - Chat con el Copiloto Regulatorio (RegulatoryAgent), con historial en session_state.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# `streamlit run` añade src/dashboard/ al sys.path, no la raíz: se agrega para importar src.*
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.agent.llm_agent import AgentError, RegulatoryAgent  # noqa: E402

MENSAJE_BIENVENIDA = (
    "Hola, soy el Copiloto Regulatorio. Pregúntame sobre la regulación CREG para "
    "baterías (BESS) o sobre la operación del despacho."
)


# --------------------------------------------------------------------------- #
# Estado de la sesión
# --------------------------------------------------------------------------- #
def inicializar_estado() -> None:
    """Crea las claves de session_state la primera vez que carga la página."""
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [{"role": "assistant", "content": MENSAJE_BIENVENIDA}]
    if "agente" not in st.session_state:
        st.session_state.agente = None
        st.session_state.error_agente = None
        try:
            st.session_state.agente = RegulatoryAgent()
        except (AgentError, ValueError) as exc:
            st.session_state.error_agente = str(exc)


def reiniciar_chat() -> None:
    st.session_state.mensajes = [{"role": "assistant", "content": MENSAJE_BIENVENIDA}]
    agente: RegulatoryAgent | None = st.session_state.get("agente")
    if agente is not None:
        agente.reiniciar()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> tuple[float, date, date]:
    """Dibuja los controles y devuelve (capacidad_kwh, fecha_inicio, fecha_fin)."""
    with st.sidebar:
        st.header("Parámetros de simulación")
        capacidad = st.slider(
            "Capacidad de Batería (kWh)", min_value=0, max_value=1000, value=100, step=10
        )
        hoy = date.today()
        rango = st.date_input(
            "Rango de fechas de simulación",
            value=(hoy - timedelta(days=30), hoy - timedelta(days=1)),
            max_value=hoy,
            format="YYYY-MM-DD",
        )
        # Mientras el usuario elige, date_input puede devolver una sola fecha.
        if isinstance(rango, (tuple, list)) and len(rango) == 2:
            inicio, fin = rango
        else:
            inicio = fin = rango[0] if isinstance(rango, (tuple, list)) else rango
            st.caption("Selecciona la fecha final del rango.")

        if capacidad == 0:
            st.warning("Con capacidad 0 kWh la batería no puede operar.")

        st.divider()
        st.button("Reiniciar conversación", on_click=reiniciar_chat, use_container_width=True)
        agente: RegulatoryAgent | None = st.session_state.get("agente")
        if agente is not None:
            st.caption(f"Modelo LLM: `{agente.modelo}` · T_max = {agente.t_max}")
    return float(capacidad), inicio, fin


# --------------------------------------------------------------------------- #
# Resultados del despacho (placeholder)
# --------------------------------------------------------------------------- #
def datos_falsos_despacho(inicio: date, fin: date, capacidad_kwh: float) -> pd.DataFrame:
    """Serie diaria sintética para el placeholder. Se reemplaza por la Parte 1."""
    fechas = pd.date_range(inicio, fin, freq="D")
    rng = np.random.default_rng(len(fechas))
    escala = max(capacidad_kwh, 1.0) / 100.0
    ahorro_rl = np.cumsum(rng.normal(450, 120, len(fechas))) * escala
    ahorro_base = np.cumsum(rng.normal(230, 90, len(fechas))) * escala
    return pd.DataFrame(
        {"Q-Learning (COP)": ahorro_rl, "Línea Base (COP)": ahorro_base}, index=fechas
    )


def seccion_resultados(capacidad_kwh: float, inicio: date, fin: date) -> None:
    st.subheader("Resultados del Despacho MDP (Próximamente)")
    st.caption(
        "Datos de ejemplo. Aquí se mostrará el ahorro acumulado del agente Q-Learning "
        "frente a la Línea Base y las decisiones sobre la curva de precios de XM."
    )
    if fin < inicio:
        st.error("La fecha final es anterior a la inicial.")
        return
    df = datos_falsos_despacho(inicio, fin, capacidad_kwh)
    col1, col2, col3 = st.columns(3)
    col1.metric("Capacidad", f"{capacidad_kwh:,.0f} kWh")
    col2.metric("Días simulados", len(df))
    col3.metric("Ahorro Q-Learning (ejemplo)", f"{df.iloc[-1, 0]:,.0f} COP")
    st.line_chart(df)


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def seccion_chat() -> None:
    st.subheader("Copiloto Regulatorio")
    agente: RegulatoryAgent | None = st.session_state.agente

    if agente is None:
        st.warning(f"El chat está deshabilitado: {st.session_state.error_agente}")

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    pregunta = st.chat_input("Escribe tu pregunta…", disabled=agente is None)
    if not pregunta or agente is None:
        return

    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Consultando…"):
                respuesta = agente.chat(pregunta)
        except (AgentError, ValueError) as exc:
            # No se guarda en el historial: el usuario puede reintentar la misma pregunta.
            st.session_state.mensajes.pop()
            st.error(f"No se pudo obtener respuesta: {exc}")
            return

        st.markdown(respuesta)
        resultado = agente.ultimo_resultado
        if resultado is not None and resultado.pasos:
            with st.expander(f"Herramientas usadas ({len(resultado.pasos)})"):
                for paso in resultado.pasos:
                    icono = "❌" if paso.error else "✅"
                    st.markdown(f"{icono} **{paso.herramienta}** `{paso.argumentos}`")
                    st.code(paso.observacion, language=None)
        if resultado is not None and resultado.limite_alcanzado:
            st.caption(f"Se alcanzó el límite de {agente.t_max} iteraciones.")

    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})


# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Despacho de Energía con RL", layout="wide")
    inicializar_estado()

    st.title("Despacho de Energía con RL y Copiloto Regulatorio")
    st.caption("Proyecto Final SI3003 — Universidad EAFIT")

    capacidad, inicio, fin = sidebar()
    seccion_resultados(capacidad, inicio, fin)
    st.divider()
    seccion_chat()


main()
