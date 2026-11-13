"""Dashboard (Streamlit) — despacho de batería con RL y Copiloto Regulatorio.

Ejecutar desde la raíz del repositorio:
    streamlit run src/dashboard/app.py

Contenido:
  - Sidebar: capacidad de batería (kWh) y rango de fechas dentro del periodo de evaluación.
  - Resultados reales de ``simular()`` (precios de XM): ahorro de Q-Learning, Línea Base
    justa e Iteración de Valores; precio con decisiones de carga/descarga y SoC; ahorro
    acumulado de las tres políticas.
  - Chat con el Copiloto Regulatorio (RegulatoryAgent), con historial en session_state.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# `streamlit run` añade src/dashboard/ al sys.path, no la raíz: se agrega para importar src.*
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.agent.llm_agent import AgentError, RegulatoryAgent  # noqa: E402
from src.part1_rl.simulacion import (  # noqa: E402
    ResultadoSimulacion,
    SimulacionError,
    periodo_evaluacion,
    simular,
)

MENSAJE_BIENVENIDA = (
    "Hola, soy el Copiloto Regulatorio. Pregúntame sobre la regulación CREG para "
    "baterías (BESS) o sobre la operación del despacho."
)
COLOR = {"rl": "#1f77b4", "base": "#ff7f0e", "ref": "#2ca02c", "precio": "#555555"}


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
# Datos (cacheados: simular() es determinista para los mismos argumentos)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _periodo() -> tuple[date, date]:
    return periodo_evaluacion()


@st.cache_data(show_spinner="Simulando despacho…")
def _simular(capacidad_kwh: float, inicio: date, fin: date) -> ResultadoSimulacion:
    return simular(capacidad_kwh, fecha_inicio=inicio, fecha_fin=fin)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(periodo: tuple[date, date] | None) -> tuple[float, date | None, date | None]:
    """Dibuja los controles y devuelve (capacidad_kwh, fecha_inicio, fecha_fin)."""
    with st.sidebar:
        st.header("Parámetros de simulación")
        capacidad = st.slider(
            "Capacidad de Batería (kWh)", min_value=10, max_value=1000, value=100, step=10
        )
        inicio = fin = None
        if periodo is not None:
            minimo, maximo = periodo
            rango = st.date_input(
                "Rango de fechas de simulación",
                value=(minimo, maximo),
                min_value=minimo,
                max_value=maximo,
                format="YYYY-MM-DD",
            )
            # Mientras el usuario elige, date_input puede devolver una sola fecha.
            if isinstance(rango, (tuple, list)) and len(rango) == 2:
                inicio, fin = rango
            else:
                inicio = fin = rango[0] if isinstance(rango, (tuple, list)) else rango
                st.caption("Selecciona la fecha final del rango.")
            st.caption(f"Periodo de evaluación disponible: {minimo} a {maximo} (precios reales XM).")

        st.divider()
        st.button("Reiniciar conversación", on_click=reiniciar_chat, width="stretch")
        agente: RegulatoryAgent | None = st.session_state.get("agente")
        if agente is not None:
            st.caption(f"Modelo LLM: `{agente.modelo}` · T_max = {agente.t_max}")
    return float(capacidad), inicio, fin


# --------------------------------------------------------------------------- #
# Resultados del despacho
# --------------------------------------------------------------------------- #
def _delta(valor: float, base: float) -> str | None:
    if not base:
        return None
    return f"{(valor - base) / abs(base):+.1%} vs. Línea Base"


def tarjetas(r: ResultadoSimulacion) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Q-Learning", f"${r.ahorro_qlearning_cop:,.0f} COP",
                _delta(r.ahorro_qlearning_cop, r.ahorro_base_cop))
    col2.metric("Línea Base Justa", f"${r.ahorro_base_cop:,.0f} COP",
                "carga 01-04 h · descarga 18-21 h", delta_color="off")
    col3.metric("Iteración de Valores (Referencia)", f"${r.ahorro_referencia_cop:,.0f} COP",
                _delta(r.ahorro_referencia_cop, r.ahorro_base_cop))
    fraccion = r.fraccion_de_referencia
    detalle = f" Q-Learning logra el {fraccion:.1%} del ahorro de la referencia." if fraccion else ""
    st.caption(
        f"{r.n_dias} días ({r.fecha_inicio} a {r.fecha_fin}) · {r.capacidad_kwh:,.0f} kWh "
        f"({r.energia_por_nivel_kwh:,.1f} kWh por escalón) · Fuente: {r.fuente_datos}.{detalle}"
    )


def grafica_precio_decisiones(r: ResultadoSimulacion) -> go.Figure:
    t = r.traza
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
        subplot_titles=("Precio de bolsa XM y decisiones de Q-Learning", "Nivel de batería (SoC)"),
    )
    fig.add_trace(go.Scatter(x=t["timestamp"], y=t["precio_cop"], mode="lines", name="Precio XM",
                             line=dict(color=COLOR["precio"], width=1)), row=1, col=1)
    for accion, nombre, color, simbolo in (
        ("CARGAR", "Cargó", "#2ca02c", "triangle-up"),
        ("DESCARGAR", "Descargó", "#d62728", "triangle-down"),
    ):
        sub = t[t["accion_rl"] == accion]
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["precio_cop"], mode="markers", name=nombre,
                                 marker=dict(color=color, symbol=simbolo, size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t["timestamp"], y=t["soc_rl"], mode="lines", name="SoC Q-Learning",
                             line=dict(color=COLOR["rl"], shape="hv"), fill="tozeroy"), row=2, col=1)
    fig.update_yaxes(title_text="COP/kWh", row=1, col=1)
    fig.update_yaxes(title_text="SoC (%)", range=[0, 105], row=2, col=1)
    fig.update_layout(height=560, hovermode="x unified", legend=dict(orientation="h", y=1.08),
                      margin=dict(t=70, b=30))
    return fig


def grafica_ahorro(r: ResultadoSimulacion) -> go.Figure:
    t = r.traza
    fig = go.Figure()
    for columna, nombre, clave in (
        ("ahorro_acum_ref", "Iteración de Valores (Referencia)", "ref"),
        ("ahorro_acum_rl", "Q-Learning", "rl"),
        ("ahorro_acum_base", "Línea Base Justa", "base"),
    ):
        fig.add_trace(go.Scatter(x=t["timestamp"], y=t[columna], mode="lines", name=nombre,
                                 line=dict(color=COLOR[clave], width=2)))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(title="Ahorro acumulado (COP)", yaxis_title="COP", height=420,
                      hovermode="x unified", legend=dict(orientation="h", y=1.1),
                      margin=dict(t=70, b=30))
    return fig


def seccion_resultados(capacidad_kwh: float, inicio: date | None, fin: date | None) -> None:
    st.subheader("Resultados del despacho con precios reales de XM")
    if inicio is None or fin is None:
        return
    try:
        resultado = _simular(capacidad_kwh, inicio, fin)
    except SimulacionError as exc:
        st.error(f"No se pudo simular: {exc}")
        return
    tarjetas(resultado)
    st.plotly_chart(grafica_precio_decisiones(resultado), width="stretch")
    st.plotly_chart(grafica_ahorro(resultado), width="stretch")


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

    try:
        periodo = _periodo()
    except SimulacionError as exc:
        periodo = None
        st.error(f"Faltan artefactos de la Parte 1: {exc}")

    capacidad, inicio, fin = sidebar(periodo)
    seccion_resultados(capacidad, inicio, fin)
    st.divider()
    seccion_chat()


main()
