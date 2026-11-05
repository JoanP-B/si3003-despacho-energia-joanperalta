"""Dashboard (Streamlit) — esqueleto.

Ejecutar:  streamlit run src/dashboard/app.py

Debe mostrar (días 9-11 nov):
  - serie de tiempo precio vs. decisiones de la política
  - ahorro acumulado
  - controles de parámetros de la batería
  - chat con el agente LLM
"""
import streamlit as st

st.set_page_config(page_title="Despacho de energía con RL", layout="wide")

st.title("Despacho de energía con RL y asistente sobre regulación")
st.caption("Proyecto Final SI3003 — Universidad EAFIT — Idea 3.3")

st.info(
    "Esqueleto del dashboard. Pendiente: serie precio vs. decisión, ahorro acumulado, "
    "parámetros de batería y chat con el agente."
)

with st.sidebar:
    st.header("Parámetros de la batería")
    st.slider("Capacidad (kWh)", 1.0, 50.0, 10.0)
    st.slider("Potencia (kW)", 0.5, 10.0, 2.5)
    st.slider("Eficiencia", 0.70, 1.0, 0.95)
