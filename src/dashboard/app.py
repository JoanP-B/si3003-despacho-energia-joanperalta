"""Dashboard (Streamlit) — despacho de batería con RL y Copiloto Regulatorio.

Ejecutar desde la raíz del repositorio (así se aplica el tema de ``.streamlit/config.toml``):
    python -m streamlit run src/dashboard/app.py

Contenido:
  - Sidebar: parámetros de negocio (capacidad, eficiencia, potencia del inversor), escenario
    de entrenamiento, periodo y fechas; notas del modelo y estado del copiloto.
  - Tarjetas con el ahorro de Q-Learning, Línea Base justa e Iteración de Valores.
  - Pestaña de análisis, con una sección por gráfica: precio de XM y operación de la batería,
    ahorro acumulado y escenarios de entrenamiento; debajo, el resumen ejecutivo del LLM.
  - Pestaña del copiloto: chat con RegulatoryAgent y preguntas sugeridas.

Toda la lógica de cálculo vive en ``src.part1_rl.simulacion.simular`` y en
``src.agent.llm_agent``; este módulo solo presenta resultados.
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# `streamlit run` añade src/dashboard/ al sys.path, no la raíz: se agrega para importar src.*
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.agent.llm_agent import (  # noqa: E402
    AgentError,
    RegulatoryAgent,
    generar_resumen_ejecutivo,
)
from src.part1_rl.simulacion import (  # noqa: E402
    ESCENARIOS,
    ResultadoSimulacion,
    SimulacionError,
    curva_aprendizaje,
    eficiencia_de_grilla,
    escenarios_disponibles,
    periodo_evaluacion,
    simular,
)

MENSAJE_BIENVENIDA = (
    "Hola, soy el **Copiloto Operativo y Regulatorio**. Puedo simular el despacho de tu "
    "batería con precios reales de XM y consultar las resoluciones CREG con cita de la "
    "fuente. Elige una pregunta sugerida o escribe la tuya."
)
SUGERENCIAS = [
    ("💰", "¿Cuánto ahorro con 200 kWh?"),
    ("📜", "¿Qué dice la CREG 098 sobre BESS?"),
    ("⚡", "Simula 500 kWh y explícame la norma que aplica."),
    ("📅", "Simula 100 kWh solo entre 2026-10-01 y 2026-10-31."),
    ("🔌", "¿Qué dice la CREG 174 sobre inyección de excedentes a la red?"),
    ("⚠️", "Simula 100 kWh para enero de 2026."),
]
ESCENARIO_INFO = {
    "corto": ("Entrenamiento corto", f"{ESCENARIOS['corto']:,} episodios: apenas empieza a aprender."),
    "base": ("Base (el que usa la app)", f"{ESCENARIOS['base']:,} episodios: el del agente LLM."),
    "largo": ("Entrenamiento largo", f"{ESCENARIOS['largo']:,} episodios: presupuesto de sobra."),
}
ETIQUETA_CORTA = {e: f"{e.capitalize()} ({n:,} ep.)" for e, n in ESCENARIOS.items()}
PERIODO_INFO = {
    "prueba": "Prueba (nunca vista)",
    "validacion": "Validación (elige hiperparámetros)",
}
COLOR = {
    "rl": "#2563EB", "base": "#F59E0B", "ref": "#10B981", "precio": "#1E3A8A",
    "soc": "#3B82F6", "cargar": "#16A34A", "descargar": "#DC2626",
    "texto": "#0F172A", "suave": "#64748B", "grilla": "#EEF2F7", "eje": "#CBD5E1",
}
CONFIG_PLOTLY = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1380px;}
[data-testid="stSidebar"] {border-right: 1px solid #1E293B;}
.brand {font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0;}
.brand-sub {font-size: .8rem; color: #94A3B8; margin-bottom: .8rem;}
.hero {background: linear-gradient(115deg, #0F2A6B 0%, #1D4ED8 58%, #3B82F6 100%);
       border-radius: 16px; padding: 1.4rem 1.7rem; margin-bottom: 1.2rem; color: #F8FAFC;
       box-shadow: 0 12px 32px rgba(37, 99, 235, .22);}
.hero h1 {font-size: 1.7rem; margin: 0 0 .3rem 0; color: #FFFFFF; font-weight: 700;}
.hero p {margin: 0; color: #DBEAFE; font-size: .95rem;}
.chips {margin-top: .8rem;}
.chip {display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
       border-radius: 999px; padding: .18rem .75rem; margin: 0 .4rem .35rem 0; font-size: .78rem; color: #F8FAFC;}
.kpi {background: #FFFFFF; border-top: 5px solid var(--c); border-radius: 14px; padding: 1rem 1.15rem .95rem;
      box-shadow: 0 6px 20px rgba(0,0,0,.28); height: 100%;}
.kpi .lbl {font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; color: #64748B; font-weight: 600;}
.kpi .val {font-size: 1.8rem; font-weight: 700; color: #0F172A; margin: .25rem 0 .4rem 0;}
.kpi .sub {font-size: .8rem; color: #64748B; margin-top: .5rem;}
.pill {display: inline-block; border-radius: 999px; padding: .12rem .6rem; font-size: .78rem; font-weight: 600;}
.pill.up {background: #DCFCE7; color: #166534;} .pill.down {background: #FEE2E2; color: #991B1B;}
.pill.neutral {background: #E2E8F0; color: #334155;}
.contexto {color: #94A3B8; font-size: .84rem; margin: .7rem 0 .2rem 0;}
.aviso {color: #FBBF24; font-size: .84rem; margin: 0 0 .2rem 0;}
div[class*="st-key-card_"] {background: #FFFFFF; border-radius: 14px; padding: 1.1rem 1.25rem .5rem 1.25rem;
                            box-shadow: 0 6px 20px rgba(0,0,0,.28);}
.card-title {color: #0F172A; font-weight: 700; font-size: 1.05rem; margin: 0;}
.card-sub {color: #64748B; font-size: .86rem; margin: .2rem 0 .3rem 0;}
.card-nota {color: #64748B; font-size: .8rem; margin: 0 0 .6rem 0;}
.panel {background: rgba(59,130,246,.08); border: 1px solid rgba(59,130,246,.35); border-radius: 14px;
        padding: .9rem 1.15rem; margin: .4rem 0 .8rem 0;}
.panel h3 {margin: 0; font-size: 1.12rem; color: #F1F5F9;}
.panel p {margin: .25rem 0 0 0; color: #94A3B8; font-size: .86rem;}
.etiqueta {font-size: .8rem; color: #94A3B8; margin: .2rem 0 .3rem 0;}
div[class*="st-key-sug_"] button {border-radius: 12px; border: 1px solid rgba(59,130,246,.35);
                                  background: rgba(59,130,246,.07); min-height: 2.9rem;}
div[class*="st-key-sug_"] button:hover {border-color: #3B82F6; background: rgba(59,130,246,.16);}
</style>
"""


# --------------------------------------------------------------------------- #
# Estado de la sesión
# --------------------------------------------------------------------------- #
def inicializar_estado() -> None:
    """Crea las claves de session_state la primera vez que carga la página."""
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [{"role": "assistant", "content": MENSAJE_BIENVENIDA}]
    if "pregunta_pendiente" not in st.session_state:
        st.session_state.pregunta_pendiente = None
    if "resumenes" not in st.session_state:
        st.session_state.resumenes = {}
    if "agente" not in st.session_state:
        st.session_state.agente = None
        st.session_state.error_agente = None
        try:
            st.session_state.agente = RegulatoryAgent()
        except (AgentError, ValueError) as exc:
            st.session_state.error_agente = str(exc)


def reiniciar_chat() -> None:
    st.session_state.mensajes = [{"role": "assistant", "content": MENSAJE_BIENVENIDA}]
    st.session_state.pregunta_pendiente = None
    agente: RegulatoryAgent | None = st.session_state.get("agente")
    if agente is not None:
        agente.reiniciar()


def encolar_pregunta(pregunta: str) -> None:
    st.session_state.pregunta_pendiente = pregunta


# --------------------------------------------------------------------------- #
# Datos (cacheados: simular() es determinista para los mismos argumentos)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _periodo(periodo: str) -> tuple[date, date]:
    return periodo_evaluacion(periodo=periodo)


@st.cache_data(show_spinner="Simulando despacho…")
def _simular(capacidad_kwh: float, inicio: date, fin: date, escenario: str, periodo: str,
             eficiencia: float, inversor_kw: float) -> ResultadoSimulacion:
    return simular(capacidad_kwh, fecha_inicio=inicio, fecha_fin=fin, periodo=periodo,
                   escenario=escenario, eficiencia=eficiencia, potencia_inversor_kw=inversor_kw)


@st.cache_data(show_spinner=False)
def _curva() -> pd.DataFrame:
    return curva_aprendizaje()


def _simular_params(params: dict, escenario: str | None = None) -> ResultadoSimulacion:
    return _simular(params["capacidad"], params["inicio"], params["fin"],
                    escenario or params["escenario"], params["periodo"],
                    params["eficiencia_pct"] / 100, float(params["inversor_kw"]))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _slider_inversor(capacidad: float) -> int:
    """Potencia del inversor con rango proporcional a la capacidad (10 % a 100 %)."""
    paso = max(1, int(capacidad // 100))
    minimo = max(paso, int(round(capacidad * 0.10 / paso)) * paso)
    maximo = max(minimo, int(capacidad))
    sin_limite = min(maximo, max(minimo, math.ceil(capacidad / 4 / paso) * paso))
    return st.slider(
        "Potencia del inversor (kW)", min_value=minimo, max_value=maximo, value=sin_limite,
        step=paso,
        help=("Limita la energía que entra o sale en cada hora. El modelo mueve un escalón "
              f"(capacidad / 4 = {capacidad / 4:,.1f} kWh) por hora: con menos potencia cada "
              "hora mueve menos energía; con más, el resultado no cambia."),
    )


def sidebar() -> dict:
    """Dibuja los controles y devuelve los parámetros elegidos."""
    params: dict = {"inicio": None, "fin": None}
    with st.sidebar:
        st.markdown("<div class='brand'>⚡ Despacho BESS</div>"
                    "<div class='brand-sub'>Proyecto Final SI3003 · Universidad EAFIT</div>",
                    unsafe_allow_html=True)

        with st.expander("⚙️ Parámetros de simulación", expanded=True):
            params["capacidad"] = float(st.slider(
                "Capacidad de la batería (kWh)", min_value=10, max_value=1000, value=100, step=10,
                help="Se reparte en 4 escalones de carga (25 % cada uno).",
            ))
            params["eficiencia_pct"] = int(st.slider(
                "Eficiencia del sistema (%)", min_value=70, max_value=100, value=90, step=5,
                help=("Eficiencia de ida y vuelta: de cada kWh comprado, la batería entrega este "
                      "porcentaje al descargar. Las compras se pagan completas."),
            ))
            params["inversor_kw"] = _slider_inversor(params["capacidad"])

            disponibles = escenarios_disponibles(params["eficiencia_pct"] / 100) or ["base"]
            params["escenario"] = st.radio(
                "Escenario de entrenamiento de Q-Learning", disponibles,
                index=disponibles.index("base") if "base" in disponibles else 0,
                format_func=lambda e: ESCENARIO_INFO[e][0],
                captions=[ESCENARIO_INFO[e][1] for e in disponibles],
            )
            params["periodo"] = st.segmented_control(
                "Periodo de evaluación", list(PERIODO_INFO), default="prueba", required=True,
                format_func=lambda p: "Prueba" if p == "prueba" else "Validación",
                width="stretch",
                help=("Prueba: días nunca usados para entrenar ni elegir hiperparámetros (cifra "
                      "oficial). Validación: días usados para elegir hiperparámetros."),
            ) or "prueba"
            try:
                minimo, maximo = _periodo(params["periodo"])
            except SimulacionError as exc:
                st.error(f"Faltan artefactos de la Parte 1: {exc}")
            else:
                rango = st.date_input(
                    "Rango de fechas", value=(minimo, maximo), min_value=minimo,
                    max_value=maximo, format="YYYY-MM-DD", key=f"rango_{params['periodo']}",
                )
                # Mientras el usuario elige, date_input puede devolver una sola fecha.
                if isinstance(rango, (tuple, list)) and len(rango) == 2:
                    params["inicio"], params["fin"] = rango
                else:
                    unico = rango[0] if isinstance(rango, (tuple, list)) else rango
                    params["inicio"] = params["fin"] = unico
                    st.caption("Selecciona la fecha final del rango.")
                st.caption(f"Disponible: {minimo} a {maximo} · precios reales de XM.")

        with st.expander("📘 Notas del modelo", expanded=False):
            st.markdown(
                "- **Línea Base justa:** carga en las horas 01-04 y descarga en las 18-21, "
                "sin mirar el precio.\n"
                "- **Q-Learning:** tabular, estado (hora, cuartil de precio, SoC), con *reward "
                "shaping* y tasa α = 1/n(s,a)^ω. Cada escenario se entrenó con la eficiencia "
                "elegida.\n"
                "- **Iteración de Valores:** referencia que resuelve el mismo MDP con el modelo "
                "empírico del entrenamiento y se replanifica con la eficiencia.\n"
                "- **Eficiencia:** la descarga evita η · precio · kWh y la compra se paga "
                "completa.\n"
                "- **Inversor:** limita la energía de cada hora.\n"
                "- **Datos:** precio de bolsa nacional (`PrecBolsNaci`) de XM. El ahorro se mide "
                "con el precio de bolsa, no con la tarifa del usuario.\n"
                "- Herramienta analítica y educativa; no es asesoría regulatoria ni financiera."
            )

        with st.expander("🤖 Copiloto", expanded=False):
            agente: RegulatoryAgent | None = st.session_state.get("agente")
            if agente is not None:
                st.markdown(f"Modelo: `{agente.modelo}`  \nLímite del loop: T_max = {agente.t_max}")
                st.caption("El copiloto simula con el escenario base, los días de prueba y una "
                           "batería ideal (eficiencia 100 %, sin límite de inversor).")
            else:
                st.warning("Chat y resumen deshabilitados: falta `GROQ_API_KEY` en `.env`.")
            st.button("🔄 Reiniciar conversación", on_click=reiniciar_chat, width="stretch")
    return params


# --------------------------------------------------------------------------- #
# Encabezado y tarjetas
# --------------------------------------------------------------------------- #
def encabezado(params: dict) -> None:
    chips = [
        "Datos: XM PrecBolsNaci",
        f"Periodo: {PERIODO_INFO[params['periodo']]}",
        f"Escenario: {ESCENARIO_INFO[params['escenario']][0]}",
        f"Batería: {params['capacidad']:,.0f} kWh",
        f"Eficiencia: {params['eficiencia_pct']} %",
        f"Inversor: {params['inversor_kw']:,} kW",
    ]
    st.markdown(
        "<div class='hero'><h1>Despacho de Energía con RL y Copiloto Regulatorio</h1>"
        "<p>Cuándo cargar y descargar una batería frente al precio de bolsa de XM, comparado "
        "con una regla fija y con una política de referencia.</p><div class='chips'>"
        + "".join(f"<span class='chip'>{c}</span>" for c in chips) + "</div></div>",
        unsafe_allow_html=True,
    )


def _cop(valor: float) -> str:
    """Pesos con separador de miles y el signo antes del símbolo: -$1,234."""
    return f"-${abs(valor):,.0f}" if valor < 0 else f"${valor:,.0f}"


def _pill(r: ResultadoSimulacion, valor: float) -> str:
    cambio = r.vs_base(valor)
    if cambio is None:
        return "<span class='pill neutral'>—</span>"
    clase = "up" if cambio > 0.0005 else "down" if cambio < -0.0005 else "neutral"
    return f"<span class='pill {clase}'>{cambio:+.1%} vs. Línea Base</span>"


def tarjetas(r: ResultadoSimulacion) -> None:
    fraccion = r.fraccion_de_referencia
    datos = [
        ("Q-Learning", COLOR["rl"], r.ahorro_qlearning_cop, _pill(r, r.ahorro_qlearning_cop),
         f"{r.cargas_qlearning} cargas · {r.descargas_qlearning} descargas"
         + (f" · {fraccion:.0%} de la referencia" if fraccion and fraccion > 0 else "")),
        ("Línea Base justa", COLOR["base"], r.ahorro_base_cop,
         "<span class='pill neutral'>Regla fija 01-04 h / 18-21 h</span>",
         f"{r.cargas_base} cargas · {r.descargas_base} descargas"),
        ("Iteración de Valores (referencia)", COLOR["ref"], r.ahorro_referencia_cop,
         _pill(r, r.ahorro_referencia_cop),
         f"{r.cargas_referencia} cargas · {r.descargas_referencia} descargas"),
    ]
    for col, (titulo, color, valor, pill, sub) in zip(st.columns(3, gap="medium"), datos):
        col.markdown(
            f"<div class='kpi' style='--c:{color}'><div class='lbl'>{titulo}</div>"
            f"<div class='val'>{_cop(valor)} COP</div>{pill}<div class='sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div class='contexto'>{r.n_dias} días ({r.fecha_inicio} a {r.fecha_fin}) · "
        f"{r.capacidad_kwh:,.0f} kWh · eficiencia {r.eficiencia:.0%} · "
        f"{r.energia_por_nivel_kwh:,.1f} kWh por hora · Fuente: {r.fuente_datos}</div>",
        unsafe_allow_html=True,
    )
    if r.limitado_por_inversor:
        st.markdown(
            f"<div class='aviso'>⚠️ El inversor de {r.potencia_inversor_kw:,.0f} kW limita cada "
            f"hora a {r.energia_por_nivel_kwh:,.1f} kWh de los {r.capacidad_kwh / 4:,.1f} kWh "
            "que permite la batería.</div>", unsafe_allow_html=True,
        )
    if abs(r.eficiencia - r.eficiencia_entrenamiento_ql) > 1e-6:
        st.markdown(
            f"<div class='aviso'>Q-Learning se entrenó con eficiencia "
            f"{r.eficiencia_entrenamiento_ql:.0%}.</div>", unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Gráficas (tema plotly_white dentro de tarjetas blancas)
# --------------------------------------------------------------------------- #
def _estilo(fig: go.Figure, alto: int, margen_sup: int = 56) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=alto, hovermode="x unified",
        font=dict(size=12, color=COLOR["texto"]),
        margin=dict(t=margen_sup, b=24, l=12, r=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=COLOR["eje"], ticks="outside",
                     tickcolor=COLOR["eje"])
    fig.update_yaxes(showgrid=True, gridcolor=COLOR["grilla"], zeroline=False, showline=False)
    return fig


def _tarjeta(clave: str, titulo: str, subtitulo: str):
    """Contenedor blanco (estilo por la clase ``st-key-card_*``) con título y subtítulo."""
    contenedor = st.container(key=f"card_{clave}")
    contenedor.markdown(
        f"<div class='card-title'>{titulo}</div><div class='card-sub'>{subtitulo}</div>",
        unsafe_allow_html=True,
    )
    return contenedor


def grafica_operacion(r: ResultadoSimulacion) -> go.Figure:
    """Arriba solo el precio de XM; abajo el SoC con las cargas y descargas de Q-Learning."""
    t = r.traza
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.52, 0.48],
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(
        x=t["timestamp"], y=t["precio_cop"], mode="lines", name="Precio de bolsa XM",
        line=dict(color=COLOR["precio"], width=1.6),
        hovertemplate="%{y:,.0f} COP/kWh<extra>Precio XM</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t["timestamp"], y=t["soc_rl"], mode="lines", name="Nivel de batería (SoC)",
        line=dict(color=COLOR["soc"], width=1.6, shape="hv"), fill="tozeroy",
        fillcolor="rgba(59,130,246,0.10)", hovertemplate="%{y:.0f} %<extra>SoC</extra>"),
        row=2, col=1)
    for accion, nombre, color, simbolo in (
        ("CARGAR", "Cargó", COLOR["cargar"], "triangle-up"),
        ("DESCARGAR", "Descargó", COLOR["descargar"], "triangle-down"),
    ):
        sub = t[t["accion_rl"] == accion]
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["soc_rl"], mode="markers", name=nombre,
            marker=dict(color=color, symbol=simbolo, size=10, line=dict(width=1, color="#FFFFFF")),
            customdata=sub["precio_cop"],
            hovertemplate=f"{nombre} a %{{customdata:,.0f}} COP/kWh<extra></extra>"), row=2, col=1)
    _estilo(fig, 600, margen_sup=44)
    fig.update_yaxes(title_text="COP/kWh", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="SoC (%)", range=[-6, 108], tickvals=[0, 25, 50, 75, 100],
                     row=2, col=1)
    inicio = t["timestamp"].min()
    vista = min(t["timestamp"].max(), inicio + pd.Timedelta(days=7))
    fig.update_xaxes(range=[inicio, vista])
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[dict(count=1, label="1 día", step="day", stepmode="backward"),
                     dict(count=7, label="7 días", step="day", stepmode="backward"),
                     dict(count=14, label="14 días", step="day", stepmode="backward"),
                     dict(step="all", label="Todo")],
            x=0, xanchor="left", y=1.02, yanchor="bottom", bgcolor="#EFF6FF",
            activecolor="#BFDBFE", font=dict(size=11),
        ),
        row=1, col=1,
    )
    return fig


def grafica_ahorro(r: ResultadoSimulacion) -> go.Figure:
    """Ahorro acumulado al cierre de cada día (menos ruido que la serie horaria)."""
    diario = (r.traza.set_index("timestamp")[["ahorro_acum_ref", "ahorro_acum_rl", "ahorro_acum_base"]]
              .resample("D").last())
    fig = go.Figure()
    for columna, nombre, clave, estilo in (
        ("ahorro_acum_ref", "Iteración de Valores", "ref", "dot"),
        ("ahorro_acum_rl", "Q-Learning", "rl", "solid"),
        ("ahorro_acum_base", "Línea Base justa", "base", "dash"),
    ):
        final = diario[columna].iloc[-1]
        fig.add_trace(go.Scatter(
            x=diario.index, y=diario[columna], mode="lines+markers",
            name=f"{nombre} · {_cop(final)}", marker=dict(size=4),
            line=dict(color=COLOR[clave], width=2.6, dash=estilo),
            hovertemplate=f"%{{y:,.0f}} COP<extra>{nombre}</extra>"))
    _estilo(fig, 440)
    fig.add_hline(y=0, line_color=COLOR["eje"], line_width=1)
    fig.update_yaxes(title_text="COP acumulados", tickformat=",.0f")
    return fig


def grafica_escenarios(params: dict) -> go.Figure | None:
    """Ahorro de Q-Learning en cada escenario de entrenamiento con los parámetros elegidos."""
    filas = []
    for escenario in escenarios_disponibles(params["eficiencia_pct"] / 100):
        try:
            r = _simular_params(params, escenario)
        except SimulacionError:
            continue
        filas.append({"escenario": escenario, "nombre": ETIQUETA_CORTA[escenario],
                      "ql": r.ahorro_qlearning_cop, "cambio": r.vs_base(r.ahorro_qlearning_cop),
                      "base": r.ahorro_base_cop, "ref": r.ahorro_referencia_cop})
    if not filas:
        return None
    df = pd.DataFrame(filas)
    # El cambio frente a la Línea Base va bajo cada barra, en verde o rojo.
    categorias = [
        f"<b>{e.capitalize()}</b> · {ESCENARIOS[e]:,} ep.<br>"
        f"<span style='color:{COLOR['cargar'] if (c or 0) > 0 else COLOR['descargar']}'>"
        f"{'▲' if (c or 0) > 0 else '▼'} {(c or 0):+.1%}</span>"
        for e, c in zip(df["escenario"], df["cambio"])
    ]
    opacidad = [1.0 if e == params["escenario"] else 0.5 for e in df["escenario"]]
    base, ref = df["base"].iloc[0], df["ref"].iloc[0]
    fig = go.Figure(go.Bar(
        x=categorias, y=df["ql"], name="Q-Learning", marker_color=COLOR["rl"],
        marker_opacity=opacidad, text=[_cop(v) for v in df["ql"]], textposition="outside",
        textangle=0, constraintext="none", textfont=dict(color=COLOR["texto"], size=13),
        cliponaxis=False, hovertemplate="%{y:,.0f} COP<extra>Q-Learning</extra>"))
    fig.add_hline(y=base, line_dash="dash", line_color=COLOR["base"], line_width=2)
    fig.add_hline(y=ref, line_dash="dot", line_color=COLOR["ref"], line_width=2)
    # Trazas vacías para que las líneas de referencia aparezcan en la leyenda.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Línea Base justa · {_cop(base)}",
                             line=dict(color=COLOR["base"], dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Iteración de Valores · {_cop(ref)}",
                             line=dict(color=COLOR["ref"], dash="dot", width=2)))
    techo = max(df["ql"].max(), base, ref, 1.0)
    piso = min(df["ql"].min(), 0.0)
    _estilo(fig, 450, margen_sup=50)
    fig.update_layout(hovermode="closest", bargap=0.4, legend=dict(font=dict(size=11)))
    fig.update_xaxes(tickangle=0, automargin=True)
    fig.update_yaxes(title_text="Ahorro (COP)", tickformat=",.0f",
                     range=[piso * 1.3, techo * 1.18])
    return fig


def grafica_curva(params: dict, r: ResultadoSimulacion) -> go.Figure | None:
    """Ahorro de Q-Learning según los episodios de entrenamiento (periodo completo)."""
    try:
        curva = _curva()
    except SimulacionError:
        return None
    eta_pct = eficiencia_de_grilla(params["eficiencia_pct"] / 100)
    sub = curva[curva["eficiencia_pct"] == eta_pct].sort_values("episodios")
    if sub.empty:
        return None
    per = "validacion" if params["periodo"] == "validacion" else "prueba"
    escala = r.energia_por_nivel_kwh  # la curva se calculó con 1 kWh por escalón
    episodios = sub["episodios"].astype(int).tolist()
    ql = (sub[f"ql_{per}"] * escala).tolist()
    base = float(sub[f"base_{per}"].iloc[0]) * escala
    ref = float(sub[f"ref_{per}"].iloc[0]) * escala

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodios, y=ql, mode="lines+markers", name="Q-Learning",
        line=dict(color=COLOR["rl"], width=2.6), marker=dict(size=7),
        hovertemplate="%{x:,} episodios: %{y:,.0f} COP<extra>Q-Learning</extra>"))
    hitos = [(n, y) for n, y in zip(episodios, ql) if n in ESCENARIOS.values()]
    nombres = {n: e.capitalize() for e, n in ESCENARIOS.items()}
    fig.add_trace(go.Scatter(
        x=[n for n, _ in hitos], y=[y for _, y in hitos], mode="markers+text",
        name="Escenarios del dashboard", text=[nombres[n] for n, _ in hitos],
        textposition=["bottom right" if n == min(ESCENARIOS.values()) else "top center"
                      for n, _ in hitos],
        textfont=dict(color=COLOR["texto"], size=12),
        marker=dict(size=13, color="#FFFFFF", line=dict(width=3, color=COLOR["rl"])),
        hoverinfo="skip", showlegend=False))
    fig.add_hline(y=base, line_dash="dash", line_color=COLOR["base"], line_width=2)
    fig.add_hline(y=ref, line_dash="dot", line_color=COLOR["ref"], line_width=2)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Línea Base justa",
                             line=dict(color=COLOR["base"], dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Iteración de Valores",
                             line=dict(color=COLOR["ref"], dash="dot", width=2)))

    valores = ql + [base, ref, 0.0]
    margen = 0.12 * (max(valores) - min(valores) or 1.0)
    _estilo(fig, 450, margen_sup=50)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(type="log", tickvals=episodios,
                     ticktext=[f"{n // 1000}k" if n >= 1000 else str(n) for n in episodios],
                     title_text="Episodios de entrenamiento (escala log)")
    fig.update_yaxes(title_text="Ahorro (COP)", tickformat=",.0f",
                     range=[min(valores) - margen, max(valores) + margen])
    return fig


def seccion_resumen(params: dict, r: ResultadoSimulacion) -> None:
    """Botón que pide al LLM un párrafo para gerencia con las cifras de la simulación."""
    st.markdown(
        "<div class='panel'><h3>🧾 Resumen ejecutivo para gerencia</h3><p>El LLM (Groq) redacta "
        "un párrafo a partir de las cifras de esta simulación. No calcula ni cambia los números: "
        "los recibe ya calculados por el simulador.</p></div>", unsafe_allow_html=True,
    )
    clave = (params["capacidad"], params["eficiencia_pct"], params["inversor_kw"],
             params["escenario"], params["periodo"], params["inicio"], params["fin"])
    disponible = st.session_state.get("agente") is not None
    if st.button("Generar Resumen Ejecutivo para Gerencia", type="primary",
                 disabled=not disponible, key="btn_resumen"):
        try:
            with st.spinner("Redactando el resumen…"):
                st.session_state.resumenes[clave] = generar_resumen_ejecutivo(r.resumen())
        except (AgentError, ValueError) as exc:
            st.error(f"No se pudo generar el resumen: {exc}")
    texto = st.session_state.resumenes.get(clave)
    if texto:
        st.success(texto, icon="📋")
    elif not disponible:
        st.caption("Requiere `GROQ_API_KEY` en `.env`.")
    else:
        st.caption("El resumen corresponde a los parámetros actuales; si los cambias, genera uno nuevo.")


def tab_analisis(params: dict, r: ResultadoSimulacion) -> None:
    secciones = st.tabs(["⚡ Precio y operación", "📈 Ahorro acumulado",
                         "🧪 Escenarios de entrenamiento"], key="seccion_graficas")
    with secciones[0]:
        tarjeta = _tarjeta(
            "operacion", "Precio de bolsa XM y operación de la batería",
            "Arriba, el precio real hora a hora. Abajo, el nivel de carga de Q-Learning con sus "
            "decisiones: ▲ verde cargó, ▼ rojo descargó.")
        tarjeta.plotly_chart(grafica_operacion(r), width="stretch", theme=None, config=CONFIG_PLOTLY)
        tarjeta.markdown("<div class='card-nota'>Se muestra la primera semana del rango; usa los "
                         "botones o arrastra para ver otros días.</div>", unsafe_allow_html=True)
    with secciones[1]:
        tarjeta = _tarjeta(
            "ahorro", "Ahorro acumulado de las tres políticas",
            "Valor al cierre de cada día, en COP: compras a la red (negativo) y costo evitado al "
            "descargar (positivo, ya con la eficiencia).")
        tarjeta.plotly_chart(grafica_ahorro(r), width="stretch", theme=None, config=CONFIG_PLOTLY)
    with secciones[2]:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            tarjeta = _tarjeta(
                "escenarios", "Q-Learning en cada escenario",
                "Misma capacidad, eficiencia, periodo y fechas. Bajo cada barra, el cambio "
                "frente a la Línea Base; la resaltada es el escenario de la barra lateral.")
            fig = grafica_escenarios(params)
            if fig is None:
                tarjeta.markdown("<div class='card-nota'>Ejecuta notebooks/02_mdp_rl.ipynb para "
                                 "generar los escenarios.</div>", unsafe_allow_html=True)
            else:
                tarjeta.plotly_chart(fig, width="stretch", theme=None, config=CONFIG_PLOTLY)
        with col2:
            tarjeta = _tarjeta(
                "curva", "Curva de aprendizaje",
                f"Ahorro según los episodios de entrenamiento, con eficiencia "
                f"{eficiencia_de_grilla(params['eficiencia_pct'] / 100)} % y todo el periodo de "
                f"{'validación' if params['periodo'] == 'validacion' else 'prueba'}.")
            fig = grafica_curva(params, r)
            if fig is None:
                tarjeta.markdown("<div class='card-nota'>Ejecuta notebooks/02_mdp_rl.ipynb para "
                                 "generar la curva.</div>", unsafe_allow_html=True)
            else:
                tarjeta.plotly_chart(fig, width="stretch", theme=None, config=CONFIG_PLOTLY)
                tarjeta.markdown("<div class='card-nota'>Escalada a la capacidad e inversor "
                                 "elegidos; no aplica el filtro de fechas.</div>",
                                 unsafe_allow_html=True)
    st.write("")
    seccion_resumen(params, r)


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def _mostrar_herramientas(pasos: list[dict]) -> None:
    if not pasos:
        return
    with st.expander(f"🔧 Herramientas usadas ({len(pasos)})"):
        for paso in pasos:
            icono = "❌" if paso["error"] else "✅"
            st.markdown(f"{icono} **{paso['herramienta']}** `{paso['argumentos']}`")
            st.code(paso["observacion"], language=None, wrap_lines=True)


def _responder(agente: RegulatoryAgent, pregunta: str) -> None:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(pregunta)
    with st.chat_message("assistant", avatar="⚡"):
        try:
            with st.spinner("Consultando herramientas…"):
                respuesta = agente.chat(pregunta)
        except (AgentError, ValueError) as exc:
            # No se guarda en el historial: el usuario puede reintentar la misma pregunta.
            st.session_state.mensajes.pop()
            st.error(f"No se pudo obtener respuesta: {exc}")
            return
        st.markdown(respuesta)
        resultado = agente.ultimo_resultado
        pasos = [
            {"herramienta": p.herramienta, "argumentos": p.argumentos,
             "observacion": p.observacion, "error": p.error}
            for p in (resultado.pasos if resultado else [])
        ]
        _mostrar_herramientas(pasos)
        if resultado is not None and resultado.limite_alcanzado:
            st.caption(f"Se alcanzó el límite de {agente.t_max} iteraciones.")
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta, "pasos": pasos})


def tab_copiloto() -> None:
    agente: RegulatoryAgent | None = st.session_state.agente
    st.markdown(
        "<div class='panel'><h3>Copiloto Operativo y Regulatorio CREG</h3>"
        "<p>Las cifras salen del simulador de la Parte 1 y la normativa de la CREG 098/2019 y "
        "174/2021, con cita de archivo y página. El copiloto simula con el escenario base, los "
        "días de prueba y una batería ideal (eficiencia 100 %).</p></div>",
        unsafe_allow_html=True,
    )
    if agente is None:
        st.warning(f"El chat está deshabilitado: {st.session_state.error_agente}")

    st.markdown("<div class='etiqueta'>Preguntas sugeridas</div>", unsafe_allow_html=True)
    columnas = st.columns(3, gap="small")
    for i, (icono, texto) in enumerate(SUGERENCIAS):
        columnas[i % 3].button(f"{icono} {texto}", key=f"sug_{i}", width="stretch",
                               disabled=agente is None, on_click=encolar_pregunta, args=(texto,))

    conversacion = st.container(height=520, border=True)
    with conversacion:
        for mensaje in st.session_state.mensajes:
            avatar = "⚡" if mensaje["role"] == "assistant" else "🧑‍💼"
            with st.chat_message(mensaje["role"], avatar=avatar):
                st.markdown(mensaje["content"])
                _mostrar_herramientas(mensaje.get("pasos", []))

    escrita = st.chat_input("Escribe tu pregunta sobre ahorro, despacho o regulación CREG…",
                            disabled=agente is None)
    pregunta = escrita or st.session_state.pregunta_pendiente
    st.session_state.pregunta_pendiente = None
    if pregunta and agente is not None:
        with conversacion:
            _responder(agente, pregunta)


# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Despacho de Energía con RL", page_icon="⚡", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    inicializar_estado()

    params = sidebar()
    encabezado(params)

    resultado = None
    if params["inicio"] is not None and params["fin"] is not None:
        try:
            resultado = _simular_params(params)
        except SimulacionError as exc:
            st.error(f"No se pudo simular: {exc}")
    if resultado is not None:
        tarjetas(resultado)
    st.write("")

    pestana_analisis, pestana_chat = st.tabs(
        ["📊 Análisis Operativo y Gráficas", "💬 Copiloto Regulatorio e Interacción"],
        key="pestana_principal",
    )
    with pestana_analisis:
        if resultado is not None:
            tab_analisis(params, resultado)
        else:
            st.info("Elige parámetros válidos en la barra lateral para ver el análisis.")
    with pestana_chat:
        tab_copiloto()


main()
