"""Parte 2 — Agente LLM ("Copiloto Regulatorio y Operativo") con loop agéntico manual.

Herramientas reales:
    - query_creg_regulation: búsqueda semántica (FAISS, top-3) en el corpus CREG.
    - simular_despacho: ejecuta la política Q-Learning entrenada (Parte 1), la Línea Base
      y la política de referencia (Iteración de Valores).

Loop agéntico (máximo T_MAX iteraciones):
    1. El LLM recibe el historial y decide: responder o pedir una herramienta.
    2. Si pide herramienta, se ejecuta localmente. Los errores se devuelven al LLM como
       observación (ToolMessage con status="error") para que los explique o corrija.
    3. Se repite hasta obtener una respuesta sin tool calls o agotar T_MAX; en ese caso
       se fuerza una respuesta final sin herramientas.

Proveedor: Groq vía ``langchain-groq`` (requiere la variable de entorno GROQ_API_KEY).

Requisitos previos:
    python -m src.rag.vectorstore          # construye data/processed/faiss_index
    notebooks/02_mdp_rl.ipynb              # genera q_table.json, politica_referencia.json
                                           # y precios_evaluacion.csv

Uso por consola:
    python -m src.agent.llm_agent "¿Qué dice la CREG sobre almacenamiento con baterías?"
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool

# Modelo por defecto: llama3-70b-8192 y llama-3.3-70b-versatile fueron retirados por
# Groq; openai/gpt-oss-120b es el reemplazo recomendado y soporta tool use.
DEFAULT_MODEL = "openai/gpt-oss-120b"
T_MAX = 5  # límite de iteraciones del loop (definido en el informe)
MAX_TURNOS_HISTORIAL = 10  # pares usuario/asistente que se conservan como memoria
TOP_K = 3  # fragmentos que devuelve el RAG
PREFIJO_ERROR = "ERROR"  # las herramientas devuelven errores como texto con este prefijo

SYSTEM_PROMPT = """Eres el Copiloto Regulatorio y Operativo de un sistema de \
almacenamiento de energía con baterías (BESS) en Colombia. El sistema hace \
desplazamiento de demanda (load shifting): carga la batería en horas de precio de \
bolsa bajo y la descarga para cubrir demanda propia en horas caras. No especula ni \
vende energía a la red.

Herramientas:
- query_creg_regulation: para cualquier pregunta sobre regulación, normas o \
resoluciones CREG. Úsala antes de responder y cita cada fragmento que uses con el \
formato (Fuente: archivo, página N), o (Fuente: archivo) si no trae página. No uses \
otros marcadores de cita.
- simular_despacho: para preguntas sobre ahorro, desempeño del agente de RL o \
comparación con la Línea Base o la política de referencia (Iteración de Valores). Tú no \
calculas cifras: repórtalas tal como las devuelve la herramienta, incluso si Q-Learning \
ahorra menos que la Línea Base. Envía fecha_inicio y fecha_fin solo si el usuario pide un \
rango de fechas. Si la herramienta indica que los precios son simulados, dilo.

Reglas:
- Toda afirmación normativa debe salir de los fragmentos devueltos. No completes con \
conocimiento propio: no inventes requisitos, cifras, artículos ni resoluciones. Si los \
fragmentos no traen el detalle pedido, dilo explícitamente.
- No repitas una llamada con los mismos argumentos; usa el resultado que ya tienes.
- Si una herramienta devuelve un texto que empieza por ERROR, explícale al usuario qué \
falló en lenguaje sencillo; si puedes corregir los argumentos, reintenta una vez.
- Responde en español, de forma breve y clara."""


# --------------------------------------------------------------------------- #
# Herramientas
# --------------------------------------------------------------------------- #
@tool
def query_creg_regulation(topic: str) -> str:
    """Busca en el corpus de resoluciones CREG los 3 fragmentos más relevantes sobre un
    tema y los devuelve con su fuente (archivo y página). Úsala para preguntas sobre
    regulación de baterías (SAEB/BESS), autogeneración, generación distribuida o
    mercado de energía.

    Args:
        topic: Tema o pregunta regulatoria, por ejemplo "remuneración de los SAEB".
    """
    tema = (topic or "").strip()
    if not tema:
        return f"{PREFIJO_ERROR}: el parámetro 'topic' está vacío; indica un tema regulatorio."
    if len(tema) > 500:
        return f"{PREFIJO_ERROR}: el tema es demasiado largo (máximo 500 caracteres)."

    try:
        from src.rag.vectorstore import RAGError, buscar
    except ImportError as exc:
        return f"{PREFIJO_ERROR}: el módulo RAG no está disponible ({exc})."
    try:
        fragmentos = buscar(tema, k=TOP_K)
    except RAGError as exc:
        return f"{PREFIJO_ERROR}: el índice regulatorio no está disponible. {exc}"
    except Exception as exc:  # noqa: BLE001 - p. ej. falta de dependencias de embeddings
        return f"{PREFIJO_ERROR}: falló la búsqueda en el corpus ({type(exc).__name__}: {exc})."

    if not fragmentos:
        return f"No se encontraron fragmentos sobre '{tema}' en el corpus CREG."
    bloques = []
    for i, frag in enumerate(fragmentos, 1):
        pagina = f", página {frag['pagina']}" if frag.get("pagina") else ""
        bloques.append(f"[{i}] Fuente: {frag['fuente']}{pagina}\n{frag['texto'].strip()}")
    return f"Fragmentos del corpus CREG sobre '{tema}':\n\n" + "\n\n---\n\n".join(bloques)


@tool
def simular_despacho(
    capacidad_kwh: float, fecha_inicio: str | None = None, fecha_fin: str | None = None
) -> str:
    """Simula el despacho de una batería sobre los días de evaluación (precios reales de
    XM) con tres políticas: Q-Learning entrenado, Línea Base (carga 01-04 h, descarga
    18-21 h) e Iteración de Valores (política de referencia). Devuelve el ahorro
    acumulado de cada una en COP.

    Args:
        capacidad_kwh: Capacidad de la batería en kWh (mayor que 0), por ejemplo 100.
        fecha_inicio: Opcional, AAAA-MM-DD. Úsalo solo si el usuario pide un rango de fechas.
        fecha_fin: Opcional, AAAA-MM-DD. Úsalo solo si el usuario pide un rango de fechas.
    """
    try:
        from src.part1_rl.simulacion import SimulacionError, simular
    except ImportError as exc:
        return f"{PREFIJO_ERROR}: el módulo de simulación no está disponible ({exc})."
    try:
        r = simular(capacidad_kwh, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    except SimulacionError as exc:
        return f"{PREFIJO_ERROR}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"{PREFIJO_ERROR}: la simulación falló ({type(exc).__name__}: {exc})."

    if r.ahorro_base_cop:
        mejora = f" ({r.diferencia_cop / abs(r.ahorro_base_cop):+.1%} frente a la Línea Base)"
    else:
        mejora = ""
    fraccion = r.fraccion_de_referencia
    ref = f" (Q-Learning logra el {fraccion:.1%} de la referencia)" if fraccion is not None else ""
    aviso = (
        " Advertencia: los precios de evaluación son SIMULADOS, no datos reales de XM."
        if r.fuente_datos == "simulado"
        else f" Fuente de precios: {r.fuente_datos} (datos reales)."
    )
    return (
        f"Simulación completada para batería de {r.capacidad_kwh:,.1f} kWh "
        f"({r.energia_por_nivel_kwh:,.2f} kWh por escalón de SoC) sobre {r.n_dias} días "
        f"de evaluación ({r.fecha_inicio} a {r.fecha_fin}). "
        f"Ahorro total usando Q-Learning: ${r.ahorro_qlearning_cop:,.0f} COP. "
        f"Ahorro Línea Base: ${r.ahorro_base_cop:,.0f} COP. "
        f"Ahorro Iteración de Valores (referencia): ${r.ahorro_referencia_cop:,.0f} COP{ref}. "
        f"Diferencia Q-Learning menos Línea Base: ${r.diferencia_cop:,.0f} COP{mejora}. "
        f"Q-Learning hizo {r.cargas_qlearning} cargas, {r.descargas_qlearning} descargas y "
        f"{r.invalidas_qlearning} acciones inválidas.{aviso}"
    )


TOOLS: list[BaseTool] = [query_creg_regulation, simular_despacho]


# --------------------------------------------------------------------------- #
# Errores y trazas
# --------------------------------------------------------------------------- #
class AgentError(RuntimeError):
    """Error de configuración o de comunicación con el proveedor LLM."""


@dataclass
class PasoTraza:
    """Registro de una llamada a herramienta dentro del loop (para depurar/dashboard)."""

    iteracion: int
    herramienta: str
    argumentos: dict[str, Any]
    observacion: str
    error: bool


@dataclass
class ResultadoChat:
    """Resultado completo del último turno."""

    respuesta: str
    iteraciones: int
    pasos: list[PasoTraza] = field(default_factory=list)
    limite_alcanzado: bool = False


# --------------------------------------------------------------------------- #
# Agente
# --------------------------------------------------------------------------- #
def _crear_llm_groq(modelo: str, temperatura: float) -> BaseChatModel:
    """Crea el cliente ChatGroq. Falla con un mensaje claro si falta la key o la lib."""
    try:
        from dotenv import load_dotenv

        load_dotenv()  # permite definir GROQ_API_KEY en un archivo .env
    except ImportError:
        pass

    if not os.getenv("GROQ_API_KEY"):
        raise AgentError(
            "Falta la variable de entorno GROQ_API_KEY. Créala en https://console.groq.com/keys "
            "y defínela en el entorno o en un archivo .env en la raíz del proyecto."
        )
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise AgentError(
            "No está instalado langchain-groq. Ejecuta: python -m pip install langchain-groq"
        ) from exc

    return ChatGroq(model=modelo, temperature=temperatura, max_retries=2, timeout=60)


def _texto(mensaje: BaseMessage) -> str:
    """Extrae el texto de un mensaje (el contenido puede ser str o lista de bloques)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido.strip()
    partes = [
        bloque.get("text", "") if isinstance(bloque, dict) else str(bloque)
        for bloque in contenido
    ]
    return "".join(partes).strip()


class RegulatoryAgent:
    """Agente LLM con loop agéntico manual, límite T_MAX y errores como observaciones.

    Parameters
    ----------
    modelo:
        ID del modelo en Groq. Por defecto la variable GROQ_MODEL o ``DEFAULT_MODEL``.
    temperatura:
        Temperatura de muestreo del LLM.
    t_max:
        Máximo de iteraciones LLM→herramienta por turno.
    tools:
        Herramientas disponibles. Por defecto ``TOOLS``.
    llm:
        Modelo ya construido (inyección para pruebas). Si es ``None`` se crea ChatGroq.
    """

    def __init__(
        self,
        modelo: str | None = None,
        temperatura: float = 0.2,
        t_max: int = T_MAX,
        tools: list[BaseTool] | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        if t_max < 1:
            raise ValueError("t_max debe ser >= 1.")
        self.modelo: str = modelo or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.t_max: int = t_max
        self.tools: dict[str, BaseTool] = {t.name: t for t in (tools or TOOLS)}

        self._llm: BaseChatModel = llm or _crear_llm_groq(self.modelo, temperatura)
        self._llm_con_tools = self._llm.bind_tools(list(self.tools.values()))

        self.historial: list[BaseMessage] = []
        self.ultimo_resultado: ResultadoChat | None = None

    # ------------------------------------------------------------------ #
    def reiniciar(self) -> None:
        """Borra la memoria de la conversación."""
        self.historial.clear()
        self.ultimo_resultado = None

    def chat(self, mensaje: str) -> str:
        """Procesa un mensaje del usuario y devuelve la respuesta final del agente.

        Raises
        ------
        ValueError
            Si el mensaje está vacío.
        AgentError
            Si falla la comunicación con el proveedor LLM (red, key, cuota).
            Los errores de herramientas NO se propagan: vuelven al LLM como observación.
        """
        return self.chat_detallado(mensaje).respuesta

    def chat_detallado(self, mensaje: str) -> ResultadoChat:
        """Igual que ``chat`` pero devuelve también la traza de herramientas."""
        texto_usuario = (mensaje or "").strip()
        if not texto_usuario:
            raise ValueError("El mensaje está vacío.")

        humano = HumanMessage(content=texto_usuario)
        mensajes: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            *self.historial,
            humano,
        ]
        pasos: list[PasoTraza] = []

        for iteracion in range(1, self.t_max + 1):
            ai = self._invocar(self._llm_con_tools, mensajes)
            mensajes.append(ai)

            if not ai.tool_calls:
                return self._cerrar_turno(humano, _texto(ai), iteracion, pasos, False)

            for llamada in ai.tool_calls:
                observacion, hubo_error = self._ejecutar_herramienta(
                    llamada["name"], llamada.get("args") or {}
                )
                pasos.append(
                    PasoTraza(iteracion, llamada["name"], llamada.get("args") or {},
                              observacion, hubo_error)
                )
                mensajes.append(
                    ToolMessage(
                        content=observacion,
                        tool_call_id=llamada["id"],
                        name=llamada["name"],
                        status="error" if hubo_error else "success",
                    )
                )

        # T_MAX agotado: forzar una respuesta final sin herramientas.
        mensajes.append(
            HumanMessage(
                content=(
                    f"Se alcanzó el límite de {self.t_max} iteraciones. Responde ahora al "
                    "usuario solo con la información obtenida, sin llamar herramientas, e "
                    "indica si la respuesta quedó incompleta."
                )
            )
        )
        ai_final = self._invocar(self._llm, mensajes)
        return self._cerrar_turno(humano, _texto(ai_final), self.t_max, pasos, True)

    # ------------------------------------------------------------------ #
    def _invocar(self, modelo: Any, mensajes: list[BaseMessage]) -> AIMessage:
        """Llama al LLM y convierte cualquier fallo del proveedor en AgentError."""
        try:
            respuesta = modelo.invoke(mensajes)
        except Exception as exc:  # noqa: BLE001 - errores de red/HTTP del proveedor
            raise AgentError(f"Error al consultar el modelo '{self.modelo}': {exc}") from exc
        if not isinstance(respuesta, AIMessage):
            raise AgentError(f"Respuesta inesperada del modelo: {type(respuesta).__name__}")
        return respuesta

    def _ejecutar_herramienta(self, nombre: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Ejecuta una herramienta. Nunca lanza: los errores se devuelven como texto."""
        herramienta = self.tools.get(nombre)
        if herramienta is None:
            disponibles = ", ".join(self.tools) or "ninguna"
            return (f"ERROR: la herramienta '{nombre}' no existe. Disponibles: {disponibles}.",
                    True)
        try:
            resultado = herramienta.invoke(args)
        except Exception as exc:  # noqa: BLE001 - el error se devuelve como observación
            return f"ERROR al ejecutar {nombre}: {type(exc).__name__}: {exc}", True

        if isinstance(resultado, str):
            # Las herramientas reportan fallos esperados como texto "ERROR: ...".
            return resultado, resultado.startswith(PREFIJO_ERROR)
        try:
            return json.dumps(resultado, ensure_ascii=False, default=str), False
        except (TypeError, ValueError):
            return str(resultado), False

    def _cerrar_turno(
        self,
        humano: HumanMessage,
        respuesta: str,
        iteraciones: int,
        pasos: list[PasoTraza],
        limite: bool,
    ) -> ResultadoChat:
        """Guarda el turno en memoria (solo pregunta y respuesta final)."""
        respuesta = respuesta or "No obtuve una respuesta del modelo. Intenta reformular."
        self.historial.extend([humano, AIMessage(content=respuesta)])
        self.historial = self.historial[-2 * MAX_TURNOS_HISTORIAL:]
        self.ultimo_resultado = ResultadoChat(respuesta, iteraciones, pasos, limite)
        return self.ultimo_resultado


# --------------------------------------------------------------------------- #
# Resumen ejecutivo (llamada independiente, sin herramientas ni memoria)
# --------------------------------------------------------------------------- #
PROMPT_RESUMEN = """Eres un analista financiero de energía. Analiza estos resultados de \
despacho BESS y redacta un párrafo de 4 líneas explicando a la gerencia el desempeño de la \
política Q-Learning frente a la Línea Base, justificando la eficiencia y el ahorro total en \
COP.

Reglas:
- Usa solo las cifras del JSON, tal como vienen. No calcules cifras nuevas ni inventes datos.
- Si Q-Learning ahorra menos que la Línea Base, dilo con claridad, sin suavizarlo.
- Menciona la eficiencia de ida y vuelta y el periodo evaluado.
- Escribe en español, en un solo párrafo de unas 4 líneas, sin títulos ni viñetas.
- Cierra con una frase corta que aclare que es una simulación con precios de bolsa de XM y no \
una recomendación de inversión."""


def generar_resumen_ejecutivo(
    datos: dict[str, Any],
    modelo: str | None = None,
    llm: BaseChatModel | None = None,
    temperatura: float = 0.3,
) -> str:
    """Redacta con el LLM un resumen ejecutivo de una simulación.

    ``datos`` son las cifras ya calculadas (``ResultadoSimulacion.resumen()``). Es una
    llamada aparte: no usa herramientas ni toca la memoria de ``RegulatoryAgent``.

    Raises
    ------
    ValueError
        Si no hay datos.
    AgentError
        Si falta la configuración del proveedor o falla la llamada.
    """
    if not datos:
        raise ValueError("No hay resultados para resumir.")
    nombre = modelo or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    cliente = llm or _crear_llm_groq(nombre, temperatura)
    mensajes: list[BaseMessage] = [
        SystemMessage(content=PROMPT_RESUMEN),
        HumanMessage(
            content="Resultados de la simulación (JSON):\n"
            + json.dumps(datos, ensure_ascii=False, indent=1, default=str)
        ),
    ]
    try:
        respuesta = cliente.invoke(mensajes)
    except Exception as exc:  # noqa: BLE001 - errores de red/HTTP del proveedor
        raise AgentError(f"Error al consultar el modelo '{nombre}': {exc}") from exc
    texto = _texto(respuesta) if isinstance(respuesta, BaseMessage) else str(respuesta).strip()
    if not texto:
        raise AgentError("El modelo no devolvió texto para el resumen.")
    return texto


def _main(argv: list[str]) -> int:
    pregunta = " ".join(argv) or "¿Qué regula la CREG sobre almacenamiento con baterías?"
    try:
        agente = RegulatoryAgent()
        resultado = agente.chat_detallado(pregunta)
    except (AgentError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    for paso in resultado.pasos:
        estado = "error" if paso.error else "ok"
        print(f"[iter {paso.iteracion}] {paso.herramienta}({paso.argumentos}) -> {estado}")
    print(resultado.respuesta)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
