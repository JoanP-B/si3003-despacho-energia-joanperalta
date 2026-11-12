"""Parte 2 — Agente LLM mínimo ("Copiloto Regulatorio") con loop agéntico manual.

Checkpoint: un agente con UNA herramienta simulada (mock RAG). La búsqueda vectorial
real sobre el corpus CREG se conecta en el siguiente hito.

Loop agéntico (máximo T_MAX iteraciones):
    1. El LLM recibe el historial y decide: responder o pedir una herramienta.
    2. Si pide herramienta, se ejecuta localmente. Cualquier excepción se captura y se
       devuelve al LLM como observación (ToolMessage) para que la explique o corrija.
    3. Se repite hasta obtener una respuesta sin tool calls o agotar T_MAX; en ese caso
       se fuerza una respuesta final sin herramientas.

Proveedor: Groq vía ``langchain-groq`` (requiere la variable de entorno GROQ_API_KEY).

Uso por consola:
    python -m src.agent.llm_agent "¿Qué dice la CREG sobre almacenamiento con baterías?"
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

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

SYSTEM_PROMPT = """Eres el Copiloto Regulatorio y Operativo de un sistema de \
almacenamiento de energía con baterías (BESS) en Colombia. El sistema hace \
desplazamiento de demanda (load shifting): carga la batería en horas de precio de \
bolsa bajo y la descarga para cubrir demanda propia en horas caras. No especula ni \
vende energía a la red.

Reglas:
- Para cualquier pregunta sobre regulación, normas o resoluciones CREG, usa la \
herramienta query_creg_regulation antes de responder y cita la resolución que devuelva.
- Toda afirmación normativa debe salir del texto que devolvió la herramienta. No \
completes con conocimiento propio: no inventes requisitos, cifras, artículos ni \
resoluciones. Si el fragmento no trae el detalle pedido, dilo explícitamente.
- No repitas una consulta con los mismos argumentos; usa el resultado que ya tienes.
- Si una herramienta devuelve un error, explícale al usuario qué falló en lenguaje \
sencillo; si puedes corregir los argumentos, reintenta una vez.
- Responde en español, de forma breve y clara."""


# --------------------------------------------------------------------------- #
# Herramientas
# --------------------------------------------------------------------------- #
@tool
def query_creg_regulation(topic: str) -> str:
    """Consulta el corpus regulatorio de la CREG sobre un tema y devuelve fragmentos
    normativos con su fuente. Úsala para preguntas sobre regulación de baterías
    (BESS), autogeneración, mercado de energía o precio de bolsa.

    Args:
        topic: Tema regulatorio a consultar, por ejemplo "almacenamiento con baterías".
    """
    # MOCK: la búsqueda vectorial (FAISS) sobre corpus_regulacion/ llega en el próximo hito.
    tema = (topic or "").strip()
    if not tema:
        raise ValueError("El parámetro 'topic' está vacío; indica un tema regulatorio.")
    if len(tema) > 300:
        raise ValueError("El tema es demasiado largo (máximo 300 caracteres).")
    return (
        f"Fragmento normativo simulado sobre {tema} según Resolución CREG 098. "
        "[MOCK] Este resultado no contiene texto normativo real: la búsqueda en el "
        "corpus CREG aún no está conectada. Solo se puede afirmar que la Resolución "
        "CREG 098 de 2019 trata sobre sistemas de almacenamiento con baterías (BESS)."
    )


TOOLS: list[BaseTool] = [query_creg_regulation]


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
            return resultado, False
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
