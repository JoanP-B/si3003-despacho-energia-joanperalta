"""Evaluación automatizada del Copiloto Regulatorio (rigor experimental).

Ejecuta una batería fija de 12 preguntas contra el ``RegulatoryAgent`` real (Groq + RAG
FAISS + simulación Q-Learning) y genera ``docs/evaluacion_agente.md``.

Criterios (deterministas, sin un LLM como juez):
    - Herramientas: el conjunto de herramientas invocadas con éxito debe ser EXACTAMENTE
      el esperado (ni faltar ni sobrar).
    - Simulación: el argumento ``capacidad_kwh`` debe ser el pedido, y la respuesta debe
      contener las cifras exactas (en pesos, redondeadas) de ahorro Q-Learning y Línea
      Base. La referencia se calcula llamando directamente a ``simular()``.
    - Regulación: la respuesta debe citar al menos una fuente con el formato del prompt, y
      cada cita (archivo y página) debe estar entre los fragmentos que devolvió el RAG
      (las citas inventadas fallan). Además, el RAG debe recuperar la resolución esperada.
    - Trampas: no debe haber una simulación exitosa con un valor que el usuario no pidió,
      ni herramientas para preguntas fuera de alcance, y la respuesta debe declinar.
    - Ningún caso puede agotar el límite T_MAX.

Uso (desde la raíz del repositorio; requiere GROQ_API_KEY, la Q-Table y el índice FAISS):
    python tests/test_agent.py
    python tests/test_agent.py --pausa 5 --salida docs/evaluacion_agente.md
    pytest tests/test_agent.py -s      # mismo flujo; se omite si falta GROQ_API_KEY
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

SALIDA_POR_DEFECTO = RAIZ / "docs" / "evaluacion_agente.md"
SIM = "simular_despacho"
RAG = "query_creg_regulation"
F098 = "resolucion_creg_0098_2019.htm"
F174 = "Creg174-2021.pdf"


# --------------------------------------------------------------------------- #
# Batería de pruebas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Caso:
    id: str
    categoria: str  # "Simulación" | "Regulación" | "Combinada" | "Trampa"
    pregunta: str
    intencion: str
    herramientas: frozenset[str]
    capacidad_kwh: float | None = None  # simulación esperada
    fuente_esperada: str | None = None  # resolución que el RAG debe recuperar
    trampa: str | None = None  # "capacidad_invalida" | "fuera_de_alcance"


CASOS: list[Caso] = [
    # --- Simulación ---
    Caso("S1", "Simulación", "Simula el despacho para una batería de 200 kWh.",
         "Ahorro Q-Learning vs. Línea Base, 200 kWh", frozenset({SIM}), capacidad_kwh=200),
    Caso("S2", "Simulación",
         "Si instalo una batería de 500 kWh, ¿cuánto ahorro contra la línea base?",
         "Ahorro frente a la Línea Base, 500 kWh", frozenset({SIM}), capacidad_kwh=500),
    Caso("S3", "Simulación",
         "¿Cuánto ahorraría el agente de RL con una batería de 1000 kWh?",
         "Ahorro del agente RL, 1000 kWh", frozenset({SIM}), capacidad_kwh=1000),
    Caso("S4", "Simulación",
         "Compara Q-Learning y la línea base para una batería de 75.5 kWh.",
         "Comparación con capacidad decimal, 75.5 kWh", frozenset({SIM}), capacidad_kwh=75.5),
    # --- Regulación ---
    Caso("R1", "Regulación", "¿Cómo se define un SAEB según la regulación?",
         "Definición de SAEB (CREG 098)", frozenset({RAG}), fuente_esperada=F098),
    Caso("R2", "Regulación", "¿Qué dice la CREG 174 sobre inyección de excedentes a la red?",
         "Excedentes de autogeneración (CREG 174)", frozenset({RAG}), fuente_esperada=F174),
    Caso("R3", "Regulación",
         "¿Cuál es el límite de potencia de un autogenerador a pequeña escala según la CREG?",
         "Límite de potencia AGPE (CREG 174)", frozenset({RAG}), fuente_esperada=F174),
    Caso("R4", "Regulación",
         "¿Con qué propósito permite la Resolución CREG 098 de 2019 instalar sistemas de "
         "almacenamiento?",
         "Objeto de la CREG 098", frozenset({RAG}), fuente_esperada=F098),
    # --- Combinadas ---
    Caso("C1", "Combinada",
         "Simula una batería de 50 kWh y dime qué dice la norma sobre su vida útil.",
         "Simulación 50 kWh + vida útil del SAEB (CREG 098)", frozenset({SIM, RAG}),
         capacidad_kwh=50, fuente_esperada=F098),
    Caso("C2", "Combinada",
         "Con una batería de 300 kWh, ¿cuánto ahorraría y qué regulación aplica a un "
         "autogenerador que entrega excedentes a la red?",
         "Simulación 300 kWh + excedentes (CREG 174)", frozenset({SIM, RAG}),
         capacidad_kwh=300, fuente_esperada=F174),
    # --- Trampas ---
    Caso("T1", "Trampa", "Simula el despacho para una batería de -100 kWh.",
         "Capacidad inválida: debe rechazarla, no cambiarla", frozenset(),
         trampa="capacidad_invalida"),
    Caso("T2", "Trampa", "Dime el clima de mañana en Medellín.",
         "Fuera de alcance: no debe usar herramientas", frozenset(),
         trampa="fuera_de_alcance"),
]


# --------------------------------------------------------------------------- #
# Utilidades de verificación
# --------------------------------------------------------------------------- #
_RE_CITA = re.compile(
    r"Fuente:\s*([\w\-.]+\.(?:pdf|html?))(?:\s*,\s*p[áa]g(?:ina)?s?\.?\s*(\d+))?", re.I
)
_RE_FRAGMENTO = re.compile(r"^\[\d+\] Fuente: (\S+?)(?:, página (\d+))?$", re.M)
_RE_NUMERO = re.compile(r"\d(?:[\d.,\u00a0\u202f]*\d)?")


def _sin_tildes(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )


def citas(texto: str) -> list[tuple[str, int | None]]:
    """Citas (archivo, página) que aparecen en la respuesta."""
    return [(a, int(p) if p else None) for a, p in _RE_CITA.findall(texto)]


def fragmentos_recuperados(observacion: str) -> set[tuple[str, int | None]]:
    """Pares (archivo, página) que devolvió query_creg_regulation."""
    return {(a, int(p) if p else None) for a, p in _RE_FRAGMENTO.findall(observacion)}


def cifra_presente(texto: str, valor: float) -> bool:
    """True si el valor (en pesos enteros) aparece en el texto con cualquier separador.

    Se compara la parte entera exacta. Una cola de 1-2 dígitos tras el último separador
    se trata como decimales ("10,369.49" → 10369); "518.473" se lee como miles.
    """
    objetivos = {str(int(round(valor))), str(int(valor))}
    for token in _RE_NUMERO.findall(texto):
        candidatos = {re.sub(r"\D", "", token)}
        decimal = re.match(r"^(.*\d)[.,]\d{1,2}$", token)
        if decimal:
            candidatos.add(re.sub(r"\D", "", decimal.group(1)))
        if candidatos & objetivos:
            return True
    return False


# --------------------------------------------------------------------------- #
# Resultado por caso
# --------------------------------------------------------------------------- #
@dataclass
class Resultado:
    caso: Caso
    respuesta: str = ""
    herramientas_ok: list[str] = field(default_factory=list)
    herramientas_error: list[str] = field(default_factory=list)
    argumentos: list[str] = field(default_factory=list)
    citas: list[tuple[str, int | None]] = field(default_factory=list)
    criterios: dict[str, bool] = field(default_factory=dict)
    notas: list[str] = field(default_factory=list)
    iteraciones: int = 0
    segundos: float = 0.0
    error_infra: str | None = None

    @property
    def aprobado(self) -> bool:
        return self.error_infra is None and bool(self.criterios) and all(self.criterios.values())


def evaluar(caso: Caso, agente, referencia: Callable[[float], object]) -> Resultado:
    """Ejecuta un caso con memoria limpia y aplica los criterios."""
    from src.agent.llm_agent import AgentError

    res = Resultado(caso)
    agente.reiniciar()
    inicio = time.perf_counter()
    try:
        salida = agente.chat_detallado(caso.pregunta)
    except AgentError as exc:
        res.error_infra = str(exc)
        res.segundos = time.perf_counter() - inicio
        return res
    res.segundos = time.perf_counter() - inicio
    res.respuesta = salida.respuesta
    res.iteraciones = salida.iteraciones

    pasos_ok = [p for p in salida.pasos if not p.error]
    res.herramientas_ok = [p.herramienta for p in pasos_ok]
    res.herramientas_error = [p.herramienta for p in salida.pasos if p.error]
    res.argumentos = [
        f"{p.herramienta}({', '.join(f'{k}={v!r}' for k, v in p.argumentos.items())})"
        + (" → ERROR" if p.error else "")
        for p in salida.pasos
    ]
    res.citas = citas(res.respuesta)
    texto = _sin_tildes(res.respuesta)
    c = res.criterios

    c["sin agotar T_MAX"] = not salida.limite_alcanzado

    if caso.trampa is None:
        c["herramientas exactas"] = set(res.herramientas_ok) == set(caso.herramientas)

    # --- Simulación: argumento correcto y cifras exactas ---
    if caso.capacidad_kwh is not None:
        sims = [p for p in pasos_ok if p.herramienta == SIM]
        c["capacidad pedida"] = any(
            math.isclose(float(p.argumentos.get("capacidad_kwh", math.nan)), caso.capacidad_kwh)
            for p in sims
        )
        ref = referencia(caso.capacidad_kwh)
        c["cifra Q-Learning exacta"] = cifra_presente(res.respuesta, ref.ahorro_qlearning_cop)
        c["cifra Línea Base exacta"] = cifra_presente(res.respuesta, ref.ahorro_base_cop)
        res.notas.append(
            f"Referencia: Q-Learning ${ref.ahorro_qlearning_cop:,.0f} · "
            f"Línea Base ${ref.ahorro_base_cop:,.0f}"
        )

    # --- Regulación: citas verificables y recuperación ---
    if RAG in caso.herramientas:
        recuperados: set[tuple[str, int | None]] = set()
        for p in pasos_ok:
            if p.herramienta == RAG:
                recuperados |= fragmentos_recuperados(p.observacion)
        archivos = {a for a, _ in recuperados}
        c["cita fuentes"] = bool(res.citas)
        c["citas respaldadas por el RAG"] = bool(res.citas) and all(
            a in archivos and (pag is None or (a, pag) in recuperados) for a, pag in res.citas
        )
        if caso.fuente_esperada:
            c[f"RAG recuperó {caso.fuente_esperada}"] = caso.fuente_esperada in archivos
        inventadas = [
            f"{a}" + (f" p.{p}" if p else "") for a, p in res.citas
            if a not in archivos or (p is not None and (a, p) not in recuperados)
        ]
        if inventadas:
            res.notas.append("Citas no respaldadas: " + ", ".join(inventadas))
        res.notas.append(
            "Recuperado: " + (", ".join(sorted(f"{a}" + (f" p.{p}" if p else "")
                                               for a, p in recuperados)) or "nada")
        )

    # --- Trampas ---
    if caso.trampa == "capacidad_invalida":
        c["no simuló con otro valor"] = SIM not in res.herramientas_ok
        c["explica que la capacidad es inválida"] = any(
            k in texto for k in ("mayor que 0", "mayor a 0", "positiv", "negativ", "invalid",
                                 "no es valid", "no puede ser")
        )
    elif caso.trampa == "fuera_de_alcance":
        c["sin herramientas"] = not salida.pasos
        c["declina sin inventar datos"] = (
            any(k in texto for k in ("no tengo", "no puedo", "no dispongo", "fuera de",
                                     "no cuento", "no es posible", "no esta"))
            and "°" not in res.respuesta
        )
    return res


# --------------------------------------------------------------------------- #
# Reporte Markdown
# --------------------------------------------------------------------------- #
def _celda(texto: str, limite: int | None = None) -> str:
    texto = re.sub(r"\s+", " ", texto).strip()
    if limite and len(texto) > limite:
        texto = texto[: limite - 1].rstrip() + "…"
    return texto.replace("|", "\\|")


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "desconocido"


def generar_reporte(resultados: list[Resultado], modelo: str, t_max: int, ruta: Path) -> None:
    total = len(resultados)
    aprobados = sum(r.aprobado for r in resultados)
    categorias: dict[str, list[Resultado]] = {}
    for r in resultados:
        categorias.setdefault(r.caso.categoria, []).append(r)

    lineas = [
        "# Evaluación del Copiloto Regulatorio",
        "",
        f"Generado automáticamente por `tests/test_agent.py` el "
        f"{datetime.now():%Y-%m-%d %H:%M}.",
        "",
        f"- **Modelo:** `{modelo}` (Groq), T_max = {t_max}, una ejecución por pregunta "
        "con memoria limpia.",
        f"- **Commit:** `{_commit()}`",
        f"- **Resultado global:** {aprobados}/{total} casos aprobados "
        f"({aprobados / total:.0%}).",
        "",
        "| Categoría | Aprobados |",
        "|---|---|",
        *[f"| {cat} | {sum(r.aprobado for r in rs)}/{len(rs)} |" for cat, rs in categorias.items()],
        "",
        "## Criterios",
        "",
        "- **Herramientas exactas:** las herramientas invocadas con éxito coinciden con las "
        "esperadas (ni faltan ni sobran).",
        "- **Simulación:** se usó la capacidad pedida y la respuesta contiene las cifras exactas "
        "de ahorro (Q-Learning y Línea Base) que da `simular()` para esa capacidad.",
        "- **Regulación:** la respuesta cita al menos una fuente, cada cita (archivo y página) "
        "corresponde a un fragmento que devolvió el RAG, y el RAG recuperó la resolución "
        "esperada.",
        "- **Trampas:** no simula con un valor distinto al pedido ni usa herramientas fuera de "
        "alcance, y declina explicando el motivo.",
        "- Ningún caso puede agotar el límite de iteraciones.",
        "",
        "## Resultados",
        "",
        "| Pregunta | Intención | Herramienta Invocada | Respuesta y Citas | Estado (Pass/Fail) |",
        "|---|---|---|---|---|",
    ]
    for r in resultados:
        herramientas = "<br>".join(_celda(a) for a in r.argumentos) or "Ninguna"
        if r.error_infra:
            respuesta = f"**Error de infraestructura:** {_celda(r.error_infra, 200)}"
        else:
            respuesta = _celda(r.respuesta, 280)
            if r.citas:
                respuesta += "<br>**Citas:** " + ", ".join(
                    _celda(a + (f" p.{p}" if p else "")) for a, p in r.citas
                )
        fallidos = [k for k, v in r.criterios.items() if not v]
        estado = "✅ Pass" if r.aprobado else "❌ Fail"
        if fallidos:
            estado += "<br>" + "; ".join(_celda(f) for f in fallidos)
        lineas.append(
            f"| **{r.caso.id}.** {_celda(r.caso.pregunta)} | {_celda(r.caso.intencion)} | "
            f"{herramientas} | {respuesta} | {estado} |"
        )

    lineas += ["", "## Detalle por caso", ""]
    for r in resultados:
        lineas += [
            f"### {r.caso.id} — {r.caso.categoria}: {r.caso.pregunta}",
            "",
            f"- **Estado:** {'Pass' if r.aprobado else 'Fail'} · {r.iteraciones} iteración(es) · "
            f"{r.segundos:.1f} s",
            f"- **Herramientas:** {', '.join(f'`{a}`' for a in r.argumentos) or 'ninguna'}",
            *[f"- {'✅' if v else '❌'} {k}" for k, v in r.criterios.items()],
            *[f"- {n}" for n in r.notas],
            "",
        ]
        if r.error_infra:
            lineas += [f"> Error de infraestructura: {r.error_infra}", ""]
        else:
            lineas += ["> " + linea for linea in r.respuesta.splitlines() or [""]]
            lineas.append("")

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
def ejecutar(salida: Path = SALIDA_POR_DEFECTO, pausa: float = 3.0) -> list[Resultado]:
    from src.agent.llm_agent import AgentError, RegulatoryAgent
    from src.part1_rl.simulacion import simular

    agente = RegulatoryAgent()
    resultados: list[Resultado] = []
    for i, caso in enumerate(CASOS, 1):
        res = evaluar(caso, agente, simular)
        if res.error_infra:  # p. ej. límite de tasa de Groq: un reintento tras esperar
            print(f"  [{caso.id}] error de infraestructura, reintento en 30 s: {res.error_infra[:90]}")
            time.sleep(30)
            res = evaluar(caso, agente, simular)
        resultados.append(res)
        estado = "PASS" if res.aprobado else "FAIL"
        fallos = [k for k, v in res.criterios.items() if not v]
        print(f"[{i:2}/{len(CASOS)}] {caso.id} {estado:4} {res.segundos:5.1f}s "
              f"tools={res.herramientas_ok or '-'} {('· falla: ' + '; '.join(fallos)) if fallos else ''}")
        if i < len(CASOS):
            time.sleep(pausa)

    generar_reporte(resultados, agente.modelo, agente.t_max, salida)
    aprobados = sum(r.aprobado for r in resultados)
    print(f"\n{aprobados}/{len(resultados)} casos aprobados. Reporte: {salida}")
    return resultados


def test_bateria_agente() -> None:
    """Punto de entrada para pytest. Exige al menos 10 de 12 casos aprobados."""
    import pytest

    try:
        from dotenv import load_dotenv

        load_dotenv(RAIZ / ".env")
    except ImportError:
        pass
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY no está definida.")
    resultados = ejecutar()
    assert sum(r.aprobado for r in resultados) >= 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa el Copiloto Regulatorio.")
    parser.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    parser.add_argument("--pausa", type=float, default=3.0,
                        help="Segundos entre preguntas (evita límites de tasa de Groq).")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    try:
        resultados = ejecutar(args.salida, args.pausa)
    except Exception as exc:  # noqa: BLE001 - p. ej. falta GROQ_API_KEY
        print(f"[ERROR] No se pudo ejecutar la evaluación: {exc}", file=sys.stderr)
        return 2
    return 0 if all(r.aprobado for r in resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
