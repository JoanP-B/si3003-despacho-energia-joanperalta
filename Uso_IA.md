# Declaración de uso de IA generativa

Proyecto Final — SI3003 Inteligencia Artificial — Universidad EAFIT — 2026-1
Idea 3.3: Despacho de energía con RL y asistente sobre regulación.

Conforme al requisito 8 del enunciado y al pacto pedagógico, se declara de forma transparente el
uso de herramientas de IA generativa durante el desarrollo del proyecto.

## Herramientas utilizadas

| Herramienta | Propósito |
|-------------|-----------|
| **Kiro (asistente de código basado en LLM)** | Generación de la estructura base del repositorio, borradores de documentación (README, este archivo), scaffolding de notebooks y módulos, y apoyo puntual en depuración. |

## Qué se hizo con la IA

Registro de los usos concretos. Se mantiene por honestidad académica: la IA **asiste**, no
reemplaza la comprensión del equipo.

### 1. Estructura inicial del proyecto (este commit)

- **Prompt (resumido):** "Lee el PDF del enunciado y dame la estructura base de la idea 3.3
  (Despacho de energía con RL y asistente sobre regulación). Avanza sobre el primer hito:
  conformación del equipo, idea elegida (README.md), datos descargados y primer EDA. Integrante:
  Joan Stiven Peralta Bedoya. Crea además un `Uso_IA.md` declarando el uso de IA, e indícame las
  herramientas de entorno a instalar (de preferencia en un `.venv`)."
- **Resultado generado por la IA:**
  - Extracción del texto del PDF del enunciado para identificar los requisitos de la idea 3.3.
  - `README.md` con el problema, las fuentes de datos (XM vía `pydataxm`, IDEAM, corpus CREG/XM),
    la arquitectura (Parte 1 RL + Parte 2 agente LLM/RAG + dashboard) y las instrucciones de
    ejecución.
  - Estructura de carpetas (`data/`, `notebooks/`, `src/`, `corpus_regulacion/`, `docs/`).
  - `requirements.txt` con el stack sugerido por el enunciado.
  - `notebooks/01_eda.ipynb`: notebook de primer análisis exploratorio de datos del precio de
    bolsa, con plan B (CSV local) si la API de XM falla.
  - Este archivo `Uso_IA.md`.

### 2. (Registrar aquí los usos siguientes)

A medida que avance el proyecto, se irán anotando los prompts y resultados relevantes:
implementación del MDP/Q-learning, diseño del agente, RAG, dashboard, etc.

| Fecha | Componente | Qué pidió el equipo | Qué revisó/ajustó el equipo |
|-------|-----------|---------------------|-----------------------------|
| 2026-11-05 | Estructura base + EDA | Scaffolding del repo e idea 3.3 | Validación de fuentes de datos y de requisitos |

## Compromiso del equipo

- El código de la **Parte 1 (RL)** es **implementado y comprendido por el equipo**, no solo
  invocado desde una librería que ya lo resuelve (requisito 2 del enunciado).
- Todo texto o código generado por IA fue **revisado, entendido y adaptado** por el equipo antes
  de integrarse.
- No hay dependencia total de la herramienta; la IA se usa como apoyo de productividad.
