# Declaración de uso de IA generativa

Proyecto Final — SI3003 Inteligencia Artificial — Universidad EAFIT — 2026-1

Conforme al proyecto, se declara de forma transparente el
uso de herramientas de IA generativa durante el desarrollo del proyecto.

## Herramientas utilizadas

| Herramienta | Propósito |
|-------------|-----------|
| **Kiro (asistente de código basado en LLM)** | Generación de la estructura base del repositorio, borradores de documentación (README, este archivo), scaffolding de notebooks y módulos, y apoyo puntual en depuración. |

## Qué se hizo con la IA

Registro de los usos concretos. Se mantiene por honestidad académica: la IA **asiste**, no
reemplaza la comprensión del equipo.

### 1. Estructura inicial del proyecto (este commit)

- **Prompt (resumido):** "Lee el PDF del enunciado y dame la estructura ideal del proyecto 3.3
  (Despacho de energía con RL y asistente sobre regulación). Avanza sobre el primer hito:
  Dame el listado en requirements.txt de las herramientas necesarias para el programa, y un mensaje adicional de
  como instalarlos en sistemas tipo linux."
- **Resultado generado por la IA:**
  - Extracción del texto del PDF del enunciado para identificar los requisitos de la idea 3.3.
  - Estructura de carpetas (`data/`, `notebooks/`, `src/`, `corpus_regulacion/`, `docs/`).
  - `requirements.txt` con el stack sugerido por el enunciado.
  - `notebooks/01_eda.ipynb`: notebook de primer análisis exploratorio de datos del precio de
    bolsa, con plan B (CSV local) si la API de XM falla.

### 2. Registro

A medida que avance el proyecto, se irán anotando los prompts y resultados relevantes:
implementación del MDP/Q-learning, diseño del agente, RAG, dashboard, etc.

| Fecha | Componente | Qué pidió el equipo | Qué revisó/ajustó el equipo |
|-------|-----------|---------------------|-----------------------------|
| 2026-11-05 | Estructura base + EDA | Scaffolding del repo e idea 3.3 | Validación de fuentes de datos y de requisitos |

## Compromiso del equipo

- El código es **implementado y comprendido por el equipo**, no solo
  invocado desde una librería que ya lo resuelve.
- Las pruebas experimentales seran puestas 100% a prueba desde el entorno local, sin pedirle a la IA que lo 
  simule y lo pruebe sin un rendimiento real.
- Todo texto o código generado por IA fue **revisado, entendido y adaptado** por el equipo antes
  de integrarse.
- No hay dependencia total de la herramienta; la IA se usa como apoyo de productividad.