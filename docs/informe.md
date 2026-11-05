# Informe breve — Despacho de energía con RL y asistente sobre regulación

> Máximo 4 páginas (entregable 2). Este es el esqueleto; se completa al final del proyecto.

## 1. Problema

_(Decidir cuándo cargar/descargar una batería según el precio de bolsa de energía en Colombia.)_

## 2. Datos

_(XM vía `pydataxm`: precio de bolsa, demanda, generación. IDEAM para radiación. Plan B: CSV
local. Describir rango de fechas, limpieza y EDA.)_

## 3. Método de la Parte 1 (RL)

_(MDP del despacho: estado, acciones, recompensa. Iteración de valor / Q-learning implementados
por el equipo. Línea base: política ingenua y de referencia. Métrica: ahorro acumulado.)_

## 4. Diseño del agente y RAG

_(Loop agéntico, herramientas `simulate` y `search_docs`, corpus CREG/XM, embeddings + FAISS.)_

## 5. Resultados

_(Tablas y gráficas: ahorro de la política RL vs. líneas base; evaluación del agente con 10-15
preguntas de prueba indicando herramienta llamada y cita de fuente.)_

## 6. Limitaciones

_(Herramienta analítica/educativa, no asesoría regulatoria ni financiera. Retos del entorno MDP,
calidad de datos, etc.)_
