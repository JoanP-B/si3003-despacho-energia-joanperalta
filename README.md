# Despacho de energía con RL y asistente sobre regulación

Proyecto Final — SI3003 Inteligencia Artificial — Universidad EAFIT — 2026-1

Idea **3.3** del enunciado: decidir **cuándo cargar o descargar una batería** (o desplazar
demanda) según el **precio de bolsa** de energía en Colombia, combinando un modelo clásico de
**Aprendizaje por Refuerzo (RL)** con un **agente LLM + RAG** sobre regulación del sector.

## Integrantes

- Joan Stiven Peralta Bedoya

## Asunto del correo de entrega

`si3003-AI-grupo-peralta`

> Pendiente de confirmar el nombre de grupo definitivo. Cada integrante debe enviar el
> repositorio a `jipadillab@eafit.edu.co`.

---

## 1. Problema

El precio de bolsa de energía en Colombia varía hora a hora. Un operador con una batería
(almacenamiento) puede ahorrar dinero si **carga cuando el precio es bajo** y **descarga cuando
el precio es alto**, respetando los límites físicos de la batería. El reto es encontrar una
**política de operación** que maximice el ahorro frente a políticas de referencia, y explicar
esas decisiones a la luz de la regulación vigente.

## 2. Datos (reales, Colombia)

| Fuente | Qué aporta | Acceso |
|--------|-----------|--------|
| **XM** (operador del mercado) | Demanda, **precio de bolsa**, generación | Librería [`pydataxm`](https://github.com/EquipoAnaliticaXM/API_XM) (API pública) |
| **IDEAM** | Radiación solar (opcional, para demanda/generación solar) | datos.gov.co / DHIME |
| **Corpus regulatorio** | Resoluciones **CREG** y documentos de **XM** | PDFs descargados → `corpus_regulacion/` |

**Plan B (requisito de las primeras 48 h):** si la API de XM falla, se usan históricos de precio
de bolsa descargados en `data/raw/` como CSV. Ver `notebooks/01_eda.ipynb`.

> **Verificación de acceso (2026-11-05):** `pydataxm` quedó instalada y el host
> `servapibi.xm.com.co` es alcanzable, pero la primera llamada falló por verificación de
> certificado SSL del entorno Windows (`CERTIFICATE_VERIFY_FAILED`). Es un problema de CA del
> entorno, no del código. Opciones: instalar/actualizar certificados de CA, o usar el plan B
> con un CSV en `data/raw/precio_bolsa.csv`. El módulo `src/data/download_xm.py` ya cae
> automáticamente al plan B ante cualquier fallo de la API.

## 3. Arquitectura de la solución

```
                 ┌─────────────────────────────────────────────┐
                 │                 Dashboard                    │
                 │        (Streamlit: precio vs decisión,       │
                 │   ahorro acumulado, parámetros batería, chat)│
                 └───────────────┬──────────────┬───────────────┘
                                 │              │
                   ┌─────────────▼───┐    ┌─────▼───────────────┐
                   │  Parte 1: RL    │    │  Parte 2: Agente LLM│
                   │  MDP / Q-learning│   │  loop + RAG + tools │
                   │  (propio)       │    │                     │
                   └─────────┬───────┘    └──────┬──────────────┘
                             │                   │
                   ┌─────────▼───────────────────▼──────────────┐
                   │        Datos XM (pydataxm) + CSV plan B     │
                   │        Corpus CREG/XM (FAISS/Chroma)        │
                   └─────────────────────────────────────────────┘
```

### Parte 1 — Técnica clásica (implementada por el equipo)

- **MDP** del problema de despacho de batería: estado (hora, nivel de carga, precio), acciones
  (cargar / descargar / mantener), recompensa (ahorro monetario), con **iteración de valor** o
  **Q-learning** implementados desde cero.
- Comparación obligatoria contra una **política ingenua** y una **política de referencia**, con
  métrica cuantitativa (ahorro acumulado, % de mejora).

### Parte 2 — Agente LLM

- **Loop agéntico** (manual u opcional LangGraph) con endpoint compatible con OpenAI de capa
  gratuita (NVIDIA NIM / Groq / Ollama).
- **RAG** sobre resoluciones CREG y documentos de XM (embeddings + FAISS o Chroma).
- **Herramientas (≥ 2):**
  - `simulate(policy, dates)` — simula una política sobre un rango de fechas y devuelve ahorro.
  - `search_docs(query)` — recupera y cita fragmentos del corpus regulatorio.
  - (opcional) `get_price(date)` — consulta el precio de bolsa.

### Dashboard

- Serie de tiempo de **precio vs. decisiones** de la política.
- **Ahorro acumulado**.
- Controles para que el usuario modifique **parámetros de la batería** (capacidad, potencia,
  eficiencia) y **converse con el agente**.

## 4. Estructura del repositorio

```
.
├── data/
│   ├── raw/                  # datos crudos descargados (CSV plan B, dumps XM)
│   └── processed/            # datos limpios listos para el MDP/EDA
├── notebooks/
│   └── 01_eda.ipynb          # primer EDA (hito Mié 4 - Jue 5 nov)
├── corpus_regulacion/        # PDFs de CREG y XM para el RAG
├── src/
│   ├── data/                 # descarga y limpieza (pydataxm + plan B)
│   ├── part1_rl/             # MDP / Q-learning (código propio)
│   ├── agent/                # loop del agente LLM y tool calling
│   ├── rag/                  # ingestión del corpus + índice vectorial
│   └── dashboard/            # app Streamlit
├── docs/
│   └── informe.md            # informe breve (máx. 4 páginas)
├── requirements.txt
├── Uso_IA.md                 # declaración de uso de IA generativa
└── README.md
```

## 5. Cómo ejecutar (reproducible)

```bash
# 1. Crear entorno virtual
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
# source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar / preparar datos (ver notebook de EDA)
jupyter notebook notebooks/01_eda.ipynb

# 4. (más adelante) lanzar el dashboard
streamlit run src/dashboard/app.py
```

## 6. Cronograma (del enunciado)

| Fecha | Hito | Estado |
|-------|------|--------|
| Mié 4 – Jue 5 nov | Equipo, idea elegida, datos descargados, primer EDA | **En curso** |
| Vie 6 – Dom 8 nov | Modelo Parte 1 con métrica y línea base | Pendiente |
| Lun 9 nov | Checkpoint: agente mínimo + dashboard esqueleto | Pendiente |
| Lun 9 – Mié 11 nov | RAG, herramientas, visualizaciones, interacción | Pendiente |
| Jue 12 nov | Congelar código, README, declaración de IA | Pendiente |
| Vie 13 nov 6:00 p.m. | Entrega final en GitHub | Pendiente |

## 7. Limitaciones

- El sistema es una herramienta analítica y educativa; **no** constituye asesoría regulatoria,
  legal ni financiera.
- El reto principal es la correcta **definición del entorno MDP** (es la opción técnicamente más
  rigurosa del enunciado).

## 8. Declaración de uso de IA

Ver [`Uso_IA.md`](Uso_IA.md).
