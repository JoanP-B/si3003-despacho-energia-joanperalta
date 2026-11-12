# Despacho de energía con RL y asistente sobre regulación

Proyecto Final — SI3003 Inteligencia Artificial — Universidad EAFIT — 2026-1

## Integrantes

- Joan Stiven Peralta Bedoya

## 1. Problema

El precio de bolsa de energía en Colombia varía cada hora. Como es un mercado volátil, no siempre es evidente cuándo conviene comprar o vender. El precio, administrado por XM, depende de factores como la oferta hídrica y el clima, la demanda y la disponibilidad del sistema.

Una empresa con un consumo alto o un operador de baterías enfrenta varios retos:

- Comprar energía cuando el precio es alto incrementa los costos.
- Sin una estrategia clara, una batería puede cargarse en horas caras o descargarse sin obtener un beneficio significativo.
- La regulación de la CREG (Comisión de Regulación de Energía y Gas) establece reglas para operar y comercializar energía.


## 2. Arquitectura de la solución

```
                   ┌─────────▼───────────────────▼──────────────┐
                   │        Datos XM (pydataxm) + CSV plan B     │
                   │        Corpus CREG/XM (FAISS/Chroma)        │
                   └─────────────────────────────────────────────┘
                                 │              │
                   ┌─────────────▼───┐    ┌─────▼───────────────┐
                   │  Parte 1: RL    │    │  Parte 2: Agente LLM│
                   │  MDP / Q-learning│   │  loop + RAG + tools │
                   │  (propio)       │    │                     │
                   └─────────┬───────┘    └──────┬──────────────┘
                             │                   │
                 ┌─────────────────────────────────────────────┐
                 │                 Dashboard                    │
                 │        (Streamlit: precio vs decisión,       │
                 │   ahorro acumulado, parámetros batería, chat)│
                 └───────────────┬──────────────┬───────────────┘

```

### La solución propuesta

El proyecto combina la toma de decisiones mediante aprendizaje por refuerzo con una herramienta de consulta regulatoria en lenguaje natural.

#### Componente A: motor de decisión

- **Técnica:** Proceso de Decisión de Markov (MDP), resuelto mediante Q-learning o iteración de valores.
- **Funcionamiento:** Cada hora, el agente observa el nivel de carga de la batería y el rango de precio, y decide si cargar, descargar o mantener el nivel actual.
- **Objetivo:** Maximizar el ahorro acumulado al evitar comprar energía de la red durante las horas caras.
- **Evaluación:** Comparar el agente de aprendizaje por refuerzo con una política base, por ejemplo, cargar a las 2:00 y descargar a las 19:00.

#### Componente B: copiloto regulatorio

- **Técnica:** Agente basado en un modelo de lenguaje (LLM), con generación aumentada por recuperación (RAG) y uso de herramientas.
- **Funcionamiento:** El sistema indexa resoluciones de la CREG y documentos de XM. El agente puede ejecutar simulaciones y buscar información en el corpus regulatorio.
- **Objetivo:** Responder consultas combinando resultados cuantitativos de las simulaciones con información regulatoria pertinente.

#### Componente C: dashboard

Desarrollado con Streamlit, permite ajustar parámetros como la capacidad de la batería (kWh), seleccionar fechas de prueba, visualizar precios y decisiones de carga o descarga, y consultar al copiloto.


## 3. Datos (reales, Colombia)

| Fuente | Qué aporta | Acceso |
|--------|-----------|--------|
| **XM** (operador del mercado) | Demanda, **precio de bolsa**, generación | Librería [`pydataxm`](https://github.com/EquipoAnaliticaXM/API_XM) (API pública) |
| **IDEAM** | Radiación solar (opcional, para demanda/generación solar) | datos.gov.co / DHIME |
| **Corpus regulatorio** | Resoluciones **CREG** y documentos de **XM** | PDFs descargados → `corpus_regulacion/` |

**Plan B:** si la API de XM falla, se usan históricos de precio de bolsa guardados como CSV en `data/raw/`. Para el análisis exploratorio, consulta [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

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
│   └── informe.md            # informe breve
├── requirements.txt
├── Uso_IA.md                 # declaración de uso de IA generativa
└── README.md
```

## 5. Instalación y ejecución

Ejecuta estos comandos desde la raíz del repositorio.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux

Instala Python, pip y el soporte para entornos virtuales con el gestor de paquetes de tu distribución.

En Debian o Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

En Arch Linux:

```bash
sudo pacman -S --needed python python-pip
```

Desde la raíz del repositorio, crea y activa un entorno virtual e instala las dependencias Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El entorno virtual evita instalar paquetes en el Python administrado por el sistema. En otras distribuciones, instala los paquetes equivalentes a Python 3, pip y `venv` con su gestor de paquetes y luego ejecuta los mismos comandos.

### Preparar datos y ejecutar la aplicación

Con el entorno virtual activado, abre el notebook de análisis exploratorio:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Para descargar el precio de bolsa desde XM, indica las fechas en formato `AAAA-MM-DD`. Si la API no está disponible, el módulo intenta usar el CSV de respaldo en `data/raw/`:

```bash
python -m src.data.download_xm --inicio 2024-01-01 --fin 2024-03-31
```

Para iniciar el dashboard:

```bash
streamlit run src/dashboard/app.py
```

## 6. Limitaciones

- El sistema es una herramienta analítica y educativa; **no** constituye asesoría regulatoria,
  legal ni financiera.
- **Incertidumbre del mercado y volatilidad del precio:** El modelo de aprendizaje por refuerzo (MDP/Q-Learning) se entrena sobre datos históricos del precio de bolsa de XM. Dado que el mercado energético colombiano está expuesto a choques externos e impredecibles (como fenómenos climáticos de El Niño o fallas), las decisiones óptimas aprendidas por el agente en períodos pasados podrían no ser óptimas frente a escenarios atípicos o cambios bruscos en la oferta y demanda.

## 7. Declaración de uso de IA

Ver [`Uso_IA.md`]( /si3003-despacho-energia-joanperalta/docs/Uso_IA.md).
