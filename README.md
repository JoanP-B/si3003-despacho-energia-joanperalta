# Despacho de energía con RL y asistente sobre regulación

Proyecto Final — SI3003 Inteligencia Artificial — Universidad EAFIT — 2026-1

## Integrantes

- Joan Stiven Peralta Bedoya

## 1. Problema

El precio de bolsa de energía en Colombia, administrado por XM, cambia cada hora según la
oferta hídrica, el clima, la demanda y la disponibilidad del sistema. Para un consumidor con
una batería (BESS), comprar energía de la red en las horas caras incrementa los costos.

El proyecto resuelve un problema de **desplazamiento de demanda** (*load shifting*): decidir,
hora a hora, cuándo cargar la batería con energía barata de la red y cuándo descargarla para
cubrir la demanda propia en horas caras. No es un bot de especulación: el sistema no vende
energía a la red, solo evita comprarla cuando es cara.

Toda la operación debe respetar la regulación de la CREG (Comisión de Regulación de Energía y
Gas), por lo que el sistema incluye un copiloto que consulta las resoluciones vigentes.

## 2. Arquitectura de la solución

```
        ┌──────────────────────────────┐    ┌──────────────────────────────┐
        │ Precios XM (pydataxm)        │    │ Corpus CREG (PDF / HTML)     │
        │ + CSV de respaldo (data/raw) │    │ corpus_regulacion/           │
        └──────────────┬───────────────┘    └──────────────┬───────────────┘
                       │                                   │ PyPDFLoader + splitter
        ┌──────────────▼───────────────┐    ┌──────────────▼───────────────┐
        │ Parte 1: MDP + Q-Learning    │    │ Índice FAISS local           │
        │ (implementación propia)      │    │ all-MiniLM-L6-v2             │
        │ → data/processed/q_table.json│    │ → data/processed/faiss_index │
        └──────────────┬───────────────┘    └──────────────┬───────────────┘
                       │ simular_despacho                  │ query_creg_regulation
                       └─────────────────┬─────────────────┘
                         ┌───────────────▼───────────────┐
                         │ Parte 2: Agente LLM (Groq)    │
                         │ loop manual, T_max = 5        │
                         └───────────────┬───────────────┘
                         ┌───────────────▼───────────────┐
                         │ Dashboard (Streamlit) + chat  │
                         └───────────────────────────────┘
```

La separación es estricta: **el LLM no hace cálculos**. Las cifras de ahorro las produce
siempre el simulador determinista de la Parte 1, y las afirmaciones normativas salen de los
fragmentos que devuelve el RAG. El LLM solo decide qué herramienta usar y redacta la respuesta.

### Parte 1: motor de decisión (MDP + Q-Learning tabular)

Implementado desde cero en Python, sin Gymnasium ni Stable-Baselines (`src/part1_rl/`).

| Elemento | Definición |
|---|---|
| Estado | `(hora_del_dia 0-23, cuartil_de_precio 0-3, nivel_soc 0-4)` → 480 estados |
| Acciones | `0: Mantener`, `1: Cargar`, `2: Descargar` |
| Transición | Determinista. Cargar al 100 % o descargar al 0 % se invalida y no cambia el estado |
| Recompensa | Cargar: `-precio`; Descargar: `+precio` (costo evitado); Mantener: `0`; acción inválida: `-1000` |
| Episodio | Un día real del historial (24 pasos de 1 hora); la batería arranca vacía |
| Aprendizaje | `Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',·) − Q(s,a)]`, ε-greedy con decaimiento lineal |
| Línea Base | Carga a las 02:00 y descarga a las 19:00, sin mirar el precio |

Hiperparámetros: 500 episodios, α = 0.7, γ = 0.99, ε de 1.0 a 0.05 en 400 episodios. Los
cuartiles de precio se calculan solo con los días de entrenamiento y se reutilizan en la
evaluación para no filtrar información.

### Parte 2: copiloto regulatorio y operativo (LLM + RAG + herramientas)

- **LLM:** Groq con el modelo `openai/gpt-oss-120b` (configurable con la variable
  `GROQ_MODEL`). Se eligió Groq por costo (tiene capa gratuita) y por velocidad de inferencia.
  Los modelos `llama3-70b-8192` y `llama-3.3-70b-versatile` fueron retirados por Groq, y
  `gpt-oss-120b` es el reemplazo que recomienda.
- **Loop agéntico manual** (`src/agent/llm_agent.py`): máximo 5 iteraciones por pregunta. Si
  una herramienta falla, el error vuelve al LLM como observación para que lo explique o corrija
  sus argumentos; la aplicación no se cae.
- **Herramientas:**
  - `simular_despacho(capacidad_kwh)`: carga la Q-Table entrenada y simula los días de
    evaluación con Q-Learning y con la Línea Base. Devuelve el ahorro de cada una en COP.
  - `query_creg_regulation(topic)`: devuelve los 3 fragmentos más relevantes del corpus CREG,
    cada uno con su archivo y página.
- **RAG:** embeddings locales `sentence-transformers/all-MiniLM-L6-v2` (sin costo de API) e
  índice **FAISS** guardado en disco. Fragmentos de 1000 caracteres con 200 de traslape.

### Dashboard

Aplicación Streamlit (`src/dashboard/app.py`) con controles de capacidad de batería
(0 a 1000 kWh) y rango de fechas, y un chat con el copiloto que muestra qué herramientas usó en
cada respuesta. La gráfica de resultados del despacho todavía muestra **datos de ejemplo**;
las cifras reales se obtienen hoy desde el chat (`simular_despacho`) o desde el notebook.

## 3. Datos

| Fuente | Qué aporta | Acceso |
|--------|-----------|--------|
| **XM** (operador del mercado) | Precio de bolsa horario (`PrecBolsNaci`) | Librería [`pydataxm`](https://github.com/EquipoAnaliticaXM/API_XM), con CSV de respaldo en `data/raw/` |
| **Resolución CREG 098 de 2019** | Incorporación de sistemas de almacenamiento con baterías (SAEB) | Versión HTML del [gestor normativo](https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0098_2019.htm), incluida en el repositorio |
| **Resolución CREG 174 de 2021** | Autogeneración a pequeña escala y generación distribuida | PDF oficial; se descarga en el paso 4 de la instalación |

El PDF oficial de la CREG 098 es un escaneo sin capa de texto, por eso se indexa su versión
HTML. Las fuentes del corpus están documentadas en
[`corpus_regulacion/README.md`](corpus_regulacion/README.md).

## 4. Estructura del repositorio

```
.
├── corpus_regulacion/
│   ├── README.md                       # fuentes y URLs del corpus
│   ├── resolucion_creg_0098_2019.htm   # CREG 098/2019 (versión con texto)
│   └── Creg174-2021.pdf                # CREG 174/2021 (se descarga; ignorado por git)
├── data/
│   ├── raw/                            # CSV de respaldo de XM (ignorado por git)
│   └── processed/
│       ├── precios_evaluacion.csv      # días de evaluación de la Parte 1
│       ├── q_table.json                # Q-Table entrenada (se genera; ignorada por git)
│       └── faiss_index/                # índice vectorial (se genera; ignorado por git)
├── docs/
│   ├── Informe Final - SI3003.pdf      # informe del proyecto
│   ├── Analisis.pdf                    # análisis de resultados
│   ├── evaluacion_agente.md            # reporte generado por la batería de QA
│   └── Uso_IA.md                       # declaración de uso de IA generativa
├── notebooks/
│   ├── 01_eda.ipynb                    # análisis exploratorio del precio de bolsa
│   └── 02_mdp_rl.ipynb                 # entrenamiento y evaluación del Q-Learning
├── src/
│   ├── data/download_xm.py             # descarga de XM con respaldo CSV
│   ├── part1_rl/
│   │   ├── environment.py              # BatteryEnv (MDP) y política Línea Base
│   │   ├── agent.py                    # QLearningAgent (tabular, ε-greedy)
│   │   └── simulacion.py               # simulación con la Q-Table entrenada
│   ├── rag/vectorstore.py              # ingesta del corpus e índice FAISS
│   ├── agent/llm_agent.py              # RegulatoryAgent, loop y herramientas
│   └── dashboard/app.py                # aplicación Streamlit
├── tests/
│   └── test_agent.py                   # batería de QA automatizada del agente
├── requirements.txt
└── README.md
```

## 5. Configuración y ejecución

Probado en Linux con Python 3.14. Se requiere Python 3.10 o superior y conexión a internet
(para Groq y para descargar el modelo de embeddings la primera vez). Todos los comandos se
ejecutan desde la raíz del repositorio.

### Paso 1. Prerrequisitos del sistema

**Windows:** instala Python desde [python.org](https://www.python.org/downloads/) y marca la
opción *Add python.exe to PATH* durante la instalación. Verifica en PowerShell:

```powershell
py --version
```

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

**Arch Linux:**

```bash
sudo pacman -S --needed python python-pip git
```

### Paso 2. Crear y activar el entorno virtual

El entorno virtual evita instalar paquetes en el Python del sistema.

**Windows (PowerShell):**

```powershell
py -m venv .venv
# Si PowerShell bloquea la activación, permite scripts solo para esta sesión:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En la shell fish se usa `source .venv/bin/activate.fish`. Con el entorno activo, el prompt
muestra `(.venv)`.

### Paso 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
# Opcional y recomendado: PyTorch solo CPU (evita descargar varios GB de librerías CUDA)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`sentence-transformers` depende de PyTorch. Si se omite la línea opcional, pip instala la
versión con soporte de GPU, que en Linux pesa varios GB.

### Paso 4. Preparar los artefactos

La Q-Table, el índice FAISS y el PDF de la CREG 174 no están en el repositorio por tamaño o
porque se generan. Hay que crearlos una vez:

**4.1. Descargar la Resolución CREG 174 de 2021**

Windows (PowerShell):

```powershell
Invoke-WebRequest -OutFile corpus_regulacion\Creg174-2021.pdf -Uri 'https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/c99b2f316a59fffb052587950077d9d1/$FILE/Creg174-2021.pdf'
```

Linux:

```bash
curl -L -o corpus_regulacion/Creg174-2021.pdf 'https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/c99b2f316a59fffb052587950077d9d1/$FILE/Creg174-2021.pdf'
```

**4.2. Construir el índice FAISS** (la primera vez descarga el modelo de embeddings, unos 90 MB):

```bash
python -m src.rag.vectorstore
# Consulta de prueba:
python -m src.rag.vectorstore --buscar "almacenamiento con baterías"
```

**4.3. Entrenar el agente de RL** (genera `data/processed/q_table.json` y
`data/processed/precios_evaluacion.csv`):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_mdp_rl.ipynb
```

También se puede abrir con `jupyter notebook notebooks/02_mdp_rl.ipynb` y ejecutar todas las
celdas. Por defecto usa precios simulados; para usar datos de XM se pone `USAR_XM = True` en la
primera celda de código.

### Paso 5. Configurar la API key de Groq

Ver la sección [Seguridad y variables de entorno](#seguridad-y-variables-de-entorno).

### Paso 6. Ejecutar

| Qué | Comando |
|---|---|
| Dashboard | `streamlit run src/dashboard/app.py` (abre `http://localhost:8501`) |
| Agente por consola | `python -m src.agent.llm_agent "¿Qué dice la CREG sobre los SAEB?"` |
| Batería de QA | `python tests/test_agent.py` (regenera `docs/evaluacion_agente.md`) |
| Descarga de XM | `python -m src.data.download_xm --inicio 2024-01-01 --fin 2024-03-31` |

### Seguridad y variables de entorno

Por buenas prácticas de seguridad, **las claves privadas no se suben al repositorio**. El
archivo `.env` está en `.gitignore`, así que cada evaluador debe crear su propia clave. La
capa gratuita de Groq es suficiente para ejecutar el proyecto.

1. Entra a [console.groq.com](https://console.groq.com) y crea una cuenta gratuita (con
   correo, Google o GitHub).
2. En el menú lateral, abre **API Keys** y pulsa **Create API Key**.
3. Ponle un nombre (por ejemplo `si3003-eafit`) y copia la clave, que empieza por `gsk_`.
   Groq solo la muestra una vez; si la pierdes, crea otra.
4. Crea un archivo llamado `.env` en la raíz del proyecto (junto a este `README.md`) con este
   contenido:

   ```dotenv
   GROQ_API_KEY="su_clave_aqui"
   ```

   Desde la terminal:

   ```powershell
   # Windows (PowerShell). -Encoding ascii evita el BOM que agrega UTF-8 en PowerShell 5,
   # que impediría leer el nombre de la variable.
   Set-Content -Path .env -Value 'GROQ_API_KEY="su_clave_aqui"' -Encoding ascii
   ```

   ```bash
   # Linux
   echo 'GROQ_API_KEY="su_clave_aqui"' > .env
   chmod 600 .env   # solo tu usuario puede leerlo
   ```

5. Comprueba que git lo ignora: `git check-ignore .env` debe imprimir `.env`.

El agente carga el `.env` automáticamente con `python-dotenv`. Si la clave falta, el dashboard
abre igual, pero deshabilita el chat y muestra un aviso. Variables opcionales:

| Variable | Uso | Valor por defecto |
|---|---|---|
| `GROQ_API_KEY` | Clave de Groq (obligatoria para el agente) | — |
| `GROQ_MODEL` | Modelo de Groq | `openai/gpt-oss-120b` |
| `EMBEDDING_MODEL` | Modelo de embeddings del RAG (hay que reconstruir el índice si cambia) | `sentence-transformers/all-MiniLM-L6-v2` |

Nunca pegues la clave en el código, en notebooks ni en capturas de pantalla. Si se filtra,
revócala en **API Keys** y genera una nueva.

## 6. Evaluación

### Parte 1: Q-Learning frente a la Línea Base

120 días de precios horarios **simulados**, divididos en orden cronológico: 96 días de
entrenamiento y 24 de evaluación. Resultado sobre los días de evaluación con 1 kWh por escalón
de carga (batería de 4 kWh):

| Política | Ahorro acumulado | Cargas | Descargas | Acciones inválidas |
|---|---|---|---|---|
| Q-Learning | 10,369 COP | 95 | 95 | 0 |
| Línea Base (02:00 / 19:00) | 5,549 COP | 24 | 24 | 0 |

Q-Learning ahorra un 86.9 % más que la Línea Base. El ahorro escala de forma lineal con la
capacidad: con 200 kWh son 518,473 COP frente a 277,449 COP.

Con α = 0.1 y solo 500 episodios el agente casi no aprendía: en un barrido con 5 semillas
el ahorro promedio fue de unos 300 COP, y 0 COP en 2 de ellas. Por eso se usa α = 0.7.
El resumen del barrido está en `notebooks/02_mdp_rl.ipynb`.

### Parte 2: batería de QA automatizada del agente

El agente cuenta con una batería de pruebas automatizadas (`tests/test_agent.py`) que verifica
el uso correcto de herramientas y de citas regulatorias. Ejecuta 12 preguntas fijas contra el
agente real y genera el reporte [`docs/evaluacion_agente.md`](docs/evaluacion_agente.md).

Los criterios son deterministas (no se usa un LLM como juez):

- Las herramientas invocadas deben ser exactamente las esperadas.
- Las respuestas de simulación deben usar la capacidad pedida y contener las cifras exactas que
  calcula el simulador.
- Cada cita regulatoria (archivo y página) debe corresponder a un fragmento que devolvió el RAG.
- Las preguntas trampa deben rechazarse sin inventar datos ni cambiar los valores del usuario.

| Categoría | Casos | Aprobados |
|---|---|---|
| Simulación | 4 | 4 |
| Regulación | 4 | 3 |
| Combinada (simulación + regulación) | 2 | 1 |
| Trampa (capacidad negativa, fuera de tema) | 2 | 2 |
| **Total** | **12** | **10 (83 %)** |

Fallos documentados:

- **R1 ("¿Cómo se define un SAEB?"):** el RAG no recuperó el fragmento con la definición; el
  agente reformuló la búsqueda 5 veces y agotó el límite de iteraciones.
- **C1 (simulación + vida útil):** las cifras fueron exactas, pero la parte normativa citó con
  marcadores `【1†】` en vez del formato con archivo y página.

Los criterios comprueban que cada cita exista entre los fragmentos recuperados, no que la
afirmación sea correcta. El resultado corresponde a una sola ejecución con temperatura 0.2, así
que puede variar entre corridas.

## 7. Limitaciones

- El sistema es una herramienta analítica y educativa; **no** constituye asesoría
  regulatoria, legal ni financiera.
- **Precios simulados:** los resultados reportados de la Parte 1 usan precios sintéticos. El
  código ya admite datos de XM (`USAR_XM = True`), pero las cifras de este README no
  corresponden al mercado real.
- **Volatilidad del mercado:** el agente aprende patrones de un período histórico. Choques como
  El Niño, el nivel de los embalses, fallas de transmisión o intervenciones de la CREG pueden
  hacer que la política aprendida deje de ser buena.
- **Modelo de batería simplificado:** 5 niveles de carga, sin pérdidas de eficiencia ni
  degradación, y un solo escalón de carga o descarga por hora.
- **RAG limitado:** el corpus tiene dos resoluciones, el modelo de embeddings está entrenado
  principalmente en inglés y solo se recuperan 3 fragmentos por consulta. Algunas respuestas
  quedan incompletas (ver caso R1).
- **Dashboard:** la gráfica de resultados aún muestra datos de ejemplo.

## 8. Declaración de uso de IA

Ver [`docs/Uso_IA.md`](docs/Uso_IA.md).
