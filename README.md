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

La operación de baterías está regulada por la CREG (Comisión de Regulación de Energía y Gas),
por lo que el sistema incluye un copiloto que consulta las resoluciones y cita la fuente.

## 2. Arquitectura de la solución

```
        ┌──────────────────────────────┐    ┌──────────────────────────────┐
        │ Precio de bolsa XM (pydataxm)│    │ Corpus CREG (PDF / HTML)     │
        │ + respaldo CSV (data/raw)    │    │ corpus_regulacion/           │
        └──────────────┬───────────────┘    └──────────────┬───────────────┘
                       │ 365 días reales                   │ PyPDFLoader / HTML + splitter
        ┌──────────────▼───────────────┐    ┌──────────────▼───────────────┐
        │ Parte 1: MDP propio          │    │ Índice FAISS local           │
        │ Q-Learning · Línea Base ·    │    │ all-MiniLM-L6-v2             │
        │ Iteración de Valores         │    │ → data/processed/faiss_index │
        │ → data/processed/*.json, csv │    │                              │
        └──────────────┬───────────────┘    └──────────────┬───────────────┘
                       │ simular_despacho                  │ query_creg_regulation
                       └─────────────────┬─────────────────┘
                         ┌───────────────▼───────────────┐
                         │ Parte 2: Agente LLM (Groq)    │
                         │ loop manual, T_max = 5        │
                         └───────────────┬───────────────┘
                         ┌───────────────▼───────────────┐
                         │ Dashboard (Streamlit + Plotly)│
                         │ simulación real + chat        │
                         └───────────────────────────────┘
```

La separación es estricta: **el LLM no hace cálculos**. Las cifras de ahorro las produce
siempre el simulador determinista de la Parte 1, y las afirmaciones normativas salen de los
fragmentos que devuelve el RAG. El LLM solo decide qué herramienta usar y redacta la respuesta.

### Parte 1: motor de decisión (MDP, Q-Learning e Iteración de Valores)

Implementado desde cero en Python, sin Gymnasium ni Stable-Baselines (`src/part1_rl/`).

| Elemento | Definición |
|---|---|
| Estado | `(hora_del_dia 0-23, cuartil_de_precio 0-3, nivel_soc 0-4)` → 480 estados |
| Cuartil de precio | Umbrales p25/p50/p75 de una **ventana móvil causal** de las 168 h anteriores (nunca usa la hora actual ni el futuro). Las primeras 168 h de la serie usan los cuartiles de esa primera semana |
| Acciones | `0: Mantener`, `1: Cargar`, `2: Descargar` |
| Transición | Determinista en hora y SoC. Cargar al 100 % o descargar al 0 % se invalida y no cambia el estado |
| Recompensa | Cargar: `-precio·kWh`; Descargar: `+precio·kWh` (costo evitado); Mantener: `0`; acción inválida: `-1000` |
| Episodio | Un día real del historial (24 pasos de 1 hora); la batería arranca vacía |
| Q-Learning | `Q(s,a) ← Q(s,a) + α(s,a)[R' + γ·max Q(s',·) − Q(s,a)]`, ε-greedy con decaimiento lineal |
| *Reward shaping* (solo entrenamiento) | `R' = R + γ·Φ(s') − Φ(s)` con `Φ = SoC · kWh por escalón · p50_t` (mediana de la ventana móvil) y `Φ = 0` al terminar el día. Cargar por debajo de la mediana o descargar por encima da recompensa positiva inmediata |
| Tasa de aprendizaje | `α(s,a) = 1 / n(s,a)^ω`, decrece con las visitas de cada par (estado, acción) |
| Línea Base justa | Carga en las horas 01-04 y descarga en las 18-21 (usa los 4 escalones), sin mirar el precio |
| Política de referencia | **Iteración de Valores** sobre el modelo empírico del entrenamiento: precio medio por (hora, cuartil) y probabilidades de transición entre cuartiles de horas consecutivas (`src/part1_rl/value_iteration.py`) |
| Cota superior | Óptimo con información perfecta (programación dinámica conociendo los 24 precios del día). No es implementable; solo sirve como techo |

Protocolo experimental (`notebooks/02_mdp_rl.ipynb`):

- **Datos:** 365 días reales de `PrecBolsNaci` (2025-11-10 a 2026-11-09).
- **División cronológica 70 / 15 / 15:** entrenamiento 255 días (hasta 2026-07-22), validación
  55 días (2026-07-23 a 2026-09-15) y prueba 55 días (2026-09-16 a 2026-11-09).
- **Selección de hiperparámetros solo con validación:** rejilla ω ∈ {0.6, 0.7, 0.8, 1.0} ×
  decaimiento de ε en el 50 % u 80 % de los episodios, 5 semillas por combinación, 5,000
  episodios cada una (40 entrenamientos). Ganó ω = 0.7 con decaimiento en el 50 %.
- **Entrenamiento final (escenario base):** 5,000 episodios, γ = 0.99, ε de 1.0 a 0.05,
  semilla 42.
- **Escenarios por presupuesto:** el mismo diseño entrenado con 100 (corto), 5,000 (base) y
  50,000 (largo) episodios, para ver cuánto mejora con más entrenamiento.
- **Métricas finales:** solo sobre los días de prueba, que no se usaron para decidir nada.

#### Por qué cambió el diseño de Q-Learning

La primera versión usaba α = 0.3 fijo y la recompensa era solo el flujo de caja. Con ese diseño
Q-Learning ahorraba un 57.6 % menos que la Línea Base. Había dos causas:

1. **La recompensa llegaba tarde.** Cargar de madrugada cuesta de inmediato, y la recompensa de
   descargar llega 15 a 18 horas después. Q-Learning la propaga hacia atrás un paso por
   episodio, a través de cuartiles que cambian al azar, así que aprendía primero que "cargar
   cuesta". El *reward shaping* basado en potencial (Ng, Harada y Russell, 1999) le da esa
   información en el mismo paso, sin cambiar la métrica: el ahorro se sigue midiendo con el
   flujo de caja real.
2. **α fijo con recompensas ruidosas.** Con α = 0.3, cada Q(s,a) era casi un promedio de los
   últimos 3 o 4 días: no convergía y el `max` volvía ese ruido una sobreestimación. Con
   `α = 1/n^ω` se cumplen las condiciones de convergencia de Robbins-Monro.

Matiz: la garantía formal del *shaping* exige que Φ dependa solo del estado; aquí depende
también de la mediana móvil `p50_t`, que el estado resume con el cuartil. Por eso es una
aproximación.

### Parte 2: copiloto regulatorio y operativo (LLM + RAG + herramientas)

- **LLM:** Groq con el modelo `openai/gpt-oss-120b` (configurable con la variable
  `GROQ_MODEL`). Se eligió Groq por costo (tiene capa gratuita) y por velocidad de inferencia.
  Los modelos `llama3-70b-8192` y `llama-3.3-70b-versatile` fueron retirados por Groq, y
  `gpt-oss-120b` es el reemplazo que recomienda.
- **Loop agéntico manual** (`src/agent/llm_agent.py`): máximo 5 iteraciones por pregunta. Si
  una herramienta falla, el error vuelve al LLM como observación para que lo explique o corrija
  sus argumentos; la aplicación no se cae. Si se agota el límite, se fuerza una respuesta final
  sin herramientas que indica si quedó incompleta.
- **Herramientas:**
  - `simular_despacho(capacidad_kwh, fecha_inicio=None, fecha_fin=None)`: ejecuta las tres
    políticas sobre los días de prueba (opcionalmente un rango de fechas dentro de ellos) y
    devuelve el ahorro de cada una en COP, la diferencia frente a la Línea Base y la fracción
    de la referencia que logra Q-Learning.
  - `query_creg_regulation(topic)`: devuelve los 3 fragmentos más relevantes del corpus CREG,
    cada uno con su archivo y página.
- **RAG:** embeddings locales `sentence-transformers/all-MiniLM-L6-v2` (sin costo de API) e
  índice **FAISS** guardado en disco. Fragmentos de 1000 caracteres con 200 de traslape
  (330 fragmentos en total).

### Dashboard

Aplicación Streamlit (`src/dashboard/app.py`) conectada a `simular()`, con tema oscuro
(`.streamlit/config.toml`), tarjetas blancas y gráficas Plotly con plantilla `plotly_white`:

- **Barra lateral:**
  - *Parámetros de simulación:* capacidad de la batería (10 a 1000 kWh), **eficiencia del
    sistema** (70 % a 100 %, por defecto 90 %), **potencia del inversor** (10 % a 100 % de la
    capacidad, en kW), escenario de entrenamiento (corto, base o largo), periodo (prueba o
    validación) y rango de fechas.
  - *Notas del modelo* y estado del *Copiloto*.
- **Tarjetas:** ahorro en COP de Q-Learning, Línea Base justa e Iteración de Valores, con la
  diferencia porcentual frente a la Línea Base.
- **Pestaña "Análisis Operativo y Gráficas"**, con una sección por gráfica:
  - *Precio y operación:* arriba solo el precio de bolsa de XM; abajo el nivel de la batería
    (SoC) con triángulos verdes donde Q-Learning cargó y rojos donde descargó. Abre en la
    primera semana del rango, con botones de 1, 7 y 14 días.
  - *Ahorro acumulado:* las tres políticas al cierre de cada día.
  - *Escenarios de entrenamiento:* Q-Learning en los tres escenarios frente a la Línea Base y
    la Iteración de Valores, y la curva de aprendizaje (ahorro según los episodios).
  - *Resumen ejecutivo para gerencia:* un botón envía las cifras de la simulación al LLM
    (Groq), que redacta un párrafo sin calcular ni cambiar los números
    (`generar_resumen_ejecutivo` en `src/agent/llm_agent.py`). Es una llamada aparte que no
    toca la memoria del chat.
- **Pestaña "Copiloto Regulatorio e Interacción":** chat con preguntas sugeridas que se envían
  con un clic, y un desplegable en cada respuesta con las herramientas usadas y lo que
  devolvieron. El copiloto simula siempre con el escenario base, los días de prueba y una
  batería ideal (eficiencia 100 %, sin límite de inversor).

#### Eficiencia e inversor: cómo se modelan

- **Eficiencia de ida y vuelta η:** de cada kWh comprado a la red, la batería solo entrega η
  kWh al descargar. Por eso η multiplica el **costo evitado de cada descarga**, y las compras
  se pagan completas. No se multiplica el ahorro neto por η: eso también abarataría la energía
  comprada, y una pérdida se vería más pequeña. Con η = 90 % un ciclo solo es rentable si el
  precio de descarga supera en un 11 % al de carga.
- **Políticas con la eficiencia elegida:** cada escenario de Q-Learning se entrenó con cada
  eficiencia de la grilla (70 %, 75 %, …, 100 %); la Iteración de Valores se replanifica con
  su modelo empírico; la Línea Base mantiene su regla.
- **Potencia del inversor:** limita la energía de cada hora. El modelo mueve un escalón
  (capacidad / 4) por hora, así que un inversor menor que capacidad / 4 kW reduce la energía
  de cada carga y descarga, y uno mayor no cambia el resultado.

## 3. Datos

| Fuente | Qué aporta | Acceso |
|--------|-----------|--------|
| **XM** (operador del mercado) | Precio de bolsa nacional horario (`PrecBolsNaci`, COP/kWh), 2025-11-10 a 2026-11-09 | Librería [`pydataxm`](https://github.com/EquipoAnaliticaXM/API_XM). Cada descarga actualiza el respaldo versionado `data/raw/xm_precio_bolsa.csv`; si la API falla se usa ese CSV |
| **Resolución CREG 098 de 2019** | Incorporación de sistemas de almacenamiento con baterías (SAEB) | Versión HTML del [gestor normativo](https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0098_2019.htm), incluida en el repositorio |
| **Resolución CREG 174 de 2021** | Autogeneración a pequeña escala y generación distribuida | [PDF oficial](https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/c99b2f316a59fffb052587950077d9d1/$FILE/Creg174-2021.pdf) (67 páginas), incluido en el repositorio |

- El PDF oficial de la CREG 098 es un escaneo sin capa de texto, por eso se indexa su versión
  HTML.
- **Vigencia:** el gestor normativo marca la CREG 098 de 2019 como *derogada por el artículo
  108 de la Resolución 101 113 de 2026*. Se mantiene en el corpus como antecedente regulatorio
  de los SAEB, y el copiloto puede citar esa nota de vigencia.
- Las fuentes del corpus están documentadas en
  [`corpus_regulacion/README.md`](corpus_regulacion/README.md).
- Solo se usan datos agregados del mercado; no hay datos personales.

## 4. Estructura del repositorio

```
.
├── corpus_regulacion/
│   ├── README.md                       # fuentes, URLs y vigencia del corpus
│   ├── resolucion_creg_0098_2019.htm   # CREG 098/2019 (versión con texto)
│   └── Creg174-2021.pdf                # CREG 174/2021 (PDF oficial)
├── data/
│   ├── raw/
│   │   └── xm_precio_bolsa.csv         # respaldo de XM (plan B), 365 días
│   └── processed/
│       ├── q_table.json                # Q-Table del escenario base (5,000 ep., batería ideal)
│       ├── escenarios/                 # q_{corto,base,largo}_eta{70..100}.json
│       ├── curva_aprendizaje.csv       # ahorro según episodios y eficiencia
│       ├── politica_referencia.json    # Iteración de Valores y su modelo empírico
│       ├── precios_evaluacion.csv      # días de prueba + umbrales causales por hora
│       ├── precios_validacion.csv      # días de validación + umbrales causales por hora
│       └── faiss_index/                # índice vectorial (se genera; ignorado por git)
├── docs/
│   ├── Informe Final.pdf               # informe del proyecto
│   ├── Analisis.pdf                    # análisis de resultados
│   ├── evaluacion_agente.md            # reporte generado por la batería de QA
│   └── Uso_IA.md                       # declaración de uso de IA generativa
├── notebooks/
│   ├── 01_eda.ipynb                    # análisis exploratorio del precio de bolsa
│   └── 02_mdp_rl.ipynb                 # entrenamiento y evaluación de la Parte 1
├── src/
│   ├── data/download_xm.py             # descarga de XM con respaldo CSV
│   ├── part1_rl/
│   │   ├── environment.py              # BatteryEnv (MDP), ventana móvil y Línea Base
│   │   ├── agent.py                    # QLearningAgent (tabular, ε-greedy)
│   │   ├── value_iteration.py          # política de referencia (Iteración de Valores)
│   │   └── simulacion.py               # simular(): 3 políticas + traza horaria
│   ├── rag/vectorstore.py              # ingesta del corpus e índice FAISS
│   ├── agent/llm_agent.py              # RegulatoryAgent, loop y herramientas
│   └── dashboard/app.py                # aplicación Streamlit
├── tests/
│   └── test_agent.py                   # batería de QA automatizada del agente
├── requirements.txt
└── README.md
```

Los artefactos de la Parte 1 (las tres Q-Tables, `politica_referencia.json` y los dos CSV de
precios) están versionados para que el dashboard abra sin reentrenar. El
índice FAISS no se versiona porque se serializa con pickle; se construye en el paso 4.

## 5. Configuración y ejecución

Probado en Windows con Python 3.13 y en Linux con Python 3.14. Se requiere Python 3.10 o
superior y conexión a internet (para Groq y para descargar el modelo de embeddings la primera
vez). Todos los comandos se ejecutan desde la raíz del repositorio.

### Paso 1. Prerrequisitos del sistema

**Windows:** instala Python desde [python.org](https://www.python.org/downloads/) y marca la
opción *Add python.exe to PATH* durante la instalación. Verifica en PowerShell:

```powershell
python --version
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
python -m venv .venv
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

### Paso 4. Construir el índice FAISS (obligatorio, una sola vez)

La primera vez descarga el modelo de embeddings (unos 90 MB), así que tarda unos minutos.

```bash
python -m src.rag.vectorstore
# Consulta de prueba:
python -m src.rag.vectorstore --buscar "almacenamiento con baterías"
```

Debe terminar con `[OK] Índice guardado en .../data/processed/faiss_index` e indexar las dos
resoluciones (330 fragmentos).

### Paso 5. (Opcional) Reentrenar la Parte 1

No es necesario para usar el dashboard: los artefactos ya están en `data/processed/`. Para
reproducir los resultados desde cero (unos 10 minutos):

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 notebooks/02_mdp_rl.ipynb
```

- Por defecto el notebook lee el respaldo versionado `data/raw/xm_precio_bolsa.csv`, así las
  cifras salen idénticas a las de este README. XM revisa los precios de días recientes (por
  ejemplo, una nueva descarga cambió las 24 horas del 2026-11-08), así que con
  `ACTUALIZAR_DESDE_API = True` el notebook descarga de nuevo y las cifras pueden variar un poco.
- También se puede abrir con `jupyter notebook notebooks/02_mdp_rl.ipynb` y ejecutar todas las
  celdas.
- El notebook regenera todos los artefactos de `data/processed/` y comprueba con `assert` que
  `simular()` da las mismas cifras que el notebook (batería ideal y escenarios con eficiencia
  90 %).

### Paso 6. Configurar la API key de Groq

Ver la sección [Seguridad y variables de entorno](#seguridad-y-variables-de-entorno).

### Paso 7. Ejecutar

| Qué | Comando |
|---|---|
| Dashboard | `python -m streamlit run src/dashboard/app.py` (abre `http://localhost:8501`) |
| Agente por consola | `python -m src.agent.llm_agent "¿Qué dice la CREG sobre los SAEB?"` |
| Batería de QA | `python tests/test_agent.py` (regenera `docs/evaluacion_agente.md`, unos 3 minutos) |
| Descarga de XM | `python -m src.data.download_xm --inicio 2025-11-10 --fin 2026-11-09` |

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
abre igual (la simulación y las gráficas funcionan), pero deshabilita el chat y muestra un
aviso. Variables opcionales:

| Variable | Uso | Valor por defecto |
|---|---|---|
| `GROQ_API_KEY` | Clave de Groq (obligatoria para el agente) | — |
| `GROQ_MODEL` | Modelo de Groq | `openai/gpt-oss-120b` |
| `EMBEDDING_MODEL` | Modelo de embeddings del RAG (hay que reconstruir el índice si cambia) | `sentence-transformers/all-MiniLM-L6-v2` |

Nunca pegues la clave en el código, en notebooks ni en capturas de pantalla. Si se filtra,
revócala en **API Keys** y genera una nueva.

### Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` al descargar de XM | Python no confía en la CA de `servapibi.xm.com.co`. `truststore` (en `requirements.txt`) hace que use los certificados del sistema. Si persiste, el notebook usa el CSV de respaldo |
| `ImportError: cannot import name 'DEFAULT_EXCLUDED_CONTENT_TYPES'` al abrir Streamlit | Versión vieja de `starlette`. Ejecuta `python -m pip install --upgrade streamlit starlette` |
| `No existe el índice FAISS` en el chat | Falta el paso 4: `python -m src.rag.vectorstore` |
| `Jupyter command jupyter-nbconvert not found` | Usa `python -m nbconvert ...` como en el paso 5 |
| `streamlit : El término 'streamlit' no se reconoce` | La carpeta de scripts de Python no está en el PATH. Usa `python -m streamlit run src/dashboard/app.py` |
| El chat aparece deshabilitado | Falta `GROQ_API_KEY` en `.env` (paso 6) |
| Error de cuota o límite de tasa de Groq | Espera un minuto y repite; la QA ya hace una pausa entre preguntas |

## 6. Evaluación

### Parte 1: Línea Base vs. Q-Learning vs. Iteración de Valores

Resultado sobre los **55 días de prueba** (2026-09-16 a 2026-11-09) con precios reales de XM y
una batería ideal de 4 kWh (1 kWh por escalón, eficiencia 100 %). Ninguno de estos días se usó para entrenar ni para
elegir hiperparámetros.

| Política | Ahorro acumulado | Cargas / Descargas | Inválidas | % de la cota superior | vs. Línea Base |
|---|---|---|---|---|---|
| Línea Base justa (01-04 h / 18-21 h) | 89,338 COP | 220 / 220 | 0 | 81.6 % | — |
| Q-Learning, diseño anterior (α = 0.3 fijo, sin *shaping*) | 37,854 COP | 152 / 152 | 0 | 34.6 % | −57.6 % |
| **Q-Learning, diseño nuevo** (*shaping* + α = 1/n^0.7, 5,000 ep.) | **82,222 COP** | 392 / 392 | 0 | 75.1 % | −8.0 % |
| Iteración de Valores (referencia) | **95,134 COP** | 415 / 415 | 0 | 86.8 % | **+6.5 %** |
| Cota superior con información perfecta | 109,546 COP | — | — | 100 % | — |

Con el diseño nuevo, Q-Learning **duplica su ahorro en prueba** (de 37,854 a 82,222 COP) y pasa
del 39.8 % al 86.4 % de la referencia. Con 5 semillas obtiene 84,994 ± 4,602 COP (antes
42,944 ± 13,169): además de ahorrar más, varía tres veces menos.

**En validación** (55 días, 2026-07-23 a 2026-09-15; promedio de 5 semillas):

| Política | Ahorro | vs. Línea Base |
|---|---|---|
| Línea Base justa | 38,598 COP | — |
| Q-Learning, diseño anterior | 23,550 ± 8,511 COP | −39.0 % |
| **Q-Learning, diseño nuevo** | **49,322 ± 1,345 COP** | **+27.8 %** |
| Iteración de Valores | 50,409 COP | +30.6 % |

**Escenarios por presupuesto de entrenamiento** (diseño nuevo, semilla 42, batería ideal):

| Escenario | Episodios | Validación | vs. Línea Base | Prueba | vs. Línea Base |
|---|---|---|---|---|---|
| Corto | 50 | 9,685 COP | −74.9 % | 39,528 COP | −55.8 % |
| **Base** (el que usan la app y el agente) | 5,000 | 46,525 COP | **+20.5 %** | 82,222 COP | −8.0 % |
| Largo | 50,000 | 53,048 COP | **+37.4 %** | 86,467 COP | −3.2 % |

**Los mismos escenarios con eficiencia de ida y vuelta de 90 %** (cada uno entrenado con esa
eficiencia; Línea Base: 16,450 en validación y 70,077 en prueba; Iteración de Valores
replanificada: 26,117 y 73,376):

| Escenario | Validación | vs. Línea Base | Prueba | vs. Línea Base |
|---|---|---|---|---|
| Corto | −19,923 COP | −221.1 % | 19,413 COP | −72.3 % |
| Base | 19,525 COP | **+18.7 %** | 62,696 COP | −10.5 % |
| Largo | 31,944 COP | **+94.2 %** | 62,588 COP | −10.7 % |

Con pérdidas, un ciclo con poca diferencia de precio deja de ser rentable, así que el
entrenamiento importa más: con 50 episodios el agente pierde dinero en validación. En prueba,
el escenario largo no mejora al base.

La curva completa (50, 300, 1,000, 2,000, 5,000, 20,000 y 50,000 episodios, para cada
eficiencia de 70 % a 100 %) está en la sección 6 del notebook y en
`data/processed/curva_aprendizaje.csv`. No es monótona: con una sola semilla hay altibajos
entre presupuestos.

Las cifras escalan de forma lineal con la capacidad, porque las políticas no dependen de ella:
con 100 kWh, en prueba y con batería ideal, son 2,055,557 (Q-Learning) / 2,233,455 (Línea
Base) / 2,378,354 (Iteración de Valores) COP. El dashboard abre con eficiencia 90 %, así que
muestra 1,567,388 / 1,751,928 / 1,834,406 COP.

Interpretación:

- **El MDP está bien planteado.** La Iteración de Valores, que resuelve el mismo MDP con un
  modelo estimado solo con entrenamiento, supera a la Línea Base en un 6.5 % y alcanza el
  86.8 % del óptimo con información perfecta.
- **La ventana móvil causal era necesaria.** Con cuartiles fijos de todo el entrenamiento, los
  precios de septiembre a noviembre (más altos) caían casi siempre en los cuartiles superiores
  y la Iteración de Valores quedaba por debajo de la Línea Base (85,919 COP, −3.8 %). Con la
  ventana móvil los umbrales al inicio de la prueba pasan de 154/266/432 a 425/492/591 COP/kWh.
- **El problema de Q-Learning era de diseño, no de iteraciones.** Con el diseño anterior, 10
  veces más episodios (50,000 con α = 0.1) solo subían su promedio en validación de 22,442 a
  32,853 COP. Con *shaping* y α decreciente, con los mismos 5,000 episodios, llega a 49,322.
- **Con más entrenamiento Q-Learning se acerca al techo del MDP.** En validación el escenario
  largo (53,048) alcanza e incluso supera a la Iteración de Valores (50,409), que usa un modelo
  aproximado. En prueba la mejora es menor (de −8.0 % a −3.2 % frente a la Línea Base).
- **En los días de prueba la Línea Base es muy difícil de superar.** Acierta el patrón diario
  del precio de bolsa (barato de madrugada, caro en la noche) y captura el 81.6 % del óptimo.
  Solo la Iteración de Valores le gana (+6.5 %). Q-Learning le gana en validación pero no en
  prueba, y así se reporta.
- **Más datos antiguos no ayudan.** En una prueba exploratoria, entrenar con 2022 a 2026
  (1,664 días, incluye el periodo de El Niño) empeoró tanto a la Iteración de Valores
  (43,995 en validación) como a Q-Learning. El comportamiento reciente del mercado es más
  informativo que años con otro régimen de precios.

### Parte 2: batería de QA automatizada del agente

`tests/test_agent.py` ejecuta 12 preguntas fijas contra el agente real (Groq + RAG + simulador)
y genera el reporte [`docs/evaluacion_agente.md`](docs/evaluacion_agente.md). Los criterios son
deterministas (no se usa un LLM como juez):

- Las herramientas invocadas deben ser exactamente las esperadas.
- Las respuestas de simulación deben usar la capacidad pedida y contener las cifras exactas de
  Q-Learning y de la Línea Base que calcula el simulador.
- Cada cita regulatoria (archivo y página) debe corresponder a un fragmento que devolvió el RAG,
  y el RAG debe recuperar la resolución esperada.
- Las preguntas trampa deben rechazarse sin inventar datos ni cambiar los valores del usuario.
- Ningún caso puede agotar el límite de iteraciones.

Resultado de la última ejecución (2026-11-13 02:42, con el diseño nuevo de Q-Learning y
precios de XM; el reporte muestra el commit `d7b175a` porque se generó antes de commitear estos
cambios):

| Categoría | Casos | Aprobados |
|---|---|---|
| Simulación | 4 | 3 |
| Regulación | 4 | 4 |
| Combinada (simulación + regulación) | 2 | 1 |
| Trampa (capacidad negativa, fuera de tema) | 2 | 2 |
| **Total** | **12** | **10 (83 %)** |

Fallos documentados:

- **S3 ("¿Cuánto ahorraría el agente de RL con 1000 kWh?"):** la herramienta y la cifra de
  Q-Learning fueron correctas, pero la respuesta omitió el ahorro de la Línea Base. La pregunta
  solo pide el del agente, así que el criterio es estricto.
- **C2 (300 kWh + excedentes):** las cifras de las tres políticas fueron exactas y el RAG
  recuperó la CREG 174, pero el LLM citó con marcadores `【1】` en lugar del formato
  `(Fuente: archivo, página N)`, así que la cita no se pudo verificar.

Ninguna cifra reportada fue incorrecta. Los criterios comprueban que cada cita exista entre los
fragmentos recuperados, no que la afirmación sea correcta. El resultado corresponde a una sola
ejecución con temperatura 0.2 y puede variar entre corridas: en ejecuciones anteriores fallaron
R1 y C1, o S3 y C2 por omitir la Línea Base.

## 7. Limitaciones

- El sistema es una herramienta analítica y educativa; **no** constituye asesoría
  regulatoria, legal ni financiera.
- **Q-Learning supera a la Línea Base en validación (+20.5 %) pero no en prueba (−8.0 %)**
  (ver sección 6). La política recomendable con estos datos es la de Iteración de Valores.
- **Reward shaping aproximado:** el potencial depende de la mediana móvil del precio, que no es
  parte exacta del estado, así que la garantía teórica de que no cambia la política óptima es
  aproximada.
- **Precio de bolsa, no tarifa:** el ahorro se calcula con el precio de bolsa nacional. Un
  usuario regulado paga una tarifa que incluye transmisión, distribución, comercialización y
  otros cargos, así que el ahorro real sería distinto.
- **Volatilidad del mercado:** las políticas aprenden de un periodo de 255 días. Choques como
  El Niño, el nivel de los embalses, fallas de transmisión o intervenciones de la CREG pueden
  hacer que dejen de ser buenas. La ventana móvil reduce, pero no elimina, este problema.
- **Modelo de batería simplificado:** 5 niveles de carga, un solo escalón de carga o descarga
  por hora y batería vacía al inicio de cada día. La eficiencia es una sola cifra de ida y
  vuelta (no depende de la potencia ni de la temperatura) y no hay degradación.
- **Inversor:** con un inversor menor que capacidad / 4 kW, cada escalón mueve menos energía y
  se aprovecha solo parte de la capacidad; un controlador real podría cargar durante más horas.
- **Resultados oficiales con batería ideal:** las tablas de la sección 6 y la QA usan
  eficiencia 100 %. Las cifras con otras eficiencias son de los escenarios del dashboard.
- **Evaluación de un solo periodo:** la prueba cubre 55 días de septiembre a noviembre de 2026.
- **RAG limitado:** el corpus tiene dos resoluciones (una de ellas derogada), el modelo de
  embeddings está entrenado principalmente en inglés y solo se recuperan 3 fragmentos por
  consulta.
- **QA de una sola corrida:** el 83 % puede variar entre ejecuciones por la naturaleza del LLM.

## 8. Declaración de uso de IA

Ver [`docs/Uso_IA.md`](docs/Uso_IA.md).


## 9. Video demostracion de la aplicacion

https://drive.google.com/drive/folders/11pX-ySpsovOYtaO4yMj3FlGHHu-Wtybf?usp=sharing
