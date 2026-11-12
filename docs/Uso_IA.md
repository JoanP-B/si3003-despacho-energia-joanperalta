# Declaración de uso de IA generativa

Proyecto Final — SI3003 Inteligencia Artificial — Universidad EAFIT — 2026-1

Autor: Joan Stiven Peralta Bedoya

En cumplimiento del pacto pedagógico del curso, este documento declara cómo se usaron
herramientas de IA generativa en el desarrollo del proyecto *Despacho de energía con RL y
asistente sobre regulación*. La IA actuó como copiloto técnico y tutor. La definición del
problema, las reglas de negocio, las decisiones de diseño, la aceptación de cada cambio y la
interpretación de los resultados estuvieron a cargo del autor.

## Herramientas utilizadas

| Herramienta | Tipo | Uso en el proyecto |
|---|---|---|
| Kiro (asistente de programación basado en LLM, ejecutado en la terminal) | Asistente de código y documentación | Tutoría de diseño, generación y depuración de código, ejecución de pruebas en el entorno local y borradores de documentación |
| Groq (`openai/gpt-oss-120b`) | Componente del sistema | Es el LLM del copiloto regulatorio (Parte 2). No se usó para escribir el proyecto |

La segunda fila se incluye para distinguir el LLM que forma parte del producto del asistente
que se usó para construirlo.

## 1. Ayuda con decisiones arquitectónicas y modelado

La IA se usó como tutor para contrastar alternativas y justificar decisiones que el autor ya
había planteado en el informe final.

- **Separación entre cálculo y orquestación.** Se discutió cómo garantizar que el LLM no
  produjera cifras. La decisión fue que todo número de ahorro lo calcule el simulador
  determinista de la Parte 1 (MDP y Q-Learning), que todo contenido normativo salga del RAG, y
  que el LLM solo decida qué herramienta llamar y redacte la respuesta. Esta separación es la
  base de la arquitectura y de los criterios de evaluación.
- **Modelado del MDP.** El autor definió el problema como desplazamiento de demanda (*load
  shifting*) y no como especulación financiera, junto con el estado, las acciones, la
  recompensa y la Línea Base. La IA ayudó a validar la discretización
  `(hora, cuartil de precio, nivel de carga)`: 24 × 4 × 5 = 480 estados, un tamaño manejable
  para una tabla Q frente a la explosión combinatoria de un estado de carga continuo. También
  señaló dos detalles de modelado que el autor revisó y aprobó: arrancar cada día con la
  batería vacía, para no contar energía "gratis" como ahorro, y calcular los cuartiles de
  precio solo con los días de entrenamiento, para no filtrar información de la evaluación.
- **Elección del proveedor de LLM.** El autor descartó OpenAI por costo. La IA ayudó a
  comparar opciones y a confirmar, en la documentación oficial de Groq, que los modelos
  inicialmente previstos (`llama3-70b-8192` y `llama-3.3-70b-versatile`) estaban retirados.
  Se adoptó `openai/gpt-oss-120b`, que tiene capa gratuita, baja latencia y soporte de
  llamadas a herramientas.
- **Diseño del loop agéntico.** Se compararon un loop manual y LangGraph. Se eligió el loop
  manual porque hace explícitos el límite de iteraciones (T_max = 5) y el tratamiento de
  errores como observación, que son requisitos de la rúbrica.

## 2. Asistencia para código y depuración

El asistente generó buena parte del código a partir de especificaciones escritas por el autor
(firmas, reglas de negocio, restricciones y casos de error). El autor revisó cada componente,
lo ejecutó en su entorno local y decidió qué cambios se integraban.

- **Entorno MDP y Q-Learning en Python puro** (`src/part1_rl/`). Implementación del entorno
  `BatteryEnv` con las restricciones físicas de carga y descarga, del agente `QLearningAgent`
  con actualización tabular y ε-greedy con decaimiento lineal, y del notebook de
  entrenamiento. Durante la depuración se detectaron y corrigieron: que el entorno promediaba
  todos los días en uno solo; que con α = 0.1 y 500 episodios el agente no aprendía, lo que
  llevó a un barrido de hiperparámetros; que la Q-Table no guardaba los umbrales de precio
  necesarios para reutilizarla; y que la evaluación no era determinista por desempates
  aleatorios.
- **Loop agéntico manual** (`src/agent/llm_agent.py`). Bucle con límite de iteraciones,
  memoria acotada y manejo del error como observación: si una herramienta falla, el mensaje
  vuelve al LLM en lugar de detener la aplicación. Las pruebas con el LLM real mostraron que el
  modelo completaba la normativa con conocimiento propio; el prompt de sistema se reforzó para
  que solo afirme lo que devuelven las herramientas.
- **Herramienta RAG con FAISS** (`src/rag/vectorstore.py`). Ingesta del corpus, troceo,
  embeddings locales `all-MiniLM-L6-v2` e índice FAISS. Al construir el índice se encontró que
  el PDF oficial de la Resolución CREG 098 de 2019 es un escaneo sin capa de texto; se
  incorporó la lectura de su versión HTML oficial del gestor normativo de la CREG. También se
  comparó este modelo de embeddings con uno multilingüe (16/18 frente a 15/18 fragmentos de la
  fuente correcta) antes de mantener el elegido.
- **Script determinista de pruebas** (`tests/test_agent.py`). El autor definió la batería de
  12 preguntas por categoría (simulación, regulación, combinadas y trampas) y la exigencia de
  verificar herramientas, citas y cifras. El asistente implementó los verificadores sin usar un
  LLM como juez, los probó sin conexión antes de gastar llamadas a la API (lo que reveló y
  corrigió un falso positivo en la comparación de cifras) y ejecutó la batería contra el agente
  real en el entorno local del autor.

## 3. Revisión y formateo de documentación

- **Informe final.** Se usó IA para pulir la redacción técnica del informe
  (`docs/Informe Final - SI3003.pdf`). El contenido, los argumentos y las conclusiones son del
  autor.
- **Casos de prueba.** Se usó IA para estructurar la justificación de cada caso de prueba y
  de cada criterio de aprobación, y para redactar el análisis de los fallos (casos R1 y C1)
  sin ocultarlos.
- **Tablas y reportes.** La IA dio formato a las tablas de métricas del `README.md` y al
  reporte `docs/evaluacion_agente.md`, que se genera automáticamente a partir de la ejecución
  real de las pruebas.

## Verificación y responsabilidad del autor

- **Las cifras son reales.** Todas las métricas reportadas (ahorro del Q-Learning frente a la
  Línea Base y el 10/12 de la batería de QA) provienen de ejecuciones reales en el entorno
  local, no de estimaciones de la IA. Los resultados desfavorables también se reportan.
- **Las fuentes se verificaron.** Las resoluciones del corpus se descargaron del gestor
  normativo oficial de la CREG y sus URLs están documentadas en
  `corpus_regulacion/README.md`.
- **Se declaran las limitaciones.** Los precios de la Parte 1 son simulados, y el README lo
  advierte junto con las demás limitaciones del sistema.
- **El código propio cumple el requisito del curso.** El MDP y el Q-Learning se implementaron
  desde cero, sin librerías que ya resuelvan el problema.
- **El autor responde por el resultado.** Todo el código y el texto generados con ayuda de IA
  fueron revisados y aceptados por el autor antes de integrarse al repositorio.
