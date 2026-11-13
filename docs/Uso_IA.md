# Declaración de uso de IA generativa

Proyecto Final — SI3003 Inteligencia Artificial — Universidad EAFIT — 2026-1

Autor: Joan Stiven Peralta Bedoya

En cumplimiento del pacto pedagógico del curso, este documento declara cómo se usaron
herramientas de IA generativa en el desarrollo del proyecto *Despacho de energía con RL y
asistente sobre regulación*. La IA actuó como copiloto técnico, revisor y tutor. La definición
del problema, las reglas de negocio, las decisiones de diseño, la aceptación de cada cambio y
la interpretación de los resultados estuvieron a cargo del autor.

## Herramientas utilizadas

| Herramienta | Tipo | Uso en el proyecto |
|---|---|---|
| Kiro (asistente de programación basado en LLM, en el IDE) | Asistente de código, revisión y documentación | Tutoría de diseño, revisión del proyecto contra el enunciado y la rúbrica, generación y depuración de código, ejecución de notebooks y pruebas en el entorno local, borradores de documentación |
| Groq (`openai/gpt-oss-120b`) | Componente del sistema | Es el LLM del copiloto regulatorio (Parte 2). No se usó para escribir el proyecto |

La segunda fila se incluye para distinguir el LLM que forma parte del producto del asistente
que se usó para construirlo.

## 1. Ayuda con decisiones arquitectónicas y modelado

La IA se usó como tutor para contrastar alternativas y justificar decisiones que el autor ya
había planteado en el informe final.

- **Separación entre cálculo y orquestación.** Se discutió cómo garantizar que el LLM no
  produjera cifras. La decisión fue que todo número de ahorro lo calcule el simulador
  determinista de la Parte 1, que todo contenido normativo salga del RAG, y que el LLM solo
  decida qué herramienta llamar y redacte la respuesta.
- **Modelado del MDP.** El autor definió el problema como desplazamiento de demanda (*load
  shifting*) y no como especulación financiera, junto con el estado, las acciones y la
  recompensa. La IA ayudó a validar la discretización `(hora, cuartil de precio, nivel de
  carga)`: 24 × 4 × 5 = 480 estados. También señaló detalles que el autor revisó y aprobó:
  arrancar cada día con la batería vacía, para no contar energía "gratis" como ahorro, y no
  usar información de los días de evaluación para construir el estado.
- **Elección del proveedor de LLM.** El autor descartó OpenAI por costo. La IA ayudó a
  comparar opciones y a confirmar, en la documentación oficial de Groq, que los modelos
  inicialmente previstos (`llama3-70b-8192` y `llama-3.3-70b-versatile`) estaban retirados.
  Se adoptó `openai/gpt-oss-120b`, que tiene capa gratuita, baja latencia y soporte de
  llamadas a herramientas.
- **Diseño del loop agéntico.** Se compararon un loop manual y LangGraph. Se eligió el loop
  manual porque hace explícitos el límite de iteraciones (T_max = 5) y el tratamiento de
  errores como observación, que son requisitos de la rúbrica.

## 2. Revisión crítica antes de la entrega

En la fase final el autor le pidió a la IA revisar el código y los documentos contra el
enunciado (`EAFIT_Proyecto_Final_IA.pdf`). La revisión encontró problemas que el autor decidió
corregir:

- **Línea base injusta.** La línea base original cargaba un solo escalón a las 02:00 y
  descargaba uno a las 19:00, es decir, usaba un cuarto de la batería. Eso inflaba la mejora
  reportada de Q-Learning (+86.9 %). Un cálculo de control mostró que una regla fija que usa
  los 4 escalones ahorraba el doble que el agente.
- **Datos simulados.** Los resultados se habían obtenido con precios sintéticos, mientras que
  el enunciado exige datos reales.
- **Falta de una política de referencia.** El enunciado pide comparar contra una política
  ingenua y una de referencia; solo existía la ingenua.
- **Dashboard con datos de ejemplo** e inconsistencias entre el informe y el código (por
  ejemplo, el informe mencionaba LangGraph y herramientas con otros nombres y parámetros).

## 3. Asistencia para código y depuración

El asistente generó buena parte del código a partir de especificaciones escritas por el autor
(firmas, reglas de negocio, restricciones y casos de error). El autor revisó cada componente,
lo ejecutó en su entorno local y decidió qué cambios se integraban.

- **Entorno MDP y Q-Learning en Python puro** (`src/part1_rl/`). Implementación de
  `BatteryEnv`, de `QLearningAgent` (actualización tabular y ε-greedy con decaimiento lineal) y
  del notebook de entrenamiento. Durante la depuración se corrigieron: que el entorno promediaba
  todos los días en uno solo, que la Q-Table no guardaba lo necesario para reutilizarla y que la
  evaluación no era determinista por desempates aleatorios.
- **Datos reales de XM** (`src/data/download_xm.py`). Descarga de 365 días de `PrecBolsNaci` con
  `pydataxm` y respaldo automático en `data/raw/xm_precio_bolsa.csv`. Se diagnosticó un error de
  certificado SSL con la API de XM en Windows y se resolvió con `truststore`.
- **Línea Base justa y política de referencia.** Por especificación del autor, la línea base
  pasó a cargar en las horas 01-04 y descargar en las 18-21. Se implementó la Iteración de
  Valores (`src/part1_rl/value_iteration.py`) sobre un modelo empírico estimado solo con los
  días de entrenamiento.
- **Protocolo sin fuga de datos.** División cronológica 70/15/15, búsqueda de α y del
  decaimiento de ε solo con validación, y métricas finales solo con prueba.
- **Ventana móvil causal para los cuartiles.** Con datos reales, la IA diagnosticó que los
  cuartiles fijos del entrenamiento dejaban de distinguir horas baratas y caras cuando subía el
  nivel de precios. El autor decidió usar una ventana causal de 168 horas. Con ese cambio la
  Iteración de Valores pasó de −3.8 % a +6.5 % frente a la Línea Base.
- **Rediseño del aprendizaje de Q-Learning.** El autor pidió explicar por qué Q-Learning perdía
  contra la Línea Base. La IA planteó hipótesis y las comprobó con experimentos cortos elegidos
  solo con validación (máscara de acciones, Doble Q-Learning, tasa decreciente y *reward
  shaping*). Los dos cambios que funcionaron, *reward shaping* basado en potencial con la
  mediana móvil y `α = 1/n(s,a)^ω`, se integraron por decisión del autor. Con los mismos 5,000
  episodios, Q-Learning pasó de 37,854 a 82,222 COP en prueba. El autor también decidió agregar
  escenarios por presupuesto de entrenamiento (50, 5,000 y 50,000 episodios) al dashboard.
- **Loop agéntico manual** (`src/agent/llm_agent.py`). Bucle con límite de iteraciones, memoria
  acotada y manejo del error como observación. Las pruebas con el LLM real mostraron que el
  modelo completaba la normativa con conocimiento propio; el prompt de sistema se reforzó para
  que solo afirme lo que devuelven las herramientas y para que reporte las cifras aunque
  Q-Learning pierda.
- **Herramienta RAG con FAISS** (`src/rag/vectorstore.py`). Ingesta del corpus, troceo,
  embeddings locales e índice FAISS. Se encontró que el PDF oficial de la CREG 098 es un escaneo
  sin texto y se incorporó su versión HTML. Se comparó el modelo de embeddings con uno
  multilingüe (16/18 frente a 15/18 fragmentos de la fuente correcta) antes de mantener el
  elegido.
- **Dashboard** (`src/dashboard/app.py`). Conexión de los controles a `simular()`, tarjetas de
  ahorro de las tres políticas, gráficas de Plotly y un rediseño visual pedido por el autor a
  partir de una maqueta propia: tema oscuro con tarjetas blancas, pestañas de análisis y
  copiloto, una sección por gráfica, barra lateral con secciones plegables y preguntas
  sugeridas en el chat. Se verificó sin navegador con el `AppTest` de Streamlit.
- **Parámetros de negocio.** El autor pidió agregar la eficiencia del sistema y la potencia del
  inversor, y propuso multiplicar el ahorro por la eficiencia. La IA señaló que eso también
  abarataría la energía comprada (y achicaría una pérdida), y se aplicó la eficiencia de ida y
  vuelta al costo evitado de cada descarga. Como las políticas entrenadas con batería ideal
  hacen ciclos que dejan de ser rentables con pérdidas, cada escenario de Q-Learning se entrenó
  con cada eficiencia de la grilla y la Iteración de Valores se replanifica con su modelo.
- **Resumen ejecutivo.** Por pedido del autor, un botón envía las cifras ya calculadas de la
  simulación al LLM, que redacta un párrafo para gerencia. El prompt prohíbe calcular cifras
  nuevas y obliga a decir si Q-Learning ahorra menos que la Línea Base.
- **Reproducibilidad de los datos.** Al reejecutar el notebook se detectó que XM había revisado
  los precios del 2026-11-08, lo que cambiaba las cifras. Desde entonces el notebook usa por
  defecto el respaldo versionado y solo descarga de la API si se pide.
- **Script determinista de pruebas** (`tests/test_agent.py`). El autor definió la batería de 12
  preguntas y la exigencia de verificar herramientas, citas y cifras. El asistente implementó
  los verificadores sin usar un LLM como juez y los probó sin conexión antes de gastar llamadas
  a la API.

## 4. Revisión y formateo de documentación

- **Informe final.** Se usó IA para pulir la redacción técnica del informe
  (`docs/Informe Final.pdf`). El contenido, los argumentos y las conclusiones son del autor.
- **README y análisis de fallos.** La IA redactó borradores de las secciones de evaluación y
  limitaciones del `README.md`, incluidos los resultados desfavorables, y de la explicación de
  los fallos de la QA.
- **Reporte de QA.** `docs/evaluacion_agente.md` se genera automáticamente a partir de la
  ejecución real de las pruebas; la IA no lo editó a mano.

## Verificación y responsabilidad del autor

- **Las cifras son reales.** Todas las métricas reportadas provienen de ejecuciones reales en
  el entorno local con precios de XM, no de estimaciones de la IA. El notebook comprueba con
  `assert` que la herramienta del agente reproduce sus cifras.
- **Los resultados desfavorables se reportan.** Incluso con el diseño nuevo, Q-Learning queda
  un 8.0 % por debajo de la Línea Base en prueba (le gana en validación), y así está
  documentado junto con los resultados del diseño anterior. No se ajustaron hiperparámetros ni
  se eligieron diseños mirando los días de prueba.
- **Las fuentes se verificaron.** Las resoluciones del corpus se descargaron del gestor
  normativo oficial de la CREG, sus URLs están documentadas en `corpus_regulacion/README.md` y
  se declara que la CREG 098 aparece derogada.
- **El código propio cumple el requisito del curso.** El MDP, Q-Learning y la Iteración de
  Valores se implementaron desde cero, sin librerías que ya resuelvan el problema.
- **El autor responde por el resultado.** Todo el código y el texto generados con ayuda de IA
  fueron revisados y aceptados por el autor antes de integrarse al repositorio, y el autor
  puede explicar cada componente.
