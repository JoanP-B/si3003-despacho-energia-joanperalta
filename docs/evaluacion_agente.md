# Evaluación del Copiloto Regulatorio

Generado automáticamente por `tests/test_agent.py` el 2026-11-13 00:16.

- **Modelo:** `openai/gpt-oss-120b` (Groq), T_max = 5, una ejecución por pregunta con memoria limpia.
- **Commit:** `b72d532`
- **Resultado global:** 10/12 casos aprobados (83%).

| Categoría | Aprobados |
|---|---|
| Simulación | 3/4 |
| Regulación | 4/4 |
| Combinada | 1/2 |
| Trampa | 2/2 |

## Criterios

- **Herramientas exactas:** las herramientas invocadas con éxito coinciden con las esperadas (ni faltan ni sobran).
- **Simulación:** se usó la capacidad pedida y la respuesta contiene las cifras exactas de ahorro (Q-Learning y Línea Base) que da `simular()` para esa capacidad.
- **Regulación:** la respuesta cita al menos una fuente, cada cita (archivo y página) corresponde a un fragmento que devolvió el RAG, y el RAG recuperó la resolución esperada.
- **Trampas:** no simula con un valor distinto al pedido ni usa herramientas fuera de alcance, y declina explicando el motivo.
- Ningún caso puede agotar el límite de iteraciones.

## Resultados

| Pregunta | Intención | Herramienta Invocada | Respuesta y Citas | Estado (Pass/Fail) |
|---|---|---|---|---|
| **S1.** Simula el despacho para una batería de 200 kWh. | Ahorro Q-Learning vs. Línea Base, 200 kWh | simular_despacho(capacidad_kwh=200) | **Simulación de despacho para una batería de 200 kWh** \| Política \| Ahorro total (COP) \| \|-------------------------\|--------------------\| \| Q‑Learning (entrenado) \| **$1 892 701** \| \| Línea Base (carga 01‑04 h, descarga 18‑21 h) \| **$4 466 909** \| \| Iteración de Valores (referen… | ✅ Pass |
| **S2.** Si instalo una batería de 500 kWh, ¿cuánto ahorro contra la línea base? | Ahorro frente a la Línea Base, 500 kWh | simular_despacho(capacidad_kwh=500) | Con una batería de **500 kWh** la simulación indica: \| Política \| Ahorro acumulado \| \|----------\|------------------\| \| **Q‑Learning** (política entrenada) \| **$4 731 752 COP** \| \| **Línea Base** (carga 01‑04 h, descarga 18‑21 h) \| **$11 167 273 COP** \| La diferencia entre ambas… | ✅ Pass |
| **S3.** ¿Cuánto ahorraría el agente de RL con una batería de 1000 kWh? | Ahorro del agente RL, 1000 kWh | simular_despacho(capacidad_kwh=1000) | El agente de aprendizaje por refuerzo (Q‑Learning) generaría un ahorro total de **$9 463 505 COP** con una batería de 1 000 kWh (simulación sobre 55 días de precios reales de XM). | ❌ Fail<br>cifra Línea Base exacta |
| **S4.** Compara Q-Learning y la línea base para una batería de 75.5 kWh. | Comparación con capacidad decimal, 75.5 kWh | simular_despacho(capacidad_kwh=75.5) | **Comparación de ahorro (batería 75.5 kWh)** \| Política \| Ahorro total (COP) \| \|----------\|-------------------\| \| **Q‑Learning** \| **$714,495** \| \| **Línea Base** (carga 01‑04 h, descarga 18‑21 h) \| **$1,686,258** \| **Interpretación** - La política basada en Q‑Learning genera **… | ✅ Pass |
| **R1.** ¿Cómo se define un SAEB según la regulación? | Definición de SAEB (CREG 098) | query_creg_regulation(topic='definición SAEB') | Según la Resolución 0098 de 2019 de la CREG, el concepto de SAEB (Sistema de Almacenamiento de Energía con Baterías) se incluye dentro del **Artículo 3 – Definiciones**. En ese artículo se menciona, entre otros aspectos, la **eficiencia mínima de un SAEB**, que se define como “l…<br>**Citas:** resolucion_creg_0098_2019.htm | ✅ Pass |
| **R2.** ¿Qué dice la CREG 174 sobre inyección de excedentes a la red? | Excedentes de autogeneración (CREG 174) | query_creg_regulation(topic='CREG 174 inyección de excedentes a la red') | La Resolución CREG 174 de 2021 regula la forma en que se pueden inyectar los excedentes de energía a la red y establece dos puntos clave: 1. **Destino de la energía** – El artículo 25 indica que la energía inyectada como excedente debe destinarse **exclusivamente a usuarios regu…<br>**Citas:** Creg174-2021.pdf p.30, Creg174-2021.pdf p.30 | ✅ Pass |
| **R3.** ¿Cuál es el límite de potencia de un autogenerador a pequeña escala según la CREG? | Límite de potencia AGPE (CREG 174) | query_creg_regulation(topic='límite de potencia autogenerador pequeña escala CREG') | El límite de potencia máximo para que un autogenerador sea considerado de **pequeña escala** es **1 MW** (capacidad instalada o nominal igual o inferior a ese valor), según lo definido en la Resolución UPME 281 de 2015 y adoptado por la CREG【Fuente: Creg174-2021.pdf, página 2】【F…<br>**Citas:** Creg174-2021.pdf p.2, Creg174-2021.pdf p.7 | ✅ Pass |
| **R4.** ¿Con qué propósito permite la Resolución CREG 098 de 2019 instalar sistemas de almacenamiento? | Objeto de la CREG 098 | query_creg_regulation(topic='Resolución CREG 098 de 2019 propósito instalación sistemas de almacenamiento') | La Resolución CREG 098 de 2019 autoriza la puesta en operación de sistemas de almacenamiento **con el único fin de mitigar los problemas existentes por la falta o insuficiencia de redes de transporte de energía eléctrica** mientras se van definiendo los servicios complementarios…<br>**Citas:** resolucion_creg_0098_2019.htm, resolucion_creg_0098_2019.htm | ✅ Pass |
| **C1.** Simula una batería de 50 kWh y dime qué dice la norma sobre su vida útil. | Simulación 50 kWh + vida útil del SAEB (CREG 098) | query_creg_regulation(topic='vida útil de baterías BESS')<br>simular_despacho(capacidad_kwh=50) | **Simulación de despacho (BESS = 50 kWh)** - **Ahorro total con política Q‑Learning:** **$473 175 COP**. - **Ahorro con la Línea Base (carga 01‑04 h, descarga 18‑21 h):** $1 116 727 COP. - **Ahorro con la política de referencia (Iteración de Valores):** $1 189 177 COP. El modelo…<br>**Citas:** resolucion_creg_0098_2019.htm, resolucion_creg_0098_2019.htm, resolucion_creg_0098_2019.htm, resolucion_creg_0098_2019.htm | ✅ Pass |
| **C2.** Con una batería de 300 kWh, ¿cuánto ahorraría y qué regulación aplica a un autogenerador que entrega excedentes a la red? | Simulación 300 kWh + excedentes (CREG 174) | query_creg_regulation(topic='autogenerador entrega excedentes a la red regulación Colombia')<br>query_creg_regulation(topic='excedentes a la red autogeneración resolución 174 2021')<br>simular_despacho(capacidad_kwh=300) | **Ahorro con la batería de 300 kWh** Según la simulación del despacho con la política de Q‑Learning, la batería generaría un ahorro acumulado de **$2 839 051 COP** durante el período evaluado (55 días). **Regulación aplicable a un autogenerador que entrega excedentes a la red**…<br>**Citas:** Creg174-2021.pdf p.3, Creg174-2021.pdf p.2 | ❌ Fail<br>cifra Línea Base exacta |
| **T1.** Simula el despacho para una batería de -100 kWh. | Capacidad inválida: debe rechazarla, no cambiarla | Ninguna | La solicitud contiene un valor de capacidad negativo (‑100 kWh). La herramienta **simular_despacho** solo acepta capacidades mayores que 0, por lo que no puede procesar la petición tal como está. Si deseas la simulación, por favor indica una capacidad positiva (por ejemplo 100 k… | ✅ Pass |
| **T2.** Dime el clima de mañana en Medellín. | Fuera de alcance: no debe usar herramientas | Ninguna | Lo siento, pero no dispongo de información sobre el clima. Mi función está enfocada en temas regulatorios y operativos de sistemas de almacenamiento de energía con baterías (BESS) en Colombia. Si necesita ayuda con esos asuntos, estaré encantado de asistirle. | ✅ Pass |

## Detalle por caso

### S1 — Simulación: Simula el despacho para una batería de 200 kWh.

- **Estado:** Pass · 2 iteración(es) · 1.7 s
- **Herramientas:** `simular_despacho(capacidad_kwh=200)`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ capacidad pedida
- ✅ cifra Q-Learning exacta
- ✅ cifra Línea Base exacta
- Referencia: Q-Learning $1,892,701 · Línea Base $4,466,909

> **Simulación de despacho para una batería de 200 kWh**
> 
> | Política                | Ahorro total (COP) |
> |-------------------------|--------------------|
> | Q‑Learning (entrenado)  | **$1 892 701** |
> | Línea Base (carga 01‑04 h, descarga 18‑21 h) | **$4 466 909** |
> | Iteración de Valores (referencia) | **$4 756 707** |
> 
> - El Q‑Learning alcanza el **39.8 %** de la referencia y el **42.4 %** de la Línea Base.  
> - Diferencia Q‑Learning vs. Línea Base: **‑$2 574 208** (‑57.6 %).  
> - Operaciones realizadas: **152 cargas**, **152 descargas**, **0 acciones inválidas**.  
> 
> Los precios utilizados provienen de los datos reales del mercado XM.

### S2 — Simulación: Si instalo una batería de 500 kWh, ¿cuánto ahorro contra la línea base?

- **Estado:** Pass · 2 iteración(es) · 1.6 s
- **Herramientas:** `simular_despacho(capacidad_kwh=500)`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ capacidad pedida
- ✅ cifra Q-Learning exacta
- ✅ cifra Línea Base exacta
- Referencia: Q-Learning $4,731,752 · Línea Base $11,167,273

> Con una batería de **500 kWh** la simulación indica:
> 
> | Política | Ahorro acumulado |
> |----------|------------------|
> | **Q‑Learning** (política entrenada) | **$4 731 752 COP** |
> | **Línea Base** (carga 01‑04 h, descarga 18‑21 h) | **$11 167 273 COP** |
> 
> La diferencia entre ambas es:
> 
> **Ahorro contra la Línea Base = $4 731 752 COP − $11 167 273 COP = –$6 435 521 COP**
> 
> Es decir, la política de Q‑Learning ahorra **$6 435 521 COP menos** que la política de Línea Base (un 57,6 % menos).  
> 
> Los precios utilizados provienen de los datos reales de la Bolsa de Energía (XM).

### S3 — Simulación: ¿Cuánto ahorraría el agente de RL con una batería de 1000 kWh?

- **Estado:** Fail · 2 iteración(es) · 1.4 s
- **Herramientas:** `simular_despacho(capacidad_kwh=1000)`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ capacidad pedida
- ✅ cifra Q-Learning exacta
- ❌ cifra Línea Base exacta
- Referencia: Q-Learning $9,463,505 · Línea Base $22,334,547

> El agente de aprendizaje por refuerzo (Q‑Learning) generaría un ahorro total de **$9 463 505 COP** con una batería de 1 000 kWh (simulación sobre 55 días de precios reales de XM).

### S4 — Simulación: Compara Q-Learning y la línea base para una batería de 75.5 kWh.

- **Estado:** Pass · 2 iteración(es) · 2.0 s
- **Herramientas:** `simular_despacho(capacidad_kwh=75.5)`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ capacidad pedida
- ✅ cifra Q-Learning exacta
- ✅ cifra Línea Base exacta
- Referencia: Q-Learning $714,495 · Línea Base $1,686,258

> **Comparación de ahorro (batería 75.5 kWh)**  
> 
> | Política | Ahorro total (COP) |
> |----------|-------------------|
> | **Q‑Learning** | **$714,495** |
> | **Línea Base** (carga 01‑04 h, descarga 18‑21 h) | **$1,686,258** |
> 
> **Interpretación**
> 
> - La política basada en Q‑Learning genera **menos ahorro** que la línea base: una diferencia de **‑$971,764 COP**, lo que representa un **57.6 % menos** que la línea base.
> - En la simulación se consideraron 55 días de evaluación (del 2026‑09‑16 al 2026‑11‑09) con precios reales del mercado mayorista (XM PrecBolsNaci).  
> - Q‑Learning realizó 152 ciclos de carga‑descarga y no tuvo acciones inválidas.
> 
> En resumen, para una batería de 75.5 kWh, la estrategia de carga‑descarga simple de la línea base sigue siendo más rentable que la política aprendida por Q‑Learning bajo las condiciones simuladas.

### R1 — Regulación: ¿Cómo se define un SAEB según la regulación?

- **Estado:** Pass · 2 iteración(es) · 11.3 s
- **Herramientas:** `query_creg_regulation(topic='definición SAEB')`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ cita fuentes
- ✅ citas respaldadas por el RAG
- ✅ RAG recuperó resolucion_creg_0098_2019.htm
- Recuperado: resolucion_creg_0098_2019.htm

> Según la Resolución 0098 de 2019 de la CREG, el concepto de SAEB (Sistema de Almacenamiento de Energía con Baterías) se incluye dentro del **Artículo 3 – Definiciones**. En ese artículo se menciona, entre otros aspectos, la **eficiencia mínima de un SAEB**, que se define como “la cantidad de energía que puede devolver al sistema eléctrico comparada con la cantidad de energía que toma de este; ambas energías medidas en el punto de conexión del SAEB al SIN” (Fuente: resolucion_creg_0098_2019.htm, Artículo 3).  
> 
> El fragmento consultado no reproduce la redacción completa de la definición del SAEB; solo indica que el artículo contiene esa definición y describe el criterio de eficiencia mínima. Por lo tanto, con la información disponible podemos afirmar que la regulación define al SAEB como un sistema de almacenamiento que debe cumplir con un porcentaje mínimo de eficiencia de ciclo completo, pero el texto exacto de la definición no se encuentra en los fragmentos obtenidos.

### R2 — Regulación: ¿Qué dice la CREG 174 sobre inyección de excedentes a la red?

- **Estado:** Pass · 2 iteración(es) · 5.2 s
- **Herramientas:** `query_creg_regulation(topic='CREG 174 inyección de excedentes a la red')`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ cita fuentes
- ✅ citas respaldadas por el RAG
- ✅ RAG recuperó Creg174-2021.pdf
- Recuperado: Creg174-2021.pdf p.23, Creg174-2021.pdf p.30, Creg174-2021.pdf p.61

> La Resolución CREG 174 de 2021 regula la forma en que se pueden inyectar los excedentes de energía a la red y establece dos puntos clave:
> 
> 1. **Destino de la energía** – El artículo 25 indica que la energía inyectada como excedente debe destinarse **exclusivamente a usuarios regulados**.  
>    (Fuente: Creg174-2021.pdf, página 30)
> 
> 2. **Límite de precio** – En situaciones críticas, el **precio de compra de los excedentes** que se haya pactado con base en el precio de bolsa nacional **no podrá superar el precio de escasez ponderado** (según la Resolución CREG 071 de 2006 y sus modificatorias).  
>    (Fuente: Creg174-2021.pdf, página 30)
> 
> En resumen, la CREG 174 permite la inyección de excedentes solo para usuarios regulados y fija un techo al precio de compra de esos excedentes, alineándolo con el precio de escasez del mercado.

### R3 — Regulación: ¿Cuál es el límite de potencia de un autogenerador a pequeña escala según la CREG?

- **Estado:** Pass · 2 iteración(es) · 9.6 s
- **Herramientas:** `query_creg_regulation(topic='límite de potencia autogenerador pequeña escala CREG')`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ cita fuentes
- ✅ citas respaldadas por el RAG
- ✅ RAG recuperó Creg174-2021.pdf
- Recuperado: Creg174-2021.pdf p.2, Creg174-2021.pdf p.3, Creg174-2021.pdf p.7

> El límite de potencia máximo para que un autogenerador sea considerado de **pequeña escala** es **1 MW** (capacidad instalada o nominal igual o inferior a ese valor), según lo definido en la Resolución UPME 281 de 2015 y adoptado por la CREG【Fuente: Creg174-2021.pdf, página 2】【Fuente: Creg174-2021.pdf, página 7】.

### R4 — Regulación: ¿Con qué propósito permite la Resolución CREG 098 de 2019 instalar sistemas de almacenamiento?

- **Estado:** Pass · 2 iteración(es) · 12.1 s
- **Herramientas:** `query_creg_regulation(topic='Resolución CREG 098 de 2019 propósito instalación sistemas de almacenamiento')`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ cita fuentes
- ✅ citas respaldadas por el RAG
- ✅ RAG recuperó resolucion_creg_0098_2019.htm
- Recuperado: Creg174-2021.pdf p.3, resolucion_creg_0098_2019.htm

> La Resolución CREG 098 de 2019 autoriza la puesta en operación de sistemas de almacenamiento **con el único fin de mitigar los problemas existentes por la falta o insuficiencia de redes de transporte de energía eléctrica** mientras se van definiendo los servicios complementarios y la normativa correspondiente (Fuente: resolucion_creg_0098_2019.htm, fragmento 1; Fuente: resolucion_creg_0098_2019.htm, fragmento 2).

### C1 — Combinada: Simula una batería de 50 kWh y dime qué dice la norma sobre su vida útil.

- **Estado:** Pass · 3 iteración(es) · 30.0 s
- **Herramientas:** `query_creg_regulation(topic='vida útil de baterías BESS')`, `simular_despacho(capacidad_kwh=50)`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ capacidad pedida
- ✅ cifra Q-Learning exacta
- ✅ cifra Línea Base exacta
- ✅ cita fuentes
- ✅ citas respaldadas por el RAG
- ✅ RAG recuperó resolucion_creg_0098_2019.htm
- Referencia: Q-Learning $473,175 · Línea Base $1,116,727
- Recuperado: resolucion_creg_0098_2019.htm

> **Simulación de despacho (BESS = 50 kWh)**  
> 
> - **Ahorro total con política Q‑Learning:** **$473 175 COP**.  
> - **Ahorro con la Línea Base (carga 01‑04 h, descarga 18‑21 h):** $1 116 727 COP.  
> - **Ahorro con la política de referencia (Iteración de Valores):** $1 189 177 COP.  
> 
> El modelo Q‑Learning logra el **39,8 %** del ahorro de la política de referencia y es **57,6 %** peor que la Línea Base en este periodo.  
> Los precios utilizados provienen de la bolsa de energía (XM) y son datos reales.  
> 
> **Normativa sobre la vida útil de la batería**  
> 
> Los fragmentos consultados de la Resolución 0098 de 2019 de la CREG (artículos 25‑27) describen requisitos de eficiencia, fronteras comerciales y procesos de carga/descarga, pero **no contienen disposiciones específicas sobre la vida útil esperada de las baterías BESS**.  
> 
> - Fuente: resolucion_creg_0098_2019.htm (artículos 25‑27)【1†Fuente: resolucion_creg_0098_2019.htm】【2†Fuente: resolucion_creg_0098_2019.htm】【3†Fuente: resolucion_creg_0098_2019.htm】
> 
> En consecuencia, la normativa vigente no establece un número de ciclos, años de operación o porcentaje de degradación como “vida útil” mínima para los SAEB/BESS. Si necesita información más detallada, sería necesario revisar otras resoluciones o lineamientos técnicos que complementen la Resolución 0098.

### C2 — Combinada: Con una batería de 300 kWh, ¿cuánto ahorraría y qué regulación aplica a un autogenerador que entrega excedentes a la red?

- **Estado:** Fail · 4 iteración(es) · 55.5 s
- **Herramientas:** `query_creg_regulation(topic='autogenerador entrega excedentes a la red regulación Colombia')`, `query_creg_regulation(topic='excedentes a la red autogeneración resolución 174 2021')`, `simular_despacho(capacidad_kwh=300)`
- ✅ sin agotar T_MAX
- ✅ herramientas exactas
- ✅ capacidad pedida
- ✅ cifra Q-Learning exacta
- ❌ cifra Línea Base exacta
- ✅ cita fuentes
- ✅ citas respaldadas por el RAG
- ✅ RAG recuperó Creg174-2021.pdf
- Referencia: Q-Learning $2,839,051 · Línea Base $6,700,364
- Recuperado: Creg174-2021.pdf p.1, Creg174-2021.pdf p.2, Creg174-2021.pdf p.3, Creg174-2021.pdf p.50, Creg174-2021.pdf p.67

> **Ahorro con la batería de 300 kWh**  
> Según la simulación del despacho con la política de Q‑Learning, la batería generaría un ahorro acumulado de **$2 839 051 COP** durante el período evaluado (55 días).  
> 
> **Regulación aplicable a un autogenerador que entrega excedentes a la red**  
> 
> - La **Resolución 174 de 2021** de la CREG regula las actividades de autogeneración a pequeña escala y la entrega de excedentes a la red mediante un esquema de medición bidireccional que se reconoce como **créditos de energía**【Fuente: Creg174-2021.pdf, página 3】.  
> - El **Decreto 348 de 2017**, citado en la misma resolución, establece los lineamientos para la medición, conexión, contrato de respaldo y liquidación de esos excedentes【Fuente: Creg174-2021.pdf, página 2】.  
> 
> En conjunto, estos instrumentos normativos definen cómo el autogenerador puede conectar su planta, medir la energía entregada y recibir los créditos correspondientes por los excedentes inyectados al sistema de distribución.

### T1 — Trampa: Simula el despacho para una batería de -100 kWh.

- **Estado:** Pass · 1 iteración(es) · 0.7 s
- **Herramientas:** ninguna
- ✅ sin agotar T_MAX
- ✅ no simuló con otro valor
- ✅ explica que la capacidad es inválida

> La solicitud contiene un valor de capacidad negativo (‑100 kWh). La herramienta **simular_despacho** solo acepta capacidades mayores que 0, por lo que no puede procesar la petición tal como está.
> 
> Si deseas la simulación, por favor indica una capacidad positiva (por ejemplo 100 kWh) y, opcionalmente, el rango de fechas que te interese. Con esos datos podré ejecutar la simulación.

### T2 — Trampa: Dime el clima de mañana en Medellín.

- **Estado:** Pass · 1 iteración(es) · 0.7 s
- **Herramientas:** ninguna
- ✅ sin agotar T_MAX
- ✅ sin herramientas
- ✅ declina sin inventar datos

> Lo siento, pero no dispongo de información sobre el clima. Mi función está enfocada en temas regulatorios y operativos de sistemas de almacenamiento de energía con baterías (BESS) en Colombia. Si necesita ayuda con esos asuntos, estaré encantado de asistirle.

