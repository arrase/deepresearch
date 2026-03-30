# Prompt De Implementacion: Deep Research

Actua como un Arquitecto de Software Principal y como un implementador senior de sistemas con LangChain y LangGraph.

Tu tarea es disenar e implementar desde cero una herramienta de investigacion asistida por IA capaz de producir informes utiles, profundos, trazables y respaldados por evidencia verificable a partir de una pregunta abierta del usuario.

La implementacion debe estar optimizada para modelos pequenos ejecutados localmente con Ollama, debe usar Lightpanda como navegador principal para recuperar contenido web y debe priorizar profundidad real de investigacion por encima de prosa convincente.

No construyas un agente ReAct generico ni un pipeline que reinyecte todo el historial en cada llamada. Implementa una arquitectura explicita y auditable basada en LangGraph, con control estricto del estado, del flujo y de la seleccion de contexto en funcion de la configuracion activa.

## Mision Del Sistema

El sistema debe convertir una pregunta amplia o ambigua en un proceso disciplinado de:

1. descomposicion en subpreguntas,
2. descubrimiento y priorizacion de fuentes,
3. navegacion web fiable,
4. extraccion de evidencia atomica,
5. evaluacion de cobertura y huecos,
6. sintesis final respaldada por citas.

La tesis operativa es esta:

- la calidad con modelos pequenos depende de seleccionar mejor el contexto, no de enviar mas texto,
- el contexto bruto es caro,
- la memoria de trabajo debe ser pequena,
- la evidencia debe ser atomica y citable,
- los resumenes intermedios deben ser jerarquicos, acumulativos y auditables,
- la investigacion debe detenerse por suficiencia informativa, no por agotamiento mecanico del pipeline.

## Objetivo Del Entregable

La aplicacion final debe aceptar una pregunta de investigacion abierta y producir un informe que incluya como minimo:

- respuesta a la pregunta principal,
- hallazgos clave,
- nivel de confianza y reservas,
- huecos pendientes,
- fuentes citadas de forma trazable.

El sistema solo es valido si el informe final esta sustentado por evidencia localizable. Un texto bien redactado sin soporte verificable no cumple el objetivo.

## Restricciones No Negociables

### Modelos

- Usa ChatOllama exclusivamente como backend de LLM.
- Asume ventanas de contexto modestas y salida estructurada imperfecta.
- Prefiere pocas llamadas bien preparadas.
- Disena prompts cortos, defensivos y atomicos.
- No dependas de prompts largos para obtener calidad.

### Framework

- Usa Python.
- Usa LangChain y LangGraph.
- Usa Pydantic para esquemas y validacion estructurada.

### Navegacion

- Usa Lightpanda como navegador principal: [Lightpanda](https://github.com/lightpanda-io/browser)
- No disenes el producto alrededor de Chromium como requisito base.
- La integracion concreta puede ser via API, CLI o wrapper propio, pero el contrato funcional debe quedar resuelto en codigo.

### Memoria Y Contexto

- Prohibido usar ConversationBufferMemory o cualquier acumulacion ilimitada de mensajes.
- Prohibido reenviar por defecto todo el historial, todo el plan o toda la evidencia acumulada.
- Cada nodo debe construir su propio contexto minimo.
- El tamano de contexto efectivo debe definirse desde configuracion o desde un parametro de ejecucion segun la VRAM disponible.
- La arquitectura debe adaptarse al tamano de contexto elegido sin exigir un presupuesto manual por etapa.

## Principios Arquitectonicos

1. La evidencia manda. La arquitectura se organiza alrededor de evidencia contrastable, no alrededor de historiales largos entre prompts.
2. El contexto se gestiona como una capacidad configurable. Cada etapa debe recibir solo la informacion que necesita dentro del tamano de contexto fijado por el usuario.
3. El estado de investigacion debe ser estructurado y util para decidir.
4. La profundidad se construye iterativamente cerrando huecos concretos.
5. Las decisiones del sistema deben ser explicables y auditables.
6. La degradacion ante fallos del modelo, navegador o fuentes debe ser elegante y diagnosticable.
7. La escritura final es una etapa de sintesis, no el lugar donde el sistema compensa una investigacion superficial.

## Arquitectura Objetivo

Implementa como minimo estos componentes logicos:

### 1. Orquestador De Investigacion

Responsabilidades:

- coordinar el ciclo global,
- mantener el estado canonico,
- decidir el orden de las etapas,
- aplicar criterios de parada,
- emitir eventos o checkpoints observables.

No debe convertirse en un objeto monolitico con toda la logica incrustada.

### 2. Gestor De Contexto

Este es el componente mas importante.

Responsabilidades:

- decidir que contexto recibe cada llamada,
- comprimir hallazgos sin destruir matices utiles,
- mantener resumenes de trabajo en varios niveles,
- seleccionar solo la evidencia necesaria para cada decision,
- evitar enviar historiales completos o paginas enteras salvo necesidad estricta.

Debe operar con capas como:

- pregunta principal y objetivo,
- subpreguntas activas,
- hipotesis abiertas,
- backlog de huecos,
- evidencia atomica validada,
- dossier resumido,
- contexto local de la etapa actual.

### 3. Coordinador De Subagentes Internos

La arquitectura debe apoyarse en subagentes internos o workers especializados para tratar informacion intermedia sin inflar el contexto de un unico agente central.

Responsabilidades:

- aislar tareas estrechas y repetibles,
- procesar evidencia intermedia antes de que llegue al dossier principal,
- resumir, normalizar o clasificar contenido localmente,
- reducir el acoplamiento entre navegacion, extraccion, evaluacion y sintesis,
- permitir que cada paso opere con contexto pequeno y especifico.

Estos subagentes no deben comportarse como agentes autonomos de proposito general con memoria conversacional infinita. Deben ser unidades acotadas, con entradas y salidas claras, integradas en el flujo del grafo.

El diseno debe separar de forma explicita dos familias de subagentes:

- subagentes deterministas: reglas, validadores, rankers heuristicos, deduplicadores, selectores, clasificadores por umbral, ensambladores de contexto y transformadores de estado que no requieren inferencia abierta,
- subagentes basados en LLM: tareas donde si existe una necesidad real de interpretacion, compresion semantica, extraccion flexible, evaluacion de cobertura o sintesis.

Regla de arquitectura:

- toda tarea que pueda resolverse de forma fiable y mantenible sin LLM debe implementarse como subagente determinista,
- el uso de LLM debe reservarse para pasos donde aporte capacidad real que no pueda sustituirse razonablemente con logica determinista,
- cada subagente debe declarar su tipo, sus entradas, sus salidas y el motivo de esa eleccion.

### 4. Planificador

Responsabilidades:

- convertir la pregunta en una agenda de investigacion,
- definir subpreguntas reales,
- proponer ejes de contraste,
- generar terminos de busqueda iniciales de alta rentabilidad.

No debe intentar resolver toda la investigacion en esta etapa.

### 5. Descubridor Y Priorizador De Fuentes

Responsabilidades:

- descubrir fuentes candidatas,
- deduplicar URLs y variantes reales,
- agrupar documentos o historias equivalentes,
- priorizar por valor informativo,
- introducir diversidad de fuentes cuando aporte cobertura.

No debe ser una cola ingenua de URLs.

### 6. Navegador Web Con Lightpanda

Responsabilidades:

- cargar paginas,
- esperar el contenido util sin sobrerenderizar,
- extraer el texto visible relevante,
- detectar bloqueos, captchas, paywalls, vacios y errores,
- devolver una representacion textual limpia y estable.

El resultado del navegador debe distinguir claramente entre:

- pagina util,
- pagina parcial,
- pagina bloqueada,
- pagina vacia,
- error tecnico recuperable,
- error tecnico terminal.

### 7. Extractor De Evidencia

Responsabilidades:

- convertir contenido recuperado en evidencia atomica,
- separar hechos, afirmaciones, reservas y contexto,
- adjuntar metadatos utiles,
- conservar cita suficiente para rastrear el origen.

Cada evidencia atomica debe ser pequena, legible y reutilizable sin arrastrar la pagina completa.

### 8. Evaluador De Cobertura

Responsabilidades:

- medir que subpreguntas tienen soporte suficiente,
- detectar contradicciones y huecos,
- decidir la siguiente mejor accion,
- impedir cierres prematuros.

No debe basarse solo en numero de iteraciones, URLs visitadas o ausencia de pendientes.

### 9. Sintetizador Final

Responsabilidades:

- redactar la respuesta final solo con evidencia seleccionada,
- separar conclusiones, evidencia y reservas,
- hacer visibles los huecos abiertos,
- citar las fuentes usadas.

### 10. Telemetria Y Artefactos

Responsabilidades:

- registrar decisiones importantes,
- guardar checkpoints utiles,
- permitir auditoria posterior,
- facilitar debugging con modelos pequenos y web inestable.

## Estado Global Obligatorio

El nucleo debe ser un estado estructurado de LangGraph, definido con TypedDict y/o modelos Pydantic.

Debe incluir como minimo:

- query: pregunta original,
- active_subqueries: subpreguntas pendientes,
- resolved_subqueries: subpreguntas con cobertura suficiente,
- search_queue: cola priorizada de URLs o candidatos,
- visited_urls: registro de URLs procesadas y su estado,
- discarded_sources: fuentes descartadas y motivo,
- atomic_evidence: almacen de evidencias aceptadas,
- contradictions: contradicciones detectadas,
- open_gaps: vacios de informacion abiertos,
- working_dossier: resumen jerarquico acumulativo,
- context_window_config: tamano de contexto objetivo, origen de configuracion y politica de seleccion aplicada,
- final_report: estado o artefacto de salida,
- is_sufficient: booleano de suficiencia.

El estado no debe ser una bolsa amorfa de texto. Cada campo debe servir para decidir algo.

## Estrategia Obligatoria De Contexto

### Regla General

Cada llamada al modelo debe construirse con una composicion deliberada de contexto. Nunca reenvies por defecto todo el historial ni toda la evidencia acumulada.

### Capas De Contexto

La arquitectura debe contemplar al menos:

- contexto permanente: pregunta principal y restricciones del encargo,
- contexto estrategico: subpreguntas, hipotesis y huecos activos,
- contexto operativo: tarea puntual de la etapa actual,
- contexto probatorio: solo la evidencia necesaria para esa etapa,
- contexto local de fuente: fragmentos minimos del documento actual.

### Compresion Jerarquica

No uses truncado ciego por caracteres o tokens. Implementa compresion semantica con varios niveles, por ejemplo:

- fragmentos o evidencias pequenas,
- mini-resumenes por fuente,
- resumenes por subpregunta o tema,
- dossier general de investigacion,
- contexto final de evaluacion o redaccion.

Cada nivel debe sintetizar el anterior, no solo recortarlo.

### Configuracion De Contexto

No impongas un presupuesto interno de tokens por etapa como parte central del diseno. El tamano de contexto a utilizar debe venir definido por el usuario, ya sea desde fichero de configuracion o desde un parametro de ejecucion, en funcion de la VRAM disponible y del modelo elegido.

La responsabilidad del sistema es adaptarse bien a ese tamano de contexto:

- seleccionar solo el contexto realmente necesario,
- comprimir y jerarquizar informacion intermedia,
- delegar tratamiento local de informacion en subagentes acotados,
- evitar que una sola llamada cargue con todo el trabajo acumulado.

El sistema puede estimar longitudes o validar que una llamada no exceda el limite real del modelo, pero eso debe tratarse como una comprobacion tecnica, no como la pieza central de la arquitectura.

La seleccion de contexto tambien debe apoyarse preferentemente en componentes deterministas cuando la tarea sea de filtrado, ranking, deduplicacion o ensamblado, dejando al LLM solo la parte semantica que realmente lo necesite.

### Seleccion De Evidencia

La evidencia incluida en una llamada debe entrar por pertinencia, no por orden de llegada.

Considera criterios como:

- relacion con la subpregunta activa,
- novedad respecto al dossier,
- confiabilidad aproximada,
- diversidad de fuente,
- capacidad de cerrar un hueco,
- capacidad de refutar una hipotesis incorrecta.

## Lightpanda Como Requisito Funcional

Disena la integracion con Lightpanda para que el sistema pueda:

- extraer texto visible de forma estable,
- esperar lo suficiente para contenido importante sin degradar toda la investigacion,
- detectar si una pagina carga pero no aporta valor,
- conservar metadatos utiles para diagnostico,
- distinguir entre fallo de renderizado, bloqueo, contenido pobre e irrelevancia real.

No trates una pagina sin evidencia util como si automaticamente hubiese sido una visita exitosa.

## Extraccion Y Validacion De Evidencia

La salida estructurada del modelo es una ayuda imperfecta. No la trates como verdad incuestionable.

Debes contemplar:

- validacion estructural,
- coercion prudente ante fallos menores de tipos,
- reintentos controlados cuando la salida sea casi util,
- fallback razonable cuando el JSON no salga perfecto,
- descartes explicables cuando una fuente no aporte evidencia util.

Cada evidencia aceptada debe incluir, como minimo:

- URL o identificador de fuente,
- titulo legible,
- resumen breve,
- hechos o afirmaciones concretas,
- cita suficiente para volver al origen,
- senal de relevancia,
- senal de confianza o reservas.

## Criterio De Parada

La investigacion no debe terminar solo porque:

- se agotaron las URLs pendientes,
- se alcanzo un maximo de iteraciones,
- el navegador fallo varias veces,
- ya existe texto redactable.

La investigacion debe terminar cuando exista base razonable para responder y se cumplan de forma suficiente estas condiciones:

- las subpreguntas centrales tienen cobertura,
- los huecos restantes son conocidos y acotados,
- la evidencia aceptada sostiene los hallazgos principales,
- el informe puede distinguir hechos, inferencias y dudas.

Los limites tecnicos pueden existir como proteccion, pero no sustituyen el criterio informativo.

## Especificacion De Nodos En LangGraph

Implementa al menos los siguientes nodos como funciones, clases Runnable o composiciones LCEL:

Cada nodo puede apoyarse en subagentes internos o workers especializados siempre que se cumplan estas reglas:

- su responsabilidad sea estrecha y verificable,
- no dependan de memoria conversacional acumulativa,
- produzcan artefactos intermedios claros,
- no sustituyan el control explicito del estado global del grafo.

Ademas, cada nodo debe preferir subagentes deterministas por defecto. Si usa un subagente LLM, debe ser por una necesidad semantica concreta y no por comodidad de implementacion.

### node_planner

Entrada:

- query

Accion:

- descomponer la pregunta en subpreguntas,
- generar hipotesis de trabajo,
- producir terminos de busqueda iniciales.

Puede apoyarse en subagentes de expansion de consulta o de normalizacion de agenda si eso mejora la calidad sin inflar el contexto.

La normalizacion estructural de la agenda, el formateo y la validacion posterior deben ser deterministas.

Salida esperada:

- active_subqueries,
- search intents o consultas iniciales,
- criterios iniciales de exito.

### node_source_manager

Entrada:

- active_subqueries,
- working_dossier,
- resultados de busqueda,
- visited_urls,
- open_gaps.

Accion:

- descubrir candidatos,
- deduplicar,
- priorizar por novedad y valor potencial,
- actualizar search_queue.

Puede apoyarse en subagentes de clustering, deduplicacion y scoring documental.

La deduplicacion, la canonizacion de URLs, el filtrado tecnico y el scoring heuristico inicial deben ser deterministas. El LLM solo debe intervenir si hace falta valorar novedad semantica, cobertura o utilidad probable mas alla de reglas simples.

### node_browser

Entrada:

- siguiente elemento de search_queue.

Accion:

- usar Lightpanda,
- cargar la pagina,
- extraer contenido util,
- clasificar el resultado,
- actualizar visited_urls y metadatos de navegacion.

Si hay fallo terminal, no fuerces paso al extractor.

La clasificacion tecnica de errores de navegacion debe ser determinista.

### node_extractor

Entrada:

- fragmento de texto de la fuente actual,
- subpregunta objetivo,
- minimo contexto necesario.

Accion:

- extraer evidencia atomica,
- separar hechos de reservas,
- emitir estructura validable.

Puede apoyarse en subagentes de segmentacion, seleccion de pasajes y normalizacion de evidencia antes de consolidar la salida final.

La limpieza de texto, el troceado, la deteccion de duplicados exactos y la validacion estructural de la evidencia deben ser deterministas.

Regla critica:

- este nodo no debe recibir el historial completo.

### node_context_manager

Entrada:

- nueva evidencia,
- working_dossier actual,
- contexto estrategico minimo.

Accion:

- integrar la evidencia nueva,
- resumir incrementalmente,
- eliminar redundancia,
- mantener el dossier util dentro del tamano de contexto configurado.

### node_evaluator

Entrada:

- working_dossier,
- active_subqueries,
- resolved_subqueries,
- atomic_evidence,
- open_gaps.

Accion:

- evaluar cobertura,
- mover subpreguntas resueltas,
- identificar contradicciones,
- definir siguientes huecos,
- decidir si is_sufficient pasa a verdadero.

Puede apoyarse en subagentes de deteccion de contradicciones o de analisis de huecos si mantienen entradas y salidas acotadas.

Las comprobaciones de cobertura minima, consistencia estructural y suficiencia basada en umbrales configurables deben resolverse de forma determinista siempre que sea posible.

### node_synthesizer

Entrada:

- query,
- working_dossier,
- atomic_evidence,
- contradictions,
- open_gaps.

Accion:

- redactar el informe final,
- no inferir mas alla de la evidencia,
- incluir reservas, huecos y fuentes.

## Flujo Del Grafo

Implementa edges y conditional edges equivalentes a:

1. START -> Planner
2. Planner -> Source_Manager
3. Source_Manager -> Browser
4. Browser -> Extractor si hay contenido util
5. Browser -> Source_Manager si hay fallo terminal o pagina inutil
6. Extractor -> Context_Manager
7. Context_Manager -> Evaluator
8. Evaluator -> Synthesizer si is_sufficient es verdadero o se alcanza un limite de fallback bien justificado
9. Evaluator -> Source_Manager si sigue faltando cobertura y existen huecos accionables

## Anti-Patrones Que Debes Evitar

No hagas ninguna de estas cosas:

1. Reinyectar continuamente demasiado contexto crudo.
2. Truncar texto en bruto como estrategia principal.
3. Dar por suficiente una investigacion sin evidencia aceptada.
4. Perder la informacion de que una busqueda o una fuente fallaron.
5. Suponer que el modelo devolvera JSON perfecto siempre.
6. Basar la calidad en prompts largos.
7. Convertir el sintetizador final en el verdadero investigador.
8. Usar LLM para tareas puramente mecanicas o deterministas que deberian resolverse con codigo normal.
9. Ocultar que subagentes son deterministas y cuales dependen de inferencia.

## Requisitos Tecnicos De Implementacion

- Usa PydanticOutputParser donde proceda.
- Combinalo con RetryWithErrorOutputParser o con fallbacks equivalentes para modelos pequenos.
- Implementa validacion y reparacion prudente de salidas estructuradas.
- Manten telemetria basica por ciclo.
- Implementa un self-check o mecanismo equivalente para validar modelo, navegador y piezas criticas.
- Diferencia explicitamente entre falta de evidencia, fallo tecnico y bloqueo de fuente.
- Documenta para cada subagente si es determinista o LLM-driven y por que.

## Entregables Requeridos

Genera el codigo de forma modular. Como minimo entrega:

- state.py: esquemas TypedDict y modelos Pydantic del estado, evidencia y artefactos,
- tools.py: integracion de Lightpanda y herramientas auxiliares de busqueda,
- nodes.py: implementacion de nodos y logica LCEL,
- subagents/deterministic.py o modulo equivalente: workers internos deterministas,
- subagents/llm.py o modulo equivalente: workers internos basados en LLM,
- subagents/__init__.py o modulo equivalente: contratos y registro de subagentes,
- graph.py: ensamblaje del StateGraph, nodos, edges y compilacion,
- main.py: punto de entrada con telemetria basica por ciclo.

Si durante la implementacion detectas que una estructura distinta mejora la separacion de responsabilidades, puedes ampliarla, pero debes conservar esos contratos conceptuales.

## Criterios De Aceptacion

La implementacion solo se considera correcta si cumple de forma verificable:

### Funcionales

- ejecuta una investigacion end-to-end a partir de una pregunta abierta,
- usa Ollama,
- usa Lightpanda,
- produce un informe final sustentado en evidencia,
- mantiene trazabilidad entre hallazgos y fuentes.

### Arquitectonicos

- existe un gestor de contexto real,
- el estado es estructurado y sirve para decidir,
- las etapas estan desacopladas por responsabilidad,
- el sistema degrada con elegancia ante fallos,
- existe separacion explicita entre subagentes deterministas y subagentes LLM.

### Calidad Con Modelos Pequenos

- sigue siendo util con ventanas de contexto modestas,
- no depende de prompts largos,
- la seleccion de evidencia y el dossier incremental mejoran la profundidad,
- no se da por finalizada una investigacion sin soporte suficiente.

### Operativos

- existe self-check,
- el sistema distingue entre falta de evidencia, fallo tecnico y bloqueo,
- la telemetria permite explicar resultados pobres sin inspeccion manual completa.

## Libertad De Implementacion

Tienes libertad para mejorar:

- la estructura interna,
- los contratos entre componentes,
- la politica de priorizacion,
- la representacion de evidencia,
- la estrategia de compresion de contexto,
- la integracion concreta con Lightpanda,
- la forma de evaluar suficiencia.

Lo no negociable es cumplir la finalidad del producto y respetar las restricciones de este documento.

## Instruccion Final De Ejecucion

Implementa el sistema. No entregues solo una propuesta teorica. Produce codigo real, modular y razonado. Documenta en docstrings las decisiones criticas, especialmente las relacionadas con configuracion de contexto, compresion jerarquica, uso de subagentes internos, validacion de evidencia y manejo de fallos con modelos pequenos de Ollama.

Cuando una decision de implementacion sea ambigua, elige la opcion que mejor preserve:

1. trazabilidad de evidencia,
2. robustez con modelos pequenos,
3. control explicito del estado,
4. auditabilidad del proceso,
5. utilidad practica del informe final.
