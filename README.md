# Asistente RAG de Tienda Andes

Asistente del centro de ayuda de Tienda Andes, la misma tienda de equipo de montaña de la clase de Frontend Engineering. Responde preguntas sobre despachos, devoluciones, garantía, arriendo y productos con RAG (retrieval-augmented generation): busca los fragmentos relevantes del centro de ayuda y se los entrega a Claude para que responda citando la fuente.

Sirve para ver, con código real y números medidos, cinco ideas:

1. **Ingesta y read path separados.** `uv run indexar` divide 21 documentos en 88 fragmentos, calcula sus embeddings y los guarda en SQLite. La API solo lee ese índice. La ingesta es incremental: si un documento cambia, solo se recalculan los vectores de sus fragmentos nuevos o modificados.
2. **Retrieval híbrido con reranking.** BM25 (FTS5 de SQLite) y búsqueda vectorial (embeddings E5 de 1.024 dimensiones), fusionados con RRF y reordenados con un cross-encoder.
3. **Saber cuándo no responder.** Si el mejor fragmento queda bajo el umbral del reranker, la API responde que no sabe y no llama al LLM.
4. **Evaluación en CI.** 56 preguntas con su evidencia. Cada PR mide recall@5 y falla si baja de 0,95.
5. **Observabilidad.** Cada petición deja una traza con la latencia de cada etapa, los tokens, el costo (también si responde el modelo de respaldo) y las fuentes. El RUT, el correo, el teléfono y las tarjetas se ocultan, también en lo que el agente pasa a sus herramientas.

También trae un agente (`/agente`) con tres herramientas: buscar en el centro de ayuda, consultar un pedido y consultar el stock de cada tienda. La autorización está en el código de la herramienta: un cliente solo ve sus propios pedidos. El stock no se indexa porque cambia todo el día: se consulta en el momento (`datos/stock.json` simula la API de inventario).

Y un ejemplo mínimo con PostgreSQL y pgvector, `ejemplos/rag_pgvector.py`: el mismo RAG en unas 40 líneas, sin BM25, reranker ni guardrails.

## Arquitectura

```
Ingesta (offline, uv run indexar)
  corpus/*.md -> chunking por secciones -> guardrail de inyección -> embeddings E5 -> SQLite (FTS5 + vectores)

Read path (POST /preguntar)
  pregunta -> caché exacto -> BM25 + vectorial -> RRF -> reranker -> ¿puntaje >= umbral?
     no -> "No encontré esa información" (sin LLM)
     sí -> prompt con 5 fuentes -> Claude con streaming (SSE) -> validación de citas -> traza
     si Claude falla antes del primer token (429, 5xx, red o 8 s sin primer token) y hay GEMINI_API_KEY -> responde Gemini
     si no hay respaldo, o Claude falla a mitad de la respuesta -> evento error en el stream y traza con el error
     si Claude rechaza a mitad de la respuesta -> evento reemplazar con un texto fijo
     al caché solo entra una respuesta completa que cita fuentes y todas existen
```

| Archivo | Qué hace |
|---|---|
| `src/asistente_andes/chunking.py` | Divide por secciones del markdown o en bloques fijos |
| `src/asistente_andes/indice.py` | Pipeline de ingesta e índice SQLite |
| `src/asistente_andes/busqueda.py` | BM25, vectorial, RRF y reranking |
| `src/asistente_andes/prompts.py` | System prompt y armado de las fuentes |
| `src/asistente_andes/llm.py` | Claude con streaming, uso de tokens y costo, y el router con respaldo en Gemini |
| `src/asistente_andes/app.py` | API FastAPI: `/preguntar`, `/agente`, `/metricas`, `/salud` |
| `src/asistente_andes/agente.py` | Loop del agente con tres herramientas y `MAX_PASOS` |
| `src/asistente_andes/guardrails.py` | Inyección de instrucciones, PII y citas |
| `src/asistente_andes/trazas.py` | Trazas JSONL y percentiles |
| `src/asistente_andes/evaluacion.py` | Evaluación offline del retrieval |
| `evals/preguntas.jsonl` | 56 preguntas: 50 con respuesta y 6 sin respuesta |
| `sql/pgvector.sql` | El esquema del índice en PostgreSQL con pgvector, HNSW y filtro por metadatos, para cuando el corpus crezca |
| `ejemplos/rag_pgvector.py` | RAG mínimo con pgvector: indexar y preguntar, en unas 40 líneas |
| `.github/workflows/evaluacion.yml` | Tests y evaluación en cada PR |
| `demo/regresion.patch` | Un PR que rompe el retrieval |

## Correr en local

Necesitan Python 3.13 y [uv](https://docs.astral.sh/uv/). La primera vez se descargan los modelos de embeddings y de reranking, cerca de 3 GB, a `.cache/modelos`.

```bash
uv sync
uv run indexar                       # ingesta: unos 50 s la primera vez
uv run pytest                        # 25 tests, sin modelos ni API
uv run evaluar                       # compara BM25, vectorial, híbrida y con reranker
uv run uvicorn asistente_andes.app:app --port 8000
```

Abran `http://localhost:8000`. Sin `ANTHROPIC_API_KEY`, la API corre en modo simulado: hace todo el retrieval y devuelve las fuentes, sin generar texto. Con la clave, responde Claude (por defecto `claude-opus-5-5` con esfuerzo `low`, configurable con `MODELO_LLM` y `ESFUERZO_LLM`). Si además definen `GEMINI_API_KEY`, Gemini responde cuando Claude falla antes del primer token.

Las claves van en un archivo `.env`, que no se sube al repositorio:

```bash
cp .env.example .env                 # y completen sus claves
uv run --env-file .env uvicorn asistente_andes.app:app --port 8000
curl -N -X POST localhost:8000/preguntar -H 'Content-Type: application/json' \
  -d '{"pregunta": "¿Cuánto demora un despacho a Coyhaique?"}'
curl -X POST localhost:8000/agente -H 'Content-Type: application/json' -H 'X-Cliente: cli-001' \
  -d '{"mensaje": "Mi pedido A-10231 llegó el 23 de octubre, ¿todavía puedo devolver la parka?"}'
curl localhost:8000/metricas
```

## Evaluación

```bash
uv run evaluar                  # modos de búsqueda sobre el índice actual
uv run evaluar --chunking       # fijo-300, fijo-800 y por secciones
uv run evaluar --embeddings sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 intfloat/multilingual-e5-large
uv run python evals/similitud_cache.py
uv run python evals/tokens.py   # tokens del prompt y del corpus, con el tokenizador de E5
uv run python evals/espacio_vectorial.py   # proyección PCA de 8 preguntas y coseno entre pares
uv run python evals/escala_vectorial.py    # búsqueda exacta con 88, 10.000, 100.000 y 1 millón de vectores
uv run python evals/mapa_pregunta.py       # coseno de una pregunta con cada fragmento, para el mapa de la clase

# Variantes que se guardan con otro nombre, sin pisar modos.json
CANDIDATOS=20 uv run evaluar --salida modos-candidatos-20.json
INDICE=/tmp/minilm.db MODELO_EMBEDDINGS=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 uv run indexar
INDICE=/tmp/minilm.db MODELO_EMBEDDINGS=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 uv run evaluar --salida modos-minilm.json
INDICE=/tmp/fijo300.db CHUNKING=fijo-300 uv run indexar
INDICE=/tmp/fijo300.db CHUNKING=fijo-300 uv run evaluar --salida modos-fijo-300.json
MODELO_RERANKER=BAAI/bge-reranker-v2-m3 uv run evaluar --salida modos-reranker-completo.json

uv run python evals/bm25_paso_a_paso.py "¿Puedo devolver un cartucho de gas que no usé?"

# Con la API real: necesitan ANTHROPIC_API_KEY y tienen costo
uv run --env-file .env python evals/evaluar_respuestas.py --muestra 50 --salida respuestas-opus.json
MODELO_LLM=claude-haiku-5-5 MODELO_JUEZ=claude-opus-5-5 uv run --env-file .env python evals/evaluar_respuestas.py --muestra 50 --salida respuestas-haiku.json
uv run --env-file .env python evals/tres_versiones.py      # sin contexto, todo el corpus y RAG, con la misma pregunta
uv run --env-file .env python evals/tokens_claude.py       # tokens con el tokenizador de Claude, sin costo por token
uv run python evals/latencia_api.py --n 25 --salida latencia_api_opus.json   # contra la API levantada
```

Resultados en un MacBook con Apple Silicon, en CPU (`evals/resultados/`):

| Modo | Recall@5 | MRR | p50 |
|---|---|---|---|
| BM25 | 0,88 | 0,73 | 0,0 ms |
| Vectorial | 0,98 | 0,93 | 21 ms |
| Híbrida (RRF) | 0,98 | 0,83 | 18 ms |
| Híbrida con reranker | 1,00 | 0,96 | 923 ms |

Con el umbral de −5,0, el reranker se abstiene en 4 de las 6 preguntas sin respuesta y en ninguna de las 50 con respuesta. Con la similitud coseno los grupos se mezclan más: su mejor umbral (0,82) también se abstiene en 4 de 6, pero falla en 1 de 50 con respuesta. Con 20 candidatos en vez de 10, la búsqueda con reranker tarda 2.631 ms y la calidad baja un poco: recall@5 de 0,98 (`modos-candidatos-20.json`).

Son 56 preguntas escritas a mano sobre un corpus chico. Muestran la dirección, no reemplazan una evaluación con preguntas reales de usuarios.

### Con la API real (16 de noviembre de 2026)

Respuestas de 50 preguntas, juzgadas por otro modelo (`juez-v2`, que ve el system prompt):

| Modelo que responde | Juez | Fiel | Correcta | US$ por pregunta | Primer token p50 / p95 |
|---|---|---|---|---|---|
| Claude Opus 5.5 | Sonnet 5.5 | 0,94 | 0,98 | 0,0083 | 710 / 2.103 ms |
| Claude Sonnet 5.5 | Opus 5.5 | 0,90 | 1,00 | 0,0037 | 782 / 1.145 ms |
| Claude Haiku 5.5 | Opus 5.5 | 0,92 | 1,00 | 0,0002 | 622 / 1.626 ms |

La primera versión del juez no veía el system prompt y marcó como no fieles 7 respuestas que sugerían el correo de ayuda que ese prompt pide sugerir. Esa corrida quedó en `evals/resultados/juez-v1/`.

Tiempo hasta el primer texto medido desde el cliente, 25 preguntas contra la API (`latencia_api_*.json`):

| Modelo | p50 | p95 | Dentro de 3 s |
|---|---|---|---|
| Claude Opus 5.5 | 1,98 s | 13,7 s | 76% |
| Claude Haiku 5.5 | 1,54 s | 3,0 s | 92% |

Con Opus, en 3 de las 25 preguntas el primer texto no llegó en 8 s, porque el modelo razona antes de escribir, y respondió Gemini.

Las tres versiones con «¿Cuánto se demora en llegar un pedido a Coyhaique?» (`tres_versiones.json`): solo la pregunta, 34 tokens, y Claude inventa un plazo de 3 a 7 días. Todo el corpus, 10.183 tokens y US$ 0,048, y acierta. RAG, 1.199 tokens y US$ 0,009, y acierta citando la fuente.

## Uso de disco

Los datos son livianos y los modelos, pesados. Medido con `du` en este proyecto:

| Qué | Disco | Para qué |
|---|---|---|
| E5 large | 2,25 GB | Embeddings, en producción |
| Reranker BGE-M3 int8 | 0,59 GB | Reranking, en producción |
| Dependencias de Python (`.venv`) | 0,23 GB | En producción |
| Índice con 88 fragmentos y sus vectores | 0,5 MB | Datos |
| Reranker BGE-M3 completo | 2,27 GB | Solo para comparar |
| Reranker Jina v2 | 1,13 GB | Solo para comparar |
| MiniLM-L12 | 0,25 GB | Solo para comparar |

Cada réplica carga 2,84 GB de modelos, y el CI los guarda en caché. Comparar tres modelos más agregó 3,66 GB. La versión int8 del reranker ocupa 4 veces menos que la completa, con la misma calidad (recall@5 de 1,00 en ambas, MRR de 0,96 frente a 0,95) y más rápida (923 frente a 1.235 ms de p50): `modos-reranker-completo.json`. Para recuperar espacio, se puede borrar `.cache/modelos` y dejar que se descarguen de nuevo solo los modelos que se usan.

## Romper el CI a propósito

`demo/regresion.patch` simula un PR razonable: cortar los documentos en bloques fijos de 300 caracteres para que el reranker sea más rápido. La búsqueda con reranker baja de 923 a 566 ms, pero recall@5 cae de 1,00 a 0,84 y el check falla. Parte de la caída es de la métrica: en 5 de las 8 fallas, la frase de evidencia quedó partida entre dos bloques.

```bash
git checkout -b chunks-fijos
git apply demo/regresion.patch
uv run indexar && uv run evaluar --minimo 0.95   # FALLA: recall@5 = 0.840
```
