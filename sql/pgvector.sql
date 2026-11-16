-- El índice de Tienda Andes en PostgreSQL con pgvector, para cuando el corpus crezca.
-- El proyecto usa SQLite (src/asistente_andes/indice.py) y búsqueda exacta con numpy:
-- con 88 vectores tarda 0,006 ms (evals/escala_vectorial.py).
-- Se probó con PostgreSQL 17 y pgvector 0.8.7, cargando los 88 vectores del índice: para
-- «¿Cuánto se demora en llegar un pedido a Coyhaique?», la primera fila es
-- politicas/despachos.md#1 con coseno 0,865, igual que en SQLite. Con tan pocas filas,
-- PostgreSQL recorre la tabla completa y no usa el índice HNSW: el índice sirve con miles de filas.
-- ejemplos/rag_pgvector.py usa este mismo patrón desde Python.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE fragmentos (
  id        text PRIMARY KEY,        -- politicas/despachos.md#1
  ruta      text NOT NULL,           -- documento de origen, sirve para filtrar
  titulo    text,
  seccion   text,
  texto     text NOT NULL,
  embedding vector(1024) NOT NULL    -- intfloat/multilingual-e5-large
);

-- Índice aproximado HNSW con distancia coseno. m y ef_construction son los valores
-- por defecto: más conexiones dan más recall, a cambio de memoria y tiempo de construcción.
CREATE INDEX ON fragmentos USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Candidatos que revisa cada consulta (por defecto 40): más es más recall y más latencia.
SET hnsw.ef_search = 100;

-- Los 10 fragmentos más cercanos a la pregunta ($1, su vector), solo de las políticas.
SELECT id, seccion, 1 - (embedding <=> $1) AS coseno
FROM fragmentos
WHERE ruta LIKE 'politicas/%'
ORDER BY embedding <=> $1
LIMIT 10;
