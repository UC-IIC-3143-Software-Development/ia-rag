"""RAG completo en pocas líneas: E5 para los vectores, pgvector para guardarlos y Claude para responder.

Es el núcleo del asistente, sin BM25, reranker, umbral ni guardrails, para ver lo simple que es.
Necesita PostgreSQL con la extensión pgvector y una base llamada tienda_andes:
  brew install postgresql@17 pgvector && brew services start postgresql@17 && createdb tienda_andes

Uso:
  uv run --env-file .env python ejemplos/rag_pgvector.py indexar
  uv run --env-file .env python ejemplos/rag_pgvector.py "¿Cuánto demora un despacho a Coyhaique?"
"""

import sys

import anthropic
import psycopg
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

from asistente_andes.chunking import dividir, leer_corpus
from asistente_andes.config import CONFIG

modelo = TextEmbedding("intfloat/multilingual-e5-large", cache_dir=str(CONFIG.modelos))
con = psycopg.connect("dbname=tienda_andes", autocommit=True)
con.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(con)
SYSTEM = "Responde solo con las fuentes y cita su número."


def indexar():
    # Ingesta: cortar los documentos, un vector por fragmento, guardar
    frags = dividir(leer_corpus(CONFIG.corpus))
    vecs = modelo.embed(["passage: " + f.texto for f in frags])
    con.execute("DROP TABLE IF EXISTS fragmentos")
    con.execute("CREATE TABLE fragmentos (id text PRIMARY KEY,"
                " texto text, embedding vector(1024))")
    for f, v in zip(frags, vecs):
        con.execute("INSERT INTO fragmentos VALUES (%s, %s, %s)",
                    (f.id, f.texto, v))
    con.execute("CREATE INDEX ON fragmentos"
                " USING hnsw (embedding vector_cosine_ops)")
    print(len(frags), "fragmentos guardados en pgvector")


def preguntar(pregunta):
    # Consulta: vectorizar la pregunta, traer los 5 más cercanos
    q = next(modelo.embed(["query: " + pregunta]))
    filas = con.execute("SELECT id, texto FROM fragmentos"
                        " ORDER BY embedding <=> %s LIMIT 5",
                        (q,)).fetchall()
    # Aumentar el prompt con las fuentes numeradas y generar
    fuentes = "\n".join(f"[{i}] {texto}"
                        for i, (_, texto) in enumerate(filas, 1))
    contenido = f"<fuentes>{fuentes}</fuentes>\n{pregunta}"
    r = anthropic.Anthropic().messages.create(
        model="claude-opus-5-5", max_tokens=500, system=SYSTEM,
        messages=[{"role": "user", "content": contenido}])
    print("Fuentes:", ", ".join(id_ for id_, _ in filas))
    print("".join(b.text for b in r.content if b.type == "text"))


if __name__ == "__main__":
    indexar() if sys.argv[1] == "indexar" else preguntar(sys.argv[1])
