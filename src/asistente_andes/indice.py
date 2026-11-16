"""Índice en un archivo SQLite: texto con FTS5 (BM25) y embeddings como BLOB.

En producción esto suele vivir en Postgres con pgvector o en una base vectorial
administrada. La interfaz es la misma: guardar fragmentos con su vector y buscar
por palabras y por similitud.
"""

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, dividir, leer_corpus
from .config import CONFIG
from .embeddings import Embedder
from .guardrails import parece_inyeccion

ESQUEMA = [
    "CREATE TABLE chunks (n INTEGER PRIMARY KEY, id TEXT UNIQUE, ruta TEXT, titulo TEXT,"
    " seccion TEXT, texto TEXT, embedding BLOB)",
    # Índice de texto completo con BM25. remove_diacritics: «devolución» calza con «devolucion»
    "CREATE VIRTUAL TABLE chunks_fts USING fts5(texto, tokenize = 'unicode61 remove_diacritics 2')",
    "CREATE TABLE meta (clave TEXT PRIMARY KEY, valor TEXT)",
]


@dataclass
class Indice:
    chunks: list[Chunk]
    matriz: np.ndarray  # una fila por chunk, normalizada
    db: sqlite3.Connection
    meta: dict


def huella(modelo: str, texto: str) -> str:
    """Identifica un fragmento por su contenido: si el texto o el modelo cambian, cambia la huella."""
    return hashlib.sha256((modelo + "\n" + texto).encode()).hexdigest()


def vectores_anteriores(ruta: Path, modelo: str) -> dict[str, np.ndarray]:
    """Vectores del índice anterior, por huella. Si el índice no existe o es de otro modelo, no hay nada que reutilizar."""
    if not ruta.exists():
        return {}
    try:
        with sqlite3.connect(ruta) as db:
            meta = json.loads(db.execute("SELECT valor FROM meta WHERE clave = 'meta'").fetchone()[0])
            if meta.get("modelo_embeddings") != modelo:
                return {}
            filas = db.execute("SELECT texto, embedding FROM chunks").fetchall()
    except sqlite3.Error:
        return {}
    return {huella(modelo, t): np.frombuffer(e, dtype=np.float32) for t, e in filas}


def construir(destino: Path = CONFIG.indice, estrategia: str = CONFIG.estrategia_chunking,
              embedder: Embedder | None = None, maximo: int = CONFIG.chunk_max_caracteres,
              completo: bool = False) -> dict:
    """Pipeline de ingesta: leer, dividir, filtrar, calcular embeddings y guardar.

    Es incremental: un fragmento cuyo texto no cambió reutiliza el vector del índice anterior,
    y solo se calculan los embeddings de los fragmentos nuevos o modificados. completo=True
    recalcula todo, por ejemplo al cambiar de modelo.
    """
    t0 = time.perf_counter()
    docs = leer_corpus(CONFIG.corpus)
    chunks = dividir(docs, estrategia, maximo)
    # Guardrail de ingesta: los fragmentos con instrucciones para el modelo quedan en cuarentena
    cuarentena = [c for c in chunks if parece_inyeccion(c.texto)]
    chunks = [c for c in chunks if c not in cuarentena]
    embedder = embedder or Embedder()
    anteriores = {} if completo else vectores_anteriores(destino, embedder.modelo)
    huellas = [huella(embedder.modelo, c.texto) for c in chunks]
    nuevos = [i for i, h in enumerate(huellas) if h not in anteriores]
    t1 = time.perf_counter()
    calculados = embedder.pasajes([chunks[i].texto for i in nuevos]) if nuevos else np.zeros((0, 0), np.float32)
    calculado = dict(zip(nuevos, calculados))
    matriz = np.vstack([calculado[i] if i in calculado else anteriores[h] for i, h in enumerate(huellas)])
    t2 = time.perf_counter()

    # Se escribe en un archivo temporal y se reemplaza al final: si algo falla, el índice anterior sigue intacto
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".tmp")
    temporal.unlink(missing_ok=True)
    db = sqlite3.connect(temporal)
    for sentencia in ESQUEMA:
        db.execute(sentencia)
    for n, (c, v) in enumerate(zip(chunks, matriz)):
        db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (n, c.id, c.ruta, c.titulo, c.seccion, c.texto, v.tobytes()))
        db.execute("INSERT INTO chunks_fts (rowid, texto) VALUES (?, ?)", (n, c.texto))
    meta = {
        "modelo_embeddings": embedder.modelo,
        "dimension": int(matriz.shape[1]),
        "estrategia": estrategia,
        "documentos": len(docs),
        "chunks": len(chunks),
        "cuarentena": [c.id for c in cuarentena],
        # Actualización incremental: cuántos vectores se calcularon y cuántos se reutilizaron
        "embeddings_calculados": len(nuevos),
        "embeddings_reutilizados": len(chunks) - len(nuevos),
        # Cambia si cambia cualquier fragmento o el modelo: invalida el caché de respuestas
        "version": hashlib.sha256((embedder.modelo + "".join(c.texto for c in chunks)).encode()).hexdigest()[:12],
        "ms_embeddings": round((t2 - t1) * 1000),
        "ms_total": round((time.perf_counter() - t0) * 1000),
    }
    db.execute("INSERT INTO meta VALUES ('meta', ?)", (json.dumps(meta),))
    db.commit()
    db.close()
    os.replace(temporal, destino)
    return meta


def cargar(ruta: Path = CONFIG.indice) -> Indice:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el índice {ruta}. Corre primero: uv run indexar")
    db = sqlite3.connect(ruta, check_same_thread=False)
    filas = db.execute("SELECT id, ruta, titulo, seccion, texto, embedding FROM chunks ORDER BY n").fetchall()
    chunks = [Chunk(id=f[0], ruta=f[1], titulo=f[2], seccion=f[3], texto=f[4]) for f in filas]
    matriz = np.vstack([np.frombuffer(f[5], dtype=np.float32) for f in filas])
    meta = json.loads(db.execute("SELECT valor FROM meta WHERE clave = 'meta'").fetchone()[0])
    return Indice(chunks=chunks, matriz=matriz, db=db, meta=meta)
