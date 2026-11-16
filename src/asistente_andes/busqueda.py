"""Retrieval: BM25, búsqueda vectorial, fusión RRF y reranking con un cross-encoder."""

import re
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder

from .chunking import Chunk
from .config import CONFIG
from .embeddings import Embedder
from .indice import Indice

STOPWORDS = set("""
a al algo como con cual cuales cuando cuanto cuanta cuantos cuantas de del donde e el ella en es esta
este esto hay la las le les lo los me mi mis no o para pero por que quien se si sin su sus te tengo
tiene tienen un una uno unos unas y ya yo puedo puede hacer hago
""".split())


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


def consulta_fts(pregunta: str) -> str:
    """Convierte la pregunta en una consulta FTS5: términos entre comillas unidos con OR."""
    terminos = [t for t in re.findall(r"[\w-]+", _normalizar(pregunta)) if t not in STOPWORDS and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terminos))


@dataclass
class Resultado:
    chunk: Chunk
    puntaje: float


@dataclass
class Busqueda:
    resultados: list[Resultado]
    ms: dict = field(default_factory=dict)  # latencia de cada etapa
    candidatos: int = 0


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: suma 1 / (k + posición) de cada lista. No necesita calibrar puntajes."""
    puntajes: dict[int, float] = {}
    for ranking in rankings:
        for posicion, n in enumerate(ranking, start=1):
            puntajes[n] = puntajes.get(n, 0.0) + 1.0 / (k + posicion)
    return sorted(puntajes.items(), key=lambda x: x[1], reverse=True)


class Buscador:
    def __init__(self, indice: Indice, embedder: Embedder | None = None, con_reranker: bool = True,
                 modelo_reranker: str = CONFIG.modelo_reranker):
        self.indice = indice
        self.embedder = embedder or Embedder(indice.meta["modelo_embeddings"])
        self.reranker = TextCrossEncoder(modelo_reranker, cache_dir=str(CONFIG.modelos)) if con_reranker else None

    def bm25(self, pregunta: str, n: int) -> list[int]:
        q = consulta_fts(pregunta)
        if not q:
            return []
        filas = self.indice.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (q, n)
        ).fetchall()
        return [f[0] for f in filas]

    def vectorial(self, vector: np.ndarray, n: int) -> list[int]:
        # Búsqueda exacta: producto punto contra todas las filas. Con millones de
        # vectores se usa un índice aproximado (ANN), como HNSW en pgvector.
        similitudes = self.indice.matriz @ vector
        return [int(i) for i in np.argsort(-similitudes)[:n]]

    def buscar(self, pregunta: str, modo: str = "hibrida+rerank", k: int = CONFIG.k) -> Busqueda:
        ms, n = {}, CONFIG.candidatos
        t = time.perf_counter()
        rankings = []
        if modo in ("bm25", "hibrida", "hibrida+rerank"):
            rankings.append(self.bm25(pregunta, n))
            ms["bm25"] = (time.perf_counter() - t) * 1000
        if modo in ("vectorial", "hibrida", "hibrida+rerank"):
            t = time.perf_counter()
            vector = self.embedder.consulta(pregunta)
            ms["embedding"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            rankings.append(self.vectorial(vector, n))
            ms["vectorial"] = (time.perf_counter() - t) * 1000
        fusion = rrf(rankings)[:n]
        if modo != "hibrida+rerank":
            top = fusion[:k]
            return Busqueda([Resultado(self.indice.chunks[i], p) for i, p in top], ms, len(fusion))
        t = time.perf_counter()
        textos = [self.indice.chunks[i].texto for i, _ in fusion]
        puntajes = list(self.reranker.rerank(pregunta, textos))
        ms["rerank"] = (time.perf_counter() - t) * 1000
        orden = sorted(zip((i for i, _ in fusion), puntajes), key=lambda x: x[1], reverse=True)[:k]
        return Busqueda([Resultado(self.indice.chunks[i], float(p)) for i, p in orden], ms, len(fusion))
