"""Embeddings locales con fastembed (ONNX en CPU, sin GPU ni API externa)."""

import numpy as np
from fastembed import TextEmbedding

from .config import CONFIG


class Embedder:
    def __init__(self, modelo: str = CONFIG.modelo_embeddings):
        self.modelo = modelo
        self._m = TextEmbedding(modelo, cache_dir=str(CONFIG.modelos))
        # Los modelos E5 se entrenaron con prefijos distintos para consultas y pasajes
        self._e5 = "e5" in modelo.lower()

    def _vectores(self, textos: list[str]) -> np.ndarray:
        v = np.array(list(self._m.embed(textos)), dtype=np.float32)
        # Normalizados a largo 1: así el producto punto es la similitud coseno
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def pasajes(self, textos: list[str]) -> np.ndarray:
        return self._vectores([f"passage: {t}" if self._e5 else t for t in textos])

    def consulta(self, texto: str) -> np.ndarray:
        return self._vectores([f"query: {texto}" if self._e5 else texto])[0]
