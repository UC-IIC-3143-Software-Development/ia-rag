import numpy as np

from asistente_andes import indice as idx


class EmbedderFalso:
    modelo = "falso"

    def __init__(self):
        self.calculados = 0

    def pasajes(self, textos):
        self.calculados += len(textos)
        v = np.random.default_rng(len(textos)).random((len(textos), 8), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_la_ingesta_reutiliza_los_vectores_de_fragmentos_sin_cambios(tmp_path):
    destino = tmp_path / "indice.db"
    primera = idx.construir(destino, embedder=EmbedderFalso())
    assert primera["embeddings_calculados"] == primera["chunks"]

    embedder = EmbedderFalso()
    segunda = idx.construir(destino, embedder=embedder)
    assert embedder.calculados == 0 and segunda["embeddings_reutilizados"] == segunda["chunks"]
    assert segunda["version"] == primera["version"]
    assert np.allclose(idx.cargar(destino).matriz.shape, (segunda["chunks"], 8))

    completa = idx.construir(destino, embedder=EmbedderFalso(), completo=True)
    assert completa["embeddings_calculados"] == completa["chunks"]
