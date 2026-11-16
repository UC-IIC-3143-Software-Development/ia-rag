import pytest

from asistente_andes.busqueda import Busqueda, Resultado
from asistente_andes.chunking import Chunk


class BuscadorFalso:
    """Devuelve siempre los mismos fragmentos, con el puntaje que se le indique."""

    def __init__(self, puntaje: float = 2.0):
        self.puntaje = puntaje
        self.indice = type("I", (), {"meta": {"chunks": 2, "modelo_embeddings": "falso", "cuarentena": []}})()
        self.llamadas = 0

    def buscar(self, pregunta: str, modo: str = "hibrida+rerank", k: int = 5) -> Busqueda:
        self.llamadas += 1
        chunks = [
            Chunk("politicas/cambios-y-devoluciones.md#0", "politicas/cambios-y-devoluciones.md", "Cambios y devoluciones",
                  "Plazo para cambiar o devolver", "Hay 30 días corridos desde que se recibe el producto."),
            Chunk("politicas/despachos.md#2", "politicas/despachos.md", "Despachos a domicilio", "Costo del despacho",
                  "El despacho es gratis en compras desde $49.990 dentro de la Zona Centro."),
        ]
        return Busqueda([Resultado(c, self.puntaje - i) for i, c in enumerate(chunks)], {"bm25": 0.1, "rerank": 5.0}, 2)


@pytest.fixture(autouse=True)
def trazas_temporales(tmp_path, monkeypatch):
    from asistente_andes import trazas
    ruta = tmp_path / "trazas.jsonl"
    monkeypatch.setattr(trazas.Traza.guardar, "__defaults__", (ruta,))
    return ruta
