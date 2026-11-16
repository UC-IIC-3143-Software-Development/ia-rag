"""Distancia de una pregunta a cada fragmento del índice, para dibujar la búsqueda vectorial.

Calcula el coseno entre el vector de la pregunta y el de cada fragmento, y marca los 5
más cercanos. Guarda una fila por fragmento: documento, sección y coseno. La lámina
dibuja la distancia como 1 - coseno, con la pregunta al centro.

Uso: uv run python evals/mapa_pregunta.py
"""

import json
from pathlib import Path

import numpy as np

from asistente_andes import indice as idx
from asistente_andes.embeddings import Embedder

PREGUNTA = "¿Cuánto se demora en llegar un pedido a Coyhaique?"


def main() -> None:
    indice = idx.cargar()
    q = Embedder().consulta(PREGUNTA)
    cosenos = indice.matriz @ q
    orden = np.argsort(-cosenos)
    top = set(int(i) for i in orden[:5])
    filas = [{"id": c.id, "ruta": c.ruta, "seccion": c.seccion, "coseno": round(float(cosenos[i]), 4), "top5": i in top}
             for i, c in enumerate(indice.chunks)]
    salida = Path(__file__).parent / "resultados" / "mapa_pregunta.json"
    salida.write_text(json.dumps({"pregunta": PREGUNTA, "fragmentos": filas}, ensure_ascii=False, indent=1), encoding="utf-8")
    for i in orden[:8]:
        c = indice.chunks[int(i)]
        print(round(float(cosenos[i]), 3), c.id, c.seccion)
    print("min", round(float(cosenos.min()), 3), "max", round(float(cosenos.max()), 3))


if __name__ == "__main__":
    main()
