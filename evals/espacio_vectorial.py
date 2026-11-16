"""Proyección a 2 dimensiones de los embeddings de algunas preguntas de clientes.

Calcula el vector de E5 de cada pregunta, lo proyecta con PCA a 2 dimensiones y
guarda las coordenadas y el coseno entre pares. La proyección pierde información:
sirve para ver los grupos, no para medir distancias.

Uso: uv run python evals/espacio_vectorial.py
"""

import json
from pathlib import Path

import numpy as np

from asistente_andes.embeddings import Embedder

PREGUNTAS = {
    "despachos": ["¿Cuánto demora el despacho a Arica?", "¿Cuánto demora el despacho a Punta Arenas?", "¿Cuánto cuesta el envío a regiones?"],
    "devoluciones": ["¿Puedo devolver una parka?", "Quiero cambiar unas botas por otra talla", "¿Cuántos días tengo para devolver un producto?"],
    "cuenta": ["¿Cómo borro mi cuenta?", "Olvidé mi contraseña"],
}


def main() -> None:
    textos = [t for grupo in PREGUNTAS.values() for t in grupo]
    grupos = [g for g, lista in PREGUNTAS.items() for _ in lista]
    emb = Embedder()
    v = np.vstack([emb.consulta(t) for t in textos])
    centrado = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(centrado, full_matrices=False)
    xy = centrado @ vt[:2].T
    coseno = v @ v.T
    resultado = {
        "modelo": emb.modelo,
        "dimension": int(v.shape[1]),
        "puntos": [{"texto": t, "grupo": g, "x": round(float(x), 4), "y": round(float(y), 4)} for t, g, (x, y) in zip(textos, grupos, xy)],
        "primeros_valores": {textos[0]: [round(float(x), 3) for x in v[0][:3]] + [round(float(v[0][-1]), 3)]},
        "coseno": {f"{textos[i]} | {textos[j]}": round(float(coseno[i, j]), 3) for i in range(len(textos)) for j in range(i + 1, len(textos))},
    }
    salida = Path(__file__).parent / "resultados" / "espacio_vectorial.json"
    salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in resultado["puntos"]:
        print(f"{p['grupo']:12} {p['x']:+.3f} {p['y']:+.3f}  {p['texto']}")


if __name__ == "__main__":
    main()
