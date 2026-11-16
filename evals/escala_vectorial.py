"""Cuánto tarda la búsqueda vectorial exacta a medida que crece el índice.

Usa vectores aleatorios normalizados de 1.024 dimensiones, como los de E5: el tiempo
de la búsqueda exacta depende del tamaño de la matriz, no del contenido. Mide el
producto de la matriz por la consulta más la selección de los 10 mejores, como hace
busqueda.py. No sirve para medir índices aproximados (HNSW): con vectores aleatorios,
sin la estructura de los embeddings reales, su recall no es representativo.

Uso: uv run python evals/escala_vectorial.py
"""

import json
import platform
import time
from pathlib import Path

import numpy as np

TAMANOS = [88, 10_000, 100_000, 1_000_000]


def main() -> None:
    rng = np.random.default_rng(0)
    filas = []
    for n in TAMANOS:
        matriz = rng.standard_normal((n, 1024), dtype=np.float32)
        matriz /= np.linalg.norm(matriz, axis=1, keepdims=True)
        consulta = rng.standard_normal(1024, dtype=np.float32)
        consulta /= np.linalg.norm(consulta)
        tiempos = []
        for _ in range(30):
            t0 = time.perf_counter()
            similitudes = matriz @ consulta
            np.argsort(-similitudes)[:10] if n <= 10_000 else np.argpartition(-similitudes, 10)[:10]
            tiempos.append((time.perf_counter() - t0) * 1000)
        tiempos.sort()
        filas.append({"vectores": n, "p50_ms": round(tiempos[len(tiempos) // 2], 3), "memoria_mb": round(matriz.nbytes / 1e6, 1)})
        print(filas[-1])
        del matriz
    salida = Path(__file__).parent / "resultados" / "escala_vectorial.json"
    salida.write_text(json.dumps({"cpu": platform.processor() or platform.machine(), "dimension": 1024, "filas": filas}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
