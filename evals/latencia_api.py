"""Latencia que ve el cliente: tiempo hasta el primer texto y total, medidos contra la API corriendo.

Manda preguntas distintas del conjunto de evaluación a POST /preguntar, una a la vez, y lee el
stream SSE como lo hace la interfaz. Las preguntas son distintas, así que el caché no actúa.
Necesita la API levantada con clave: uv run --env-file .env uvicorn asistente_andes.app:app

Uso: uv run python evals/latencia_api.py --n 25 --salida latencia_api_opus.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

from asistente_andes.evaluacion import cargar_preguntas


def percentil(valores: list[float], p: float) -> float:
    orden = sorted(valores)
    return orden[min(len(orden) - 1, int(round(p / 100 * len(orden) + 0.5)) - 1)]


def medir(cliente: httpx.Client, url: str, pregunta: str) -> dict:
    t0 = time.perf_counter()
    primero, fin = None, {}
    with cliente.stream("POST", url, json={"pregunta": pregunta}, timeout=60) as r:
        evento = None
        for linea in r.iter_lines():
            if linea.startswith("event: "):
                evento = linea[7:]
            elif linea.startswith("data: "):
                if evento in ("texto", "reemplazar") and primero is None:
                    primero = (time.perf_counter() - t0) * 1000
                if evento == "fin":
                    fin = json.loads(linea[6:])
    return {"primer_texto_ms": round(primero or 0), "total_ms": round((time.perf_counter() - t0) * 1000),
            "costo_usd": fin.get("costo_usd"), "cache": bool(fin.get("cache"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--url", default="http://localhost:8000/preguntar")
    parser.add_argument("--salida", default="latencia_api.json", help="archivo en evals/resultados/")
    args = parser.parse_args()
    preguntas = [p["pregunta"] for p in cargar_preguntas()][: args.n]
    filas = []
    with httpx.Client() as cliente:
        for q in preguntas:
            filas.append({"pregunta": q, **medir(cliente, args.url, q)})
            print(filas[-1])
    primeros = [f["primer_texto_ms"] for f in filas if not f["cache"]]
    resumen = {"preguntas": len(filas), "primer_texto_ms_p50": statistics.median(primeros),
               "primer_texto_ms_p95": percentil(primeros, 95),
               "total_ms_p50": statistics.median(f["total_ms"] for f in filas),
               "dentro_de_3_s": round(sum(p <= 3000 for p in primeros) / len(primeros), 3)}
    print(json.dumps(resumen, ensure_ascii=False))
    destino = Path(__file__).parent / "resultados" / args.salida
    destino.write_text(json.dumps({"resumen": resumen, "filas": filas}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
