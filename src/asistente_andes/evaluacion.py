"""Evaluación offline del retrieval, sin LLM y sin costo.

Para cada pregunta del conjunto de evaluación se sabe qué frase del corpus la
responde (la evidencia). Un acierto en k es que la evidencia aparezca en alguno
de los k primeros fragmentos. Las preguntas sin respuesta en el corpus miden si
el sistema se abstiene: el mejor puntaje del reranker debe quedar bajo el umbral.

Uso:
  uv run evaluar                     compara los cuatro modos sobre el índice actual
  uv run evaluar --chunking          compara estrategias de chunking (reconstruye índices)
  uv run evaluar --minimo 0.9        sale con error si recall@5 queda bajo 0,9 (para CI)
  uv run evaluar --salida otro.json  guarda el resultado con otro nombre en evals/resultados/
"""

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from . import indice as idx
from .busqueda import Buscador
from .config import CONFIG, RAIZ
from .embeddings import Embedder
from .trazas import percentil

PREGUNTAS = RAIZ / "evals" / "preguntas.jsonl"
RESULTADOS = RAIZ / "evals" / "resultados"
MODOS = ["bm25", "vectorial", "hibrida", "hibrida+rerank"]


def _norm(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).lower()


def cargar_preguntas() -> list[dict]:
    return [json.loads(linea) for linea in PREGUNTAS.read_text(encoding="utf-8").splitlines() if linea.strip()]


def evaluar_modo(buscador: Buscador, preguntas: list[dict], modo: str, k: int = 5) -> dict:
    aciertos = {1: 0, 3: 0, 5: 0}
    rr, ms, por_tipo, fallas = [], [], {}, []
    abstenciones_ok, falsas_abstenciones = 0, 0
    top_con, top_sin = [], []  # mejor puntaje de cada pregunta, para elegir el umbral
    con_respuesta = [p for p in preguntas if p["evidencia"]]
    sin_respuesta = [p for p in preguntas if not p["evidencia"]]
    for p in preguntas:
        t = time.perf_counter()
        busqueda = buscador.buscar(p["pregunta"], modo=modo, k=k)
        ms.append((time.perf_counter() - t) * 1000)
        res = busqueda.resultados
        if res:
            (top_con if p["evidencia"] else top_sin).append(res[0].puntaje)
        if not p["evidencia"]:
            if modo == "hibrida+rerank" and (not res or res[0].puntaje < CONFIG.umbral_reranker):
                abstenciones_ok += 1
            continue
        if modo == "hibrida+rerank" and res and res[0].puntaje < CONFIG.umbral_reranker:
            falsas_abstenciones += 1
        posicion = next((i for i, r in enumerate(res, start=1) if _norm(p["evidencia"]) in _norm(r.chunk.texto)), None)
        for corte in aciertos:
            aciertos[corte] += posicion is not None and posicion <= corte
        rr.append(1 / posicion if posicion else 0.0)
        t_tipo = por_tipo.setdefault(p["tipo"], [0, 0])
        t_tipo[0] += posicion is not None and posicion <= k
        t_tipo[1] += 1
        if not posicion:
            fallas.append(p["id"])
    n = len(con_respuesta)
    resultado = {
        "modo": modo,
        "recall@1": round(aciertos[1] / n, 3),
        "recall@3": round(aciertos[3] / n, 3),
        "recall@5": round(aciertos[5] / n, 3),
        "mrr": round(sum(rr) / n, 3),
        "ms_p50": round(percentil(ms, 0.5), 1),
        "ms_p95": round(percentil(ms, 0.95), 1),
        "recall@5_por_tipo": {t: round(a / b, 3) for t, (a, b) in sorted(por_tipo.items())},
        "fallas": fallas,
    }
    if modo == "hibrida+rerank":
        resultado["abstencion_correcta"] = f"{abstenciones_ok}/{len(sin_respuesta)}"
        resultado["abstencion_falsa"] = f"{falsas_abstenciones}/{n}"
        resultado["barrido_umbral"] = [
            {"umbral": u, "abstiene_sin_respuesta": sum(x < u for x in top_sin), "abstiene_con_respuesta": sum(x < u for x in top_con)}
            for u in (-8, -7, -6, -5, -4, -3, -2)
        ]
    resultado["puntaje_top"] = {"con_respuesta": [round(x, 3) for x in top_con], "sin_respuesta": [round(x, 3) for x in top_sin]}
    return resultado


def imprimir(filas: list[dict], titulo: str) -> None:
    print(f"\n{titulo}")
    print(f"{'':22}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}{'p50 ms':>9}{'p95 ms':>9}")
    for f in filas:
        nombre = f.get("estrategia", f["modo"])
        print(f"{nombre:22}{f['recall@1']:>7.2f}{f['recall@3']:>7.2f}{f['recall@5']:>7.2f}{f['mrr']:>7.2f}"
              f"{f['ms_p50']:>9.1f}{f['ms_p95']:>9.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación offline del retrieval")
    parser.add_argument("--chunking", action="store_true", help="compara estrategias de chunking")
    parser.add_argument("--embeddings", nargs="*", help="compara modelos de embeddings")
    parser.add_argument("--minimo", type=float, help="recall@5 mínimo del modo de producción")
    parser.add_argument("--salida", default="modos.json", help="archivo de resultados en evals/resultados/")
    args = parser.parse_args()
    preguntas = cargar_preguntas()
    RESULTADOS.mkdir(parents=True, exist_ok=True)

    if args.chunking or args.embeddings:
        filas = []
        combinaciones = ([(e, CONFIG.modelo_embeddings) for e in ["fijo-300", "fijo-800", "secciones"]]
                         if args.chunking else [("secciones", m) for m in args.embeddings])
        for estrategia, modelo in combinaciones:
            with tempfile.TemporaryDirectory() as tmp:
                embedder = Embedder(modelo)
                meta = idx.construir(Path(tmp) / "indice.db", estrategia, embedder)
                buscador = Buscador(idx.cargar(Path(tmp) / "indice.db"), embedder, con_reranker=False)
                fila = evaluar_modo(buscador, preguntas, "vectorial")
                fila.update(estrategia=f"{estrategia} · {modelo.split('/')[-1]}", chunks=meta["chunks"],
                            dimension=meta["dimension"], ms_indexar=meta["ms_embeddings"])
                filas.append(fila)
        imprimir(filas, "Modo vectorial, según chunking y modelo de embeddings")
        nombre = "chunking.json" if args.chunking else "embeddings.json"
        (RESULTADOS / nombre).write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    indice = idx.cargar()
    buscador = Buscador(indice)
    buscador.buscar("calentamiento de los modelos")  # la primera llamada carga los modelos en memoria
    filas = [evaluar_modo(buscador, preguntas, modo) for modo in MODOS]
    imprimir(filas, f"Retrieval con {indice.meta['chunks']} fragmentos ({indice.meta['estrategia']}), "
                    f"{sum(1 for p in preguntas if p['evidencia'])} preguntas con respuesta")
    # La similitud coseno del mejor fragmento, para comparar con el puntaje del reranker como umbral
    coseno = {"con_respuesta": [], "sin_respuesta": []}
    for p in preguntas:
        top = float((indice.matriz @ buscador.embedder.consulta(p["pregunta"])).max())
        coseno["con_respuesta" if p["evidencia"] else "sin_respuesta"].append(round(top, 3))
    final = filas[-1]
    print(f"\nUmbral {CONFIG.umbral_reranker}: se abstiene en {final['abstencion_correcta']} preguntas sin respuesta "
          f"y en {final['abstencion_falsa']} con respuesta. Fallas: {final['fallas']}")
    print("Barrido de umbral (sin respuesta / con respuesta que se abstienen):",
          ", ".join(f"{b['umbral']}: {b['abstiene_sin_respuesta']}/{b['abstiene_con_respuesta']}" for b in final["barrido_umbral"]))
    print(f"Coseno del mejor fragmento: con respuesta {min(coseno['con_respuesta'])} a {max(coseno['con_respuesta'])}, "
          f"sin respuesta {min(coseno['sin_respuesta'])} a {max(coseno['sin_respuesta'])}")
    config = {"candidatos": CONFIG.candidatos, "umbral_reranker": CONFIG.umbral_reranker, "modelo_reranker": CONFIG.modelo_reranker}
    (RESULTADOS / args.salida).write_text(json.dumps({"meta": indice.meta, "config": config, "modos": filas, "coseno_top": coseno},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    if args.minimo is not None and final["recall@5"] < args.minimo:
        print(f"\nFALLA: recall@5 = {final['recall@5']:.3f}, bajo el mínimo de {args.minimo}")
        sys.exit(1)
