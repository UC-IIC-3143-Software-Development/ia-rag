"""Comandos de línea: indexar el corpus y consultar sin levantar la API."""

import argparse
import json

from . import indice as idx
from .busqueda import Buscador


def indexar() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de ingesta: corpus, chunks, embeddings e índice")
    parser.add_argument("--chunking", default=None, help="secciones, fijo-300 o fijo-800")
    parser.add_argument("--completo", action="store_true", help="recalcula todos los embeddings")
    args = parser.parse_args()
    meta = idx.construir(**({"estrategia": args.chunking} if args.chunking else {}), completo=args.completo)
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def buscar() -> None:
    parser = argparse.ArgumentParser(description="Muestra qué fragmentos recupera cada modo")
    parser.add_argument("pregunta")
    parser.add_argument("--modo", default="hibrida+rerank")
    args = parser.parse_args()
    busqueda = Buscador(idx.cargar()).buscar(args.pregunta, modo=args.modo)
    for i, r in enumerate(busqueda.resultados, start=1):
        print(f"[{i}] {r.puntaje:7.3f}  {r.chunk.id}  ({r.chunk.seccion})")
    print({k: round(v, 1) for k, v in busqueda.ms.items()})
