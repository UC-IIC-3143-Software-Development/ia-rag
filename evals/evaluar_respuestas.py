"""Evaluación de respuestas con un LLM como juez. Necesita ANTHROPIC_API_KEY y tiene costo.

Para cada pregunta con respuesta: recupera las fuentes, genera la respuesta con el
mismo prompt de la API y le pide al juez dos veredictos binarios, porque un sí o no
es más estable que una escala de 1 a 10:
- fiel: cada afirmación de la respuesta está respaldada por las fuentes.
- correcta: la respuesta coincide con la respuesta esperada del conjunto de evaluación.

El prompt del juez es parte del código: si cambia, cambia la versión de la evaluación.

La muestra es aleatoria con semilla fija, y el juez puede ser otro modelo (MODELO_JUEZ)
para que un modelo no se evalúe a sí mismo.

Uso: uv run python evals/evaluar_respuestas.py --muestra 10
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import anthropic
from pydantic import BaseModel

from asistente_andes import indice as idx
from asistente_andes.busqueda import Buscador
from asistente_andes.config import CONFIG
from asistente_andes.evaluacion import cargar_preguntas
from asistente_andes.llm import FALLBACK_BETA, ClienteClaude, Uso
from asistente_andes.prompts import RESPUESTA_SIN_RESPALDO, SYSTEM, mensaje_usuario

# v2: el juez también ve el system prompt. En la v1 no lo veía y marcaba como no fiel el correo
# de ayuda que el system prompt pide sugerir: 7 de 50 respuestas de Opus eran falsos negativos.
VERSION_JUEZ = "juez-v2"
JUEZ = """Evalúas respuestas de un asistente de atención al cliente. Recibes las instrucciones del asistente,
las fuentes que tuvo, la pregunta, la respuesta esperada y la respuesta del asistente.
- fiel: true solo si cada afirmación de la respuesta se puede verificar en las fuentes o en las instrucciones.
- correcta: true si la respuesta entrega la misma información esencial que la respuesta esperada.
Explica tu veredicto en una frase."""


class Veredicto(BaseModel):
    fiel: bool
    correcta: bool
    explicacion: str


def percentil(valores: list[float], p: float) -> float | None:
    if not valores:
        return None
    orden = sorted(valores)
    return orden[min(len(orden) - 1, int(round(p / 100 * len(orden) + 0.5)) - 1)]


async def evaluar(muestra: int, nombre_salida: str) -> None:
    buscador = Buscador(idx.cargar())
    generador = ClienteClaude(interactivo=False)  # en segundo plano: más reintentos y sin tope al primer token
    juez = anthropic.AsyncAnthropic(max_retries=4)
    con_respuesta = [p for p in cargar_preguntas() if p["evidencia"]]
    preguntas = random.Random(42).sample(con_respuesta, min(muestra, len(con_respuesta)))
    salida = Path(__file__).parent / "resultados" / nombre_salida
    filas, costo, costo_completo = [], 0.0, True
    for p in preguntas:
        try:
            resultados = buscador.buscar(p["pregunta"]).resultados
            mensaje = mensaje_usuario(p["pregunta"], resultados)
            uso = Uso(modelo=generador.modelo)
            # Igual que la API: bajo el umbral no se llama al LLM
            if not resultados or resultados[0].puntaje < CONFIG.umbral_reranker:
                respuesta = RESPUESTA_SIN_RESPALDO
            else:
                respuesta = "".join([t async for t in generador.responder(SYSTEM, mensaje, uso)])
            veredicto = await juez.beta.messages.parse(
                model=CONFIG.modelo_juez,
                max_tokens=2000,
                system=JUEZ,
                messages=[{"role": "user", "content": (
                    f"<instrucciones_del_asistente>{SYSTEM}</instrucciones_del_asistente>\n\n"
                    f"{mensaje}\n\n<respuesta_esperada>{p['respuesta_esperada']}</respuesta_esperada>\n\n"
                    f"<respuesta_del_asistente>{respuesta}</respuesta_del_asistente>")}],
                output_format=Veredicto,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
            v = veredicto.parsed_output
            if v is None:  # el juez no entregó un veredicto válido, por ejemplo por un rechazo
                raise ValueError(f"sin veredicto ({veredicto.stop_reason})")
        except Exception as e:
            # Una falla no corta la corrida: queda registrada y se sigue con la siguiente pregunta
            filas.append({"id": p["id"], "error": f"{type(e).__name__}: {e}"})
            print(f"{p['id']}  ERROR {type(e).__name__}")
            continue
        if uso.costo_usd is None and uso.entrada:
            costo_completo = False
        costo += uso.costo_usd or 0.0
        filas.append({"id": p["id"], "fiel": v.fiel, "correcta": v.correcta, "explicacion": v.explicacion,
                      "respuesta": respuesta, "ttft_ms": round(uso.ttft_ms or 0), "total_ms": round(uso.total_ms),
                      "tokens_entrada": uso.entrada, "tokens_salida": uso.salida})
        print(f"{p['id']}  fiel={v.fiel!s:5}  correcta={v.correcta!s:5}  {v.explicacion}")
        # Se guarda después de cada pregunta: si la corrida se corta, no se pierde lo evaluado
        salida.write_text(json.dumps({"filas": filas}, ensure_ascii=False, indent=1), encoding="utf-8")
    juzgadas = [f for f in filas if "error" not in f]
    n = len(juzgadas)
    resumen = {
        "juez": VERSION_JUEZ, "modelo_juez": CONFIG.modelo_juez, "modelo": CONFIG.modelo_llm,
        "preguntas": len(filas), "con_error": len(filas) - n,
        "fidelidad": round(sum(f["fiel"] for f in juzgadas) / n, 3) if n else None,
        "correctas": round(sum(f["correcta"] for f in juzgadas) / n, 3) if n else None,
        # Si algún intento usó un modelo sin precio en costos.py, el total queda incompleto
        "costo_generacion_usd": round(costo, 4) if costo_completo else None,
        "costo_por_pregunta_usd": round(costo / n, 5) if n and costo_completo else None,
        "ttft_ms_p50": percentil([f["ttft_ms"] for f in juzgadas if f["ttft_ms"]], 50),
        "ttft_ms_p95": percentil([f["ttft_ms"] for f in juzgadas if f["ttft_ms"]], 95),
        "total_ms_p50": percentil([f["total_ms"] for f in juzgadas if f["ttft_ms"]], 50),
        "tokens_entrada_promedio": round(sum(f["tokens_entrada"] for f in juzgadas) / n) if n else None,
        "tokens_salida_promedio": round(sum(f["tokens_salida"] for f in juzgadas) / n) if n else None,
    }
    print(json.dumps(resumen, ensure_ascii=False))
    salida.write_text(json.dumps({"resumen": resumen, "filas": filas}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--muestra", type=int, default=10)
    parser.add_argument("--salida", default="respuestas.json", help="archivo en evals/resultados/")
    args = parser.parse_args()
    asyncio.run(evaluar(args.muestra, args.salida))
