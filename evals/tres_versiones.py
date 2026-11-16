"""Las tres versiones del asistente con la misma pregunta, contra la API real. Necesita ANTHROPIC_API_KEY.

v1: solo la pregunta. v2: todo el centro de ayuda en cada pregunta. v3: RAG con los 5 fragmentos
recuperados. Guarda la respuesta, los tokens, el costo y el tiempo hasta el primer token.

Uso: uv run --env-file .env python evals/tres_versiones.py
"""

import json
import time
from pathlib import Path

import anthropic

from asistente_andes import indice as idx
from asistente_andes.busqueda import Buscador
from asistente_andes.chunking import leer_corpus
from asistente_andes.config import CONFIG
from asistente_andes.costos import costo_usd
from asistente_andes.prompts import SYSTEM, mensaje_usuario

PREGUNTA = "¿Cuánto se demora en llegar un pedido a Coyhaique?"


def llamar(client: anthropic.Anthropic, system: str | None, contenido: str) -> dict:
    t0, ttft, texto = time.perf_counter(), None, []
    extra = {"system": system} if system else {}
    with client.messages.stream(model=CONFIG.modelo_llm, max_tokens=CONFIG.max_tokens,
                                output_config={"effort": CONFIG.esfuerzo_llm},
                                messages=[{"role": "user", "content": contenido}], **extra) as stream:
        for t in stream.text_stream:
            ttft = ttft or (time.perf_counter() - t0) * 1000
            texto.append(t)
        final = stream.get_final_message()
    u = final.usage
    return {"respuesta": "".join(texto), "tokens_entrada": u.input_tokens, "tokens_salida": u.output_tokens,
            "costo_usd": costo_usd(final.model, u.input_tokens, u.output_tokens, 0, 0),
            "ttft_ms": round(ttft or 0), "total_ms": round((time.perf_counter() - t0) * 1000)}


def main() -> None:
    client = anthropic.Anthropic()
    todo = "\n".join(d.texto for d in leer_corpus(CONFIG.corpus))
    frags = Buscador(idx.cargar()).buscar(PREGUNTA)
    salida = {
        "pregunta": PREGUNTA, "modelo": CONFIG.modelo_llm,
        "v1_solo_pregunta": llamar(client, None, PREGUNTA),
        "v2_todo_el_corpus": llamar(client, SYSTEM, f"<documentos>\n{todo}\n</documentos>\n\n<pregunta>{PREGUNTA}</pregunta>"),
        "v3_rag": llamar(client, SYSTEM, mensaje_usuario(PREGUNTA, frags.resultados)),
    }
    destino = Path(__file__).parent / "resultados" / "tres_versiones.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    for v in ("v1_solo_pregunta", "v2_todo_el_corpus", "v3_rag"):
        r = salida[v]
        print(f"{v}: {r['tokens_entrada']} tokens de entrada, US$ {r['costo_usd']}, primer token {r['ttft_ms']} ms")
        print("   ", r["respuesta"][:400].replace("\n", " "))


if __name__ == "__main__":
    main()
