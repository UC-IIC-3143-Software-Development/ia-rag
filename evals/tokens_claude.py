"""Tokens de las tres versiones con el tokenizador de Claude (messages.count_tokens). Necesita ANTHROPIC_API_KEY.

Contar tokens no genera texto y no tiene costo por token. Complementa evals/tokens.py, que usa el
tokenizador de E5 y corre sin clave.

Uso: uv run --env-file .env python evals/tokens_claude.py
"""

import json
import statistics
from pathlib import Path

import anthropic

from asistente_andes import indice as idx
from asistente_andes.busqueda import Buscador
from asistente_andes.config import CONFIG
from asistente_andes.evaluacion import cargar_preguntas
from asistente_andes.prompts import SYSTEM, mensaje_usuario


def main() -> None:
    client = anthropic.Anthropic()
    buscador = Buscador(idx.cargar())

    def cuenta(system: str, mensaje: str) -> int:
        extra = {"system": system} if system else {}
        return client.messages.count_tokens(model=CONFIG.modelo_llm, messages=[{"role": "user", "content": mensaje}],
                                            **extra).input_tokens

    preguntas = [p for p in cargar_preguntas() if p["evidencia"]]
    v3 = [cuenta(SYSTEM, mensaje_usuario(p["pregunta"], buscador.buscar(p["pregunta"]).resultados)) for p in preguntas]
    corpus = "\n\n".join(a.read_text(encoding="utf-8") for a in sorted(CONFIG.corpus.rglob("*.md")))
    resultado = {
        "tokenizador": f"{CONFIG.modelo_llm} (messages.count_tokens)",
        "v1_solo_pregunta": cuenta("", preguntas[0]["pregunta"]),
        "v2_todo_el_corpus": cuenta(SYSTEM, f"<documentos>\n{corpus}\n</documentos>\n<pregunta>{preguntas[0]['pregunta']}</pregunta>"),
        "v3_mediana": statistics.median(v3), "v3_min": min(v3), "v3_max": max(v3),
        "system_prompt_aprox": cuenta(SYSTEM, "x"),
    }
    destino = Path(__file__).parent / "resultados" / "tokens_claude.json"
    destino.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    print(resultado)


if __name__ == "__main__":
    main()
