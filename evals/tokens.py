"""Cuántos tokens tiene el prompt de una pregunta y el centro de ayuda completo.

Cuenta con el tokenizador de E5, el modelo de embeddings del proyecto. Claude usa otro
tokenizador y da otra cifra para el mismo texto: esto sirve para estimar el orden de
magnitud y el costo, no para facturar. Para contar con el tokenizador de Claude está
messages.count_tokens de la API, que necesita ANTHROPIC_API_KEY.

Uso: uv run python evals/tokens.py
"""

import json
import statistics
from pathlib import Path

from tokenizers import Tokenizer

from asistente_andes import indice as idx
from asistente_andes.busqueda import Buscador
from asistente_andes.config import CONFIG
from asistente_andes.evaluacion import cargar_preguntas
from asistente_andes.prompts import SYSTEM, mensaje_usuario


def main() -> None:
    archivo = next(CONFIG.modelos.glob("models--Qdrant--multilingual-e5-large-onnx/snapshots/*/tokenizer.json"))
    tok = Tokenizer.from_file(str(archivo))
    contar = lambda texto: len(tok.encode(texto).ids)
    buscador = Buscador(idx.cargar())
    prompts = []
    for p in cargar_preguntas():
        if p["evidencia"]:
            mensaje = mensaje_usuario(p["pregunta"], buscador.buscar(p["pregunta"]).resultados)
            prompts.append(contar(SYSTEM) + contar(mensaje))
    corpus = "".join(a.read_text(encoding="utf-8") for a in sorted(CONFIG.corpus.rglob("*.md")))
    resultado = {
        "tokenizador": "intfloat/multilingual-e5-large",
        "system_prompt": contar(SYSTEM),
        "prompt_mediana": statistics.median(prompts),
        "prompt_min": min(prompts),
        "prompt_max": max(prompts),
        "preguntas": len(prompts),
        "corpus_tokens": contar(corpus),
        "corpus_caracteres": len(corpus),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    salida = Path(__file__).parent / "resultados" / "tokens.json"
    salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
