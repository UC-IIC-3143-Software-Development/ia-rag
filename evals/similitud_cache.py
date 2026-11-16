"""¿Sirve un caché semántico en este dominio? Mide la similitud coseno entre pares de preguntas.

Un caché semántico devuelve la respuesta guardada si la pregunta nueva se parece
lo suficiente a una anterior (por ejemplo, coseno de 0,85 o más). El riesgo: dos
preguntas casi iguales con respuestas distintas.

Uso: uv run python evals/similitud_cache.py
"""

import json
from pathlib import Path

from asistente_andes.embeddings import Embedder

PARES = [
    # (pregunta A, pregunta B, ¿misma respuesta?)
    ("¿Cuánto demora el despacho a Punta Arenas?", "¿En cuántos días llega mi pedido a Punta Arenas?", True),
    ("¿Cómo borro mi cuenta?", "Quiero eliminar mi cuenta de Tienda Andes", True),
    ("¿Puedo devolver una parka que no me quedó?", "La parka me quedó chica, ¿la puedo cambiar?", True),
    ("¿Cuánto demora el despacho a Punta Arenas?", "¿Cuánto demora el despacho a Arica?", False),
    ("¿Cuánto cuesta arrendar la carpa 2P por un día?", "¿Cuánto cuesta arrendar la carpa 4P por un día?", False),
    ("¿Puedo devolver una carpa que no usé?", "¿Puedo devolver un cartucho de gas que no usé?", False),
]


def main() -> None:
    emb = Embedder()
    filas = []
    for a, b, igual in PARES:
        coseno = float(emb.consulta(a) @ emb.consulta(b))
        filas.append({"a": a, "b": b, "misma_respuesta": igual, "coseno": round(coseno, 3)})
        print(f"{coseno:.3f}  {'misma' if igual else 'DISTINTA':8}  {a}  |  {b}")
    salida = Path(__file__).parent / "resultados" / "similitud_cache.json"
    salida.write_text(json.dumps({"modelo": emb.modelo, "pares": filas}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
