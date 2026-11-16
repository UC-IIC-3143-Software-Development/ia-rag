"""Guardrails simples y deterministas, antes y después del LLM.

No reemplazan una evaluación seria, pero cortan los casos obvios sin gastar tokens.
"""

import re

# Frases típicas de inyección de instrucciones (prompt injection), en español e inglés
PATRONES_INYECCION = [
    r"ignora\w* (todas )?(tus|las) instrucciones",
    r"ignore (all )?(previous|prior) instructions",
    r"para el asistente",
    r"(system|developer) prompt",
    r"act[uú]a como",
    r"a partir de ahora eres",
]

# El orden importa: primero los patrones más largos, para que un RUT no se coma parte de una tarjeta
PII = [
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[TARJETA]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[CORREO]"),
    (re.compile(r"(?:\+?56\s?9|\b9)\s?\d{4}\s?\d{4}\b"), "[TELEFONO]"),
    (re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dkK]\b"), "[RUT]"),
]


def parece_inyeccion(texto: str) -> bool:
    return any(re.search(p, texto, re.I) for p in PATRONES_INYECCION)


def ocultar_pii(texto: str) -> str:
    """Reemplaza RUT, correos, teléfonos y tarjetas antes de guardar trazas."""
    for patron, reemplazo in PII:
        texto = patron.sub(reemplazo, texto)
    return texto


def validar_citas(respuesta: str, n_fuentes: int) -> dict:
    """Revisa que cada cita [n] apunte a una fuente entregada en el prompt."""
    citas = sorted({int(c) for c in re.findall(r"\[(\d+)\]", respuesta)})
    invalidas = [c for c in citas if not 1 <= c <= n_fuentes]
    return {"citas": citas, "invalidas": invalidas, "sin_citas": not citas}
