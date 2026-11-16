"""Costo por petición a partir del uso de tokens que devuelve la API.

Precios en dólares por millón de tokens, de la tabla oficial de precios de Anthropic
(platform.claude.com/docs/en/about-claude/pricing). Revísenla antes de usar estas cifras.
"""

PRECIOS = {
    # modelo: (entrada, salida, lectura de caché)
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    # Modelos a los que puede pasar la petición si el principal la rechaza
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
}
ESCRITURA_CACHE = 1.25  # escribir en caché por 5 minutos cuesta 1,25 veces la entrada


def costo_usd(modelo: str, entrada: int, salida: int, cache_lectura: int = 0, cache_escritura: int = 0) -> float | None:
    if modelo not in PRECIOS:
        return None
    p_in, p_out, p_cache = PRECIOS[modelo]
    total = entrada * p_in + salida * p_out + cache_lectura * p_cache + cache_escritura * p_in * ESCRITURA_CACHE
    return round(total / 1_000_000, 6)
