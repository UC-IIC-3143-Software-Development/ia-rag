import asyncio

import anthropic
import httpx

from asistente_andes.llm import ClienteConRespaldo, Uso


class Falla:
    modelo = "claude-opus-5-5"

    def __init__(self, despues_de: int | None = None):
        self.despues_de = despues_de

    async def responder(self, system, mensaje, uso):
        if self.despues_de:
            for i in range(self.despues_de):
                yield f"t{i}"
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        yield  # pragma: no cover


class Respaldo:
    modelo = "gemini-3.5-flash"

    async def responder(self, system, mensaje, uso):
        uso.modelo = self.modelo
        yield "respuesta de respaldo"


def correr(cliente):
    uso = Uso(modelo="claude-opus-5-5")

    async def juntar():
        return "".join([t async for t in cliente.responder("s", "m", uso)])

    return asyncio.run(juntar()), uso


def test_si_claude_falla_antes_del_primer_token_responde_gemini():
    texto, uso = correr(ClienteConRespaldo(Falla(), Respaldo()))
    assert texto == "respuesta de respaldo"
    assert uso.modelo == "gemini-3.5-flash" and uso.respaldo == "APIConnectionError"


def test_si_claude_falla_a_mitad_no_cambia_de_proveedor():
    try:
        correr(ClienteConRespaldo(Falla(despues_de=2), Respaldo()))
    except anthropic.APIConnectionError:
        return
    raise AssertionError("debió propagar la falla")
