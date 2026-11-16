"""Clientes del LLM con streaming. Sin credenciales de Anthropic, usa un modo simulado.

Con las dos claves, un router usa Claude y, si falla antes del primer token, Gemini.

El modo simulado no genera texto: devuelve las fuentes recuperadas. Sirve para correr
los tests y el CI sin gastar tokens, y para mostrar el retrieval por separado.
"""

import asyncio
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import anthropic

from .config import CONFIG
from .costos import costo_usd

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Uso:
    modelo: str  # el modelo que respondió, que puede ser el de respaldo
    entrada: int = 0
    salida: int = 0
    cache_lectura: int = 0
    cache_escritura: int = 0
    ttft_ms: float | None = None  # tiempo hasta el primer token de texto
    total_ms: float = 0.0
    motivo_fin: str = ""
    costo: float = 0.0
    costo_conocido: bool = True
    respaldo: str = ""  # si respondió el proveedor de respaldo, por qué falló el principal

    def sumar(self, usage, modelo: str) -> None:
        """Suma el uso de una respuesta. Si actuó el modelo de respaldo, cada intento trae su modelo y su tarifa."""
        intentos = [it for it in (getattr(usage, "iterations", None) or [])
                    if it.type in ("message", "fallback_message")]
        for it in intentos or [usage]:
            lectura = it.cache_read_input_tokens or 0
            escritura = it.cache_creation_input_tokens or 0
            self.entrada += it.input_tokens
            self.salida += it.output_tokens
            self.cache_lectura += lectura
            self.cache_escritura += escritura
            c = costo_usd(getattr(it, "model", None) or modelo, it.input_tokens, it.output_tokens, lectura, escritura)
            if c is None:
                self.costo_conocido = False
            else:
                self.costo += c

    @property
    def costo_usd(self) -> float | None:
        return round(self.costo, 6) if self.costo_conocido else None


def hay_credenciales() -> bool:
    if os.environ.get("MODO_LLM") in ("claude", "simulado"):
        return os.environ["MODO_LLM"] == "claude"
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class ClienteClaude:
    def __init__(self, modelo: str = CONFIG.modelo_llm, interactivo: bool = True):
        self.modelo = modelo
        # Con un usuario esperando: un solo reintento (429, 5xx o conexión), tope al primer token y fallar rápido.
        # En segundo plano (evaluaciones): más reintentos y sin tope. read es el máximo entre dos fragmentos.
        self.tope_primer_token = CONFIG.primer_token_max_s if interactivo else None
        self.client = anthropic.AsyncAnthropic(
            max_retries=1 if interactivo else 4,
            timeout=anthropic.Timeout(30.0, read=15.0, write=10.0, connect=2.0))

    async def responder(self, system: str, mensaje: str, uso: Uso) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        # Tope para el primer token: si no llega a tiempo, TimeoutError y la API avisa al usuario
        async with (
            asyncio.timeout(self.tope_primer_token) as tope,
            self.client.beta.messages.stream(
                model=self.modelo,
                max_tokens=CONFIG.max_tokens,
                system=system,
                messages=[{"role": "user", "content": mensaje}],
                output_config={"effort": CONFIG.esfuerzo_llm},
                # Si el modelo rechaza la petición, la API la reintenta con un modelo de respaldo
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream,
        ):
            async for texto in stream.text_stream:
                if uso.ttft_ms is None:
                    uso.ttft_ms = (time.perf_counter() - t0) * 1000
                    tope.reschedule(None)  # ya llegó el primer token: sin tope para el resto
                yield texto
            final = await stream.get_final_message()
        uso.modelo = final.model
        uso.sumar(final.usage, final.model)
        uso.motivo_fin = final.stop_reason or ""
        uso.total_ms = (time.perf_counter() - t0) * 1000


class ClienteGemini:
    """Proveedor de respaldo: Gemini, con la misma interfaz que ClienteClaude."""

    def __init__(self, modelo: str = CONFIG.modelo_respaldo):
        from google import genai

        self.modelo = modelo
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    async def responder(self, system: str, mensaje: str, uso: Uso) -> AsyncIterator[str]:
        from google.genai import types

        t0 = time.perf_counter()
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=CONFIG.max_tokens)
        ultimo = None
        async for parte in await self.client.aio.models.generate_content_stream(
                model=self.modelo, contents=mensaje, config=config):
            ultimo = parte
            if parte.text:
                if uso.ttft_ms is None:
                    uso.ttft_ms = (time.perf_counter() - t0) * 1000
                yield parte.text
        meta = ultimo.usage_metadata if ultimo else None
        uso.modelo = self.modelo
        uso.entrada += (meta.prompt_token_count or 0) if meta else 0
        uso.salida += (meta.candidates_token_count or 0) if meta else 0
        uso.costo_conocido = False  # costos.py no tiene el precio de Gemini: el costo queda desconocido
        uso.motivo_fin = "end_turn"
        uso.total_ms = (time.perf_counter() - t0) * 1000


class ClienteConRespaldo:
    """Router entre proveedores: Claude primero y, si falla antes del primer token, Gemini.

    Cubre lo que el parámetro fallbacks de Anthropic no cubre: un 429, un 5xx, una caída o el tope
    al primer token. Si Claude falla a mitad de la respuesta no cambia de proveedor, porque
    el usuario ya leyó parte del texto.
    """

    def __init__(self, principal: "ClienteClaude", respaldo: ClienteGemini):
        self.principal, self.respaldo = principal, respaldo
        self.modelo = principal.modelo

    async def responder(self, system: str, mensaje: str, uso: Uso) -> AsyncIterator[str]:
        entrego = False
        try:
            async for texto in self.principal.responder(system, mensaje, uso):
                entrego = True
                yield texto
            return
        except (anthropic.APIError, TimeoutError) as e:
            if entrego:
                raise
            uso.respaldo = type(e).__name__
        async for texto in self.respaldo.responder(system, mensaje, uso):
            yield texto


class ClienteSimulado:
    modelo = "simulado"

    async def responder(self, system: str, mensaje: str, uso: Uso) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        uso.ttft_ms = 0.0
        uso.costo_conocido = False  # no hubo llamada al LLM: el costo no es cero, es desconocido
        yield "Modo simulado, sin LLM. Estas son las fuentes que recibiría el modelo: "
        yield " ".join(f"[{i}]" for i in range(1, mensaje.count("<fuente ") + 1))
        uso.motivo_fin = "end_turn"
        uso.total_ms = (time.perf_counter() - t0) * 1000


def crear_cliente() -> ClienteClaude | ClienteConRespaldo | ClienteSimulado:
    if not hay_credenciales():
        return ClienteSimulado()
    if os.environ.get("GEMINI_API_KEY"):
        return ClienteConRespaldo(ClienteClaude(), ClienteGemini())
    return ClienteClaude()
