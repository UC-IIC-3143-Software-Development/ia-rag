"""API del asistente. Read path de RAG: guardrails, caché, retrieval, LLM con streaming y trazas.

Correr: uv run uvicorn asistente_andes.app:app --port 8000
"""

import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import indice as idx
from .agente import Agente
from .busqueda import Buscador
from .cache import CacheRespuestas
from .config import CONFIG
from .guardrails import parece_inyeccion, validar_citas
from .llm import Uso, crear_cliente, hay_credenciales
from .prompts import RESPUESTA_ERROR, RESPUESTA_RECHAZO, RESPUESTA_SIN_RESPALDO, SYSTEM, mensaje_usuario
from .trazas import Traza, resumen

ESTATICOS = Path(__file__).parent / "estaticos"


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # Se carga una vez al partir: índice en memoria, modelos de embeddings y reranker
    indice = idx.cargar()
    app.state.buscador = Buscador(indice)
    app.state.llm = crear_cliente()
    # La clave del caché cambia si cambia el índice, el prompt, el modelo o el umbral
    version = f'{indice.meta["version"]}|{CONFIG.modelo_llm}|{CONFIG.umbral_reranker}|{SYSTEM}'
    app.state.cache = CacheRespuestas(version_indice=hashlib.sha256(version.encode()).hexdigest()[:12])
    app.state.agente = Agente(app.state.buscador) if hay_credenciales() else None
    yield


app = FastAPI(title="Asistente Tienda Andes", lifespan=ciclo_de_vida)


class Pregunta(BaseModel):
    pregunta: str = Field(min_length=3, max_length=CONFIG.pregunta_max_caracteres)


def sse(evento: str, datos) -> str:
    """Formato Server-Sent Events: una línea event, una línea data y una línea en blanco."""
    return f"event: {evento}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"


@app.post("/preguntar")
async def preguntar(p: Pregunta, request: Request):
    estado = request.app.state
    traza = Traza(endpoint="/preguntar", pregunta=p.pregunta)
    t0 = time.perf_counter()

    def cerrar() -> None:
        traza.ms["total"] = (time.perf_counter() - t0) * 1000
        traza.guardar()

    async def eventos():
        # 1. Caché exacto: si la misma pregunta ya se respondió con este índice y este prompt, no se paga de nuevo
        guardado = estado.cache.obtener(p.pregunta)
        if guardado:
            traza.cache = True
            yield sse("fuentes", guardado["fuentes"])
            yield sse("texto", guardado["respuesta"])
            cerrar()
            yield sse("fin", {"traza": traza.id, "cache": True})
            return

        # 2. Retrieval híbrido con reranking. Es trabajo de CPU: corre en un hilo para no bloquear el event loop
        busqueda = await asyncio.to_thread(estado.buscador.buscar, p.pregunta)
        traza.ms.update(busqueda.ms)
        resultados = busqueda.resultados
        traza.puntaje_max = round(resultados[0].puntaje, 3) if resultados else None
        fuentes = [{"n": i, "documento": r.chunk.titulo, "seccion": r.chunk.seccion, "id": r.chunk.id}
                   for i, r in enumerate(resultados, start=1)]

        # 3. Sin respaldo suficiente, no se llama al LLM: se ahorra el costo y se evita inventar
        if not resultados or resultados[0].puntaje < CONFIG.umbral_reranker:
            traza.sin_respaldo = True
            yield sse("fuentes", [])
            yield sse("texto", RESPUESTA_SIN_RESPALDO)
            cerrar()
            yield sse("fin", {"traza": traza.id, "sin_respaldo": True})
            return

        # 4. Generación con streaming: el usuario ve el primer token sin esperar la respuesta completa
        traza.fuentes = [r.chunk.id for r in resultados]
        yield sse("fuentes", fuentes)
        uso = Uso(modelo=estado.llm.modelo)
        partes = []
        try:
            async for texto in estado.llm.responder(SYSTEM, mensaje_usuario(p.pregunta, resultados), uso):
                partes.append(texto)
                yield sse("texto", texto)
        except Exception as e:
            # El proveedor falló, no respondió a tiempo o se cortó la red a mitad del stream:
            # se avisa al usuario y la falla queda en la traza
            traza.motivo_fin, traza.error = "error", type(e).__name__
            cerrar()
            yield sse("error", {"traza": traza.id, "mensaje": RESPUESTA_ERROR})
            return
        if uso.motivo_fin == "refusal":
            # Un rechazo a mitad del stream: se descarta lo que alcanzó a llegar y se reemplaza
            partes = [RESPUESTA_RECHAZO]
            yield sse("reemplazar", RESPUESTA_RECHAZO)
        respuesta = "".join(partes)

        # 5. Guardrail de salida y traza
        citas = validar_citas(respuesta, len(resultados))
        traza.citas_invalidas, traza.sin_citas = citas["invalidas"], citas["sin_citas"]
        traza.modelo, traza.motivo_fin, traza.respaldo = uso.modelo, uso.motivo_fin, uso.respaldo
        traza.tokens_entrada, traza.tokens_salida = uso.entrada, uso.salida
        traza.tokens_cache_lectura, traza.costo_usd = uso.cache_lectura, uso.costo_usd
        traza.ms["llm_primer_token"] = uso.ttft_ms or 0.0
        traza.ms["llm_total"] = uso.total_ms
        cerrar()
        # Solo se guarda en caché una respuesta completa que cita fuentes y todas existen
        if uso.motivo_fin == "end_turn" and not citas["invalidas"] and not citas["sin_citas"]:
            estado.cache.guardar(p.pregunta, {"fuentes": fuentes, "respuesta": respuesta})
        yield sse("fin", {"traza": traza.id, "tokens": {"entrada": uso.entrada, "salida": uso.salida},
                          "costo_usd": uso.costo_usd, "citas": citas,
                          "inyeccion_en_pregunta": parece_inyeccion(p.pregunta)})

    return StreamingResponse(eventos(), media_type="text/event-stream")


class Mensaje(BaseModel):
    mensaje: str = Field(min_length=3, max_length=CONFIG.pregunta_max_caracteres)


@app.post("/agente")
async def agente(m: Mensaje, request: Request, x_cliente: str = Header(description="Cliente autenticado")):
    # En producción, el cliente sale del token de sesión, no de un header que manda el navegador
    ag = request.app.state.agente
    if ag is None:
        raise HTTPException(503, "El agente necesita ANTHROPIC_API_KEY")
    traza = Traza(endpoint="/agente", pregunta=m.mensaje)
    try:
        resultado = await ag.responder(m.mensaje, x_cliente)
    except Exception as e:
        traza.motivo_fin, traza.error = "error", type(e).__name__
        traza.guardar()
        raise HTTPException(503, RESPUESTA_ERROR) from e
    uso = resultado["uso"]
    traza.herramientas = resultado["pasos"]
    traza.modelo, traza.motivo_fin = uso.modelo, uso.motivo_fin
    traza.tokens_entrada, traza.tokens_salida = uso.entrada, uso.salida
    traza.tokens_cache_lectura, traza.costo_usd = uso.cache_lectura, uso.costo_usd
    traza.ms["total"] = uso.total_ms
    traza.guardar()
    return {"respuesta": resultado["respuesta"], "pasos": resultado["pasos"], "traza": traza.id}


@app.get("/metricas")
def metricas():
    return resumen()


@app.get("/salud")
def salud(request: Request):
    meta = request.app.state.buscador.indice.meta
    return {"ok": True, "llm": request.app.state.llm.modelo, "chunks": meta["chunks"],
            "embeddings": meta["modelo_embeddings"], "cuarentena": meta["cuarentena"]}


@app.get("/")
def inicio():
    return FileResponse(ESTATICOS / "index.html")
