import json

from fastapi.testclient import TestClient

from asistente_andes.app import app
from asistente_andes.cache import CacheRespuestas
from asistente_andes.llm import ClienteSimulado, Uso

from .conftest import BuscadorFalso


def eventos(respuesta) -> list[tuple[str, object]]:
    salida = []
    for bloque in respuesta.text.strip().split("\n\n"):
        evento, datos = bloque.split("\n")
        salida.append((evento.removeprefix("event: "), json.loads(datos.removeprefix("data: "))))
    return salida


class LLMSinCitas:
    modelo = "falso"

    async def responder(self, system, mensaje, uso):
        uso.motivo_fin = "end_turn"
        yield "La parka se puede devolver."


class LLMCaido:
    """Se corta la red a mitad del stream, después del primer fragmento."""
    modelo = "falso"

    async def responder(self, system, mensaje, uso):
        yield "La parka"
        raise ConnectionResetError()


def cliente(puntaje: float = 2.0, llm=None) -> TestClient:
    # Sin el bloque with, TestClient no corre el lifespan: no se cargan los modelos reales
    app.state.buscador = BuscadorFalso(puntaje)
    app.state.llm = llm or ClienteSimulado()
    app.state.cache = CacheRespuestas(version_indice="test")
    app.state.agente = None
    return TestClient(app)


def test_preguntar_entrega_fuentes_texto_y_fin_en_orden():
    r = cliente().post("/preguntar", json={"pregunta": "¿Puedo devolver una parka?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = eventos(r)
    assert ev[0][0] == "fuentes" and len(ev[0][1]) == 2
    assert ev[-1][0] == "fin" and ev[-1][1]["citas"]["invalidas"] == []
    assert "[1] [2]" in "".join(d for e, d in ev if e == "texto")


def test_sin_respaldo_no_llama_al_llm():
    r = cliente(puntaje=-9.0).post("/preguntar", json={"pregunta": "¿Venden bicicletas?"})
    ev = eventos(r)
    assert ev[0] == ("fuentes", [])
    assert "No encontré" in ev[1][1]
    assert ev[-1][1]["sin_respaldo"] is True


def test_segunda_pregunta_igual_sale_del_cache():
    c = cliente()
    c.post("/preguntar", json={"pregunta": "¿Puedo devolver una parka?"})
    r = c.post("/preguntar", json={"pregunta": "¿puedo devolver una PARKA"})
    assert eventos(r)[-1][1]["cache"] is True
    assert app.state.buscador.llamadas == 1


def test_valida_largo_de_la_pregunta():
    assert cliente().post("/preguntar", json={"pregunta": "x" * 600}).status_code == 422


def test_respuesta_sin_citas_no_entra_al_cache():
    c = cliente(llm=LLMSinCitas())
    ev = eventos(c.post("/preguntar", json={"pregunta": "¿Puedo devolver una parka?"}))
    assert ev[-1][1]["citas"]["sin_citas"] is True
    c.post("/preguntar", json={"pregunta": "¿Puedo devolver una parka?"})
    assert app.state.buscador.llamadas == 2


def test_falla_del_llm_avisa_con_un_evento_de_error(trazas_temporales):
    ev = eventos(cliente(llm=LLMCaido()).post("/preguntar", json={"pregunta": "¿Puedo devolver una parka?"}))
    assert ev[-1][0] == "error"
    assert '"motivo_fin": "error"' in trazas_temporales.read_text()


def test_costo_suma_cada_intento_con_su_modelo():
    from types import SimpleNamespace as NS
    intento = lambda m, i, o: NS(type="message" if m == "claude-opus-5-5" else "fallback_message", model=m, input_tokens=i,
                                 output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    usage = NS(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0,
               iterations=[intento("claude-opus-5-5", 1000, 10), intento("claude-opus-5", 1000, 100)])
    uso = Uso(modelo="claude-opus-5")
    uso.sumar(usage, "claude-opus-5")
    # Opus 5.5: 1.000 × 4 + 10 × 20 = 4.200 µUS$. Opus 5, el respaldo: 1.000 × 5 + 100 × 25 = 7.500 µUS$
    assert uso.costo_usd == 0.0117 and uso.entrada == 2000
