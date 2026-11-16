import asyncio
from datetime import date
from types import SimpleNamespace as NS

from asistente_andes.agente import DERIVAR, MAX_PASOS, Agente

from .conftest import BuscadorFalso


class LLMFalso:
    """Simula a Claude: primero pide consultar el pedido, después responde."""

    def __init__(self, numero: str):
        self.numero = numero
        self.peticiones = []
        self.beta = NS(messages=NS(create=self.create))

    async def create(self, **kwargs):
        self.peticiones.append(kwargs)
        uso = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        if len(self.peticiones) == 1:
            llamada = NS(type="tool_use", id="t1", name="consultar_pedido", input={"numero": self.numero})
            return NS(content=[llamada], stop_reason="tool_use", usage=uso, model="claude-opus-5-5")
        return NS(content=[NS(type="text", text="Listo.")], stop_reason="end_turn", usage=uso, model="claude-opus-5-5")


def correr(numero: str, cliente_id: str):
    llm = LLMFalso(numero)
    agente = Agente(BuscadorFalso(), cliente_llm=llm)
    resultado = asyncio.run(agente.responder("¿Dónde está mi pedido?", cliente_id, hoy=date(2026, 11, 15)))
    return resultado, llm.peticiones[1]["messages"][-1]["content"][0]["content"]


def test_loop_ejecuta_la_herramienta_y_devuelve_el_resultado():
    resultado, salida = correr("A-10231", "cli-001")
    assert resultado["respuesta"] == "Listo."
    # dos llamadas de 100 tokens de entrada y 20 de salida con Opus 5.5
    assert resultado["uso"].costo_usd == 0.0016
    assert resultado["pasos"][0]["herramienta"] == "consultar_pedido"
    assert '"estado": "entregado"' in salida and "cli-001" not in salida


def test_un_cliente_no_ve_pedidos_de_otro():
    _, salida = correr("A-10231", "cli-002")
    assert salida == "No existe ese pedido para este cliente."


class LLMQueNoTermina(LLMFalso):
    """Simula un modelo que pide herramientas en cada turno."""

    async def create(self, **kwargs):
        self.peticiones.append(kwargs)
        uso = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        llamada = NS(type="tool_use", id=f"t{len(self.peticiones)}", name="consultar_pedido", input={"numero": self.numero})
        return NS(content=[NS(type="text", text="Voy a revisar"), llamada], stop_reason="tool_use", usage=uso, model="claude-opus-5-5")


def test_al_llegar_al_limite_deriva_a_una_persona():
    llm = LLMQueNoTermina("A-10231")
    agente = Agente(BuscadorFalso(), cliente_llm=llm)
    resultado = asyncio.run(agente.responder("¿Dónde está mi pedido?", "cli-001", hoy=date(2026, 11, 15)))
    assert len(llm.peticiones) == MAX_PASOS
    assert resultado["uso"].motivo_fin == "max_pasos"
    assert resultado["respuesta"] == DERIVAR


def test_stock_por_tienda_viene_del_inventario():
    agente = Agente(BuscadorFalso(), cliente_llm=LLMFalso("A-10231"))
    salida = agente.ejecutar("consultar_stock", {"producto": "parka cerro plomo", "tienda": "Concepción"}, "cli-001")
    assert '"unidades": 0' in salida and "Concepción" in salida
    assert agente.ejecutar("consultar_stock", {"producto": "bicicleta", "tienda": ""}, "cli-001").startswith("No encontré")
