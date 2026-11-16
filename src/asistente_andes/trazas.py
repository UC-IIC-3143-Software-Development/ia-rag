"""Observabilidad: una traza por petición en un archivo JSONL y un resumen con percentiles.

En producción, estas trazas van a una herramienta de observabilidad (OpenTelemetry,
Langfuse, Datadog). El contenido es el mismo: latencia por etapa, tokens, costo y fuentes.
"""

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CONFIG
from .guardrails import ocultar_pii


@dataclass
class Traza:
    endpoint: str
    pregunta: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    ms: dict = field(default_factory=dict)  # latencia por etapa
    fuentes: list = field(default_factory=list)  # ids de los chunks enviados al LLM
    puntaje_max: float | None = None
    sin_respaldo: bool = False
    cache: bool = False
    modelo: str = ""
    respaldo: str = ""  # por qué respondió el proveedor de respaldo, si respondió
    tokens_entrada: int = 0
    tokens_salida: int = 0
    tokens_cache_lectura: int = 0
    costo_usd: float | None = None
    motivo_fin: str = ""
    error: str = ""
    citas_invalidas: list = field(default_factory=list)
    sin_citas: bool = False  # la respuesta llegó al usuario, pero sin citar ninguna fuente
    herramientas: list = field(default_factory=list)

    def guardar(self, ruta: Path = CONFIG.trazas) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fila = asdict(self)
        # Nunca guardar datos personales en claro, ni en la pregunta ni en lo que el agente pasó a una herramienta
        fila["pregunta"] = ocultar_pii(self.pregunta)
        fila["herramientas"] = json.loads(ocultar_pii(json.dumps(self.herramientas, ensure_ascii=False)))
        fila["ms"] = {k: round(v, 1) for k, v in self.ms.items()}
        with ruta.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def percentil(valores: list[float], p: float) -> float | None:
    """Percentil por rango más cercano: siempre devuelve un valor observado."""
    if not valores:
        return None
    orden = sorted(valores)
    return orden[max(0, math.ceil(p * len(orden)) - 1)]


def resumen(ruta: Path = CONFIG.trazas) -> dict:
    if not ruta.exists():
        return {"peticiones": 0}
    filas = [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]
    etapas = sorted({k for f in filas for k in f["ms"]})
    costos = [f["costo_usd"] for f in filas if f.get("costo_usd") is not None]
    return {
        "peticiones": len(filas),
        "sin_respaldo": round(sum(f["sin_respaldo"] for f in filas) / len(filas), 3),
        "aciertos_cache": round(sum(f["cache"] for f in filas) / len(filas), 3),
        "latencia_ms": {
            e: {"p50": percentil([f["ms"][e] for f in filas if e in f["ms"]], 0.5),
                "p95": percentil([f["ms"][e] for f in filas if e in f["ms"]], 0.95)}
            for e in etapas
        },
        "tokens_promedio": {
            "entrada": round(sum(f["tokens_entrada"] for f in filas) / len(filas)),
            "salida": round(sum(f["tokens_salida"] for f in filas) / len(filas)),
        },
        "costo_total_usd": round(sum(costos), 4) if costos else None,
    }
