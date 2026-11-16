"""Configuración leída de variables de entorno, con valores por defecto para correr en local."""

import os
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def _env(nombre: str, defecto: str) -> str:
    return os.environ.get(nombre, defecto)


@dataclass(frozen=True)
class Config:
    corpus: Path = RAIZ / "corpus"
    indice: Path = Path(_env("INDICE", str(RAIZ / "datos" / "indice.db")))
    pedidos: Path = RAIZ / "datos" / "pedidos.json"
    # Simula la API de inventario: cambia todo el día, por eso se consulta directo y no se indexa
    stock: Path = RAIZ / "datos" / "stock.json"
    trazas: Path = Path(_env("TRAZAS", str(RAIZ / "datos" / "trazas.jsonl")))
    modelos: Path = Path(_env("FASTEMBED_CACHE_PATH", str(RAIZ / ".cache" / "modelos")))
    # Embeddings multilingües que corren en CPU con ONNX, sin API externa
    modelo_embeddings: str = _env("MODELO_EMBEDDINGS", "intfloat/multilingual-e5-large")
    # Cross-encoder multilingüe que reordena los candidatos (versión cuantizada a 8 bits)
    modelo_reranker: str = _env("MODELO_RERANKER", "BAAI/bge-reranker-v2-m3-int8")
    estrategia_chunking: str = _env("CHUNKING", "secciones")
    chunk_max_caracteres: int = int(_env("CHUNK_MAX", "1200"))
    candidatos: int = int(_env("CANDIDATOS", "10"))  # cuántos trae cada buscador y cuántos se reordenan
    k: int = 5  # cuántos fragmentos llegan al prompt
    umbral_reranker: float = float(_env("UMBRAL_RERANKER", "-5.0"))  # bajo esto, no hay respaldo
    pregunta_max_caracteres: int = 500
    modelo_llm: str = _env("MODELO_LLM", "claude-opus-5-5")
    # Proveedor de respaldo si Claude falla antes del primer token: necesita GEMINI_API_KEY
    modelo_respaldo: str = _env("MODELO_RESPALDO", "gemini-3.5-flash")
    esfuerzo_llm: str = _env("ESFUERZO_LLM", "low")  # low, medium, high, xhigh o max
    primer_token_max_s: float = float(_env("PRIMER_TOKEN_MAX_S", "8"))  # si no llega, se falla rápido
    # El juez puede ser otro modelo para evitar que un modelo se evalúe a sí mismo
    # El juez es otro modelo que el que responde: un modelo que se evalúa a sí mismo tiende a aprobarse
    modelo_juez: str = _env("MODELO_JUEZ", "claude-sonnet-5-5")
    max_tokens: int = 4000


CONFIG = Config()
