"""Caché exacto de respuestas: misma pregunta normalizada y misma versión del índice.

No es un caché semántico a propósito: dos preguntas muy parecidas pueden tener
respuestas distintas (despacho a Arica o a Punta Arenas). Ver evals/similitud_cache.py.
"""

import hashlib
import time
from collections import OrderedDict

from .busqueda import _normalizar


class CacheRespuestas:
    def __init__(self, version_indice: str, ttl_segundos: int = 3600, maximo: int = 1000):
        self.version = version_indice
        self.ttl = ttl_segundos
        self.maximo = maximo
        self._datos: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def _clave(self, pregunta: str) -> str:
        normal = " ".join(_normalizar(pregunta).replace("?", "").replace("¿", "").split())
        return hashlib.sha256(f"{self.version}|{normal}".encode()).hexdigest()

    def obtener(self, pregunta: str) -> dict | None:
        clave = self._clave(pregunta)
        item = self._datos.get(clave)
        if not item or time.time() - item[0] > self.ttl:
            return None
        self._datos.move_to_end(clave)
        return item[1]

    def guardar(self, pregunta: str, valor: dict) -> None:
        self._datos[self._clave(pregunta)] = (time.time(), valor)
        if len(self._datos) > self.maximo:
            self._datos.popitem(last=False)  # saca el menos usado
