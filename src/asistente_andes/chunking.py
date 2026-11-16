"""Lectura del corpus y división en fragmentos (chunks).

Tres estrategias para comparar en la evaluación:
- secciones: respeta los encabezados del markdown y antepone el título del
  documento y de la sección a cada fragmento (encabezado contextual).
- fijo-300 y fijo-800: cortan cada N caracteres sin mirar la estructura.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Documento:
    ruta: str  # relativa al corpus, por ejemplo politicas/despachos.md
    titulo: str
    texto: str


@dataclass(frozen=True)
class Chunk:
    id: str  # ruta#número
    ruta: str
    titulo: str
    seccion: str
    texto: str  # lo que se indexa y lo que lee el LLM


def leer_corpus(carpeta: Path) -> list[Documento]:
    docs = []
    for archivo in sorted(carpeta.rglob("*.md")):
        texto = archivo.read_text(encoding="utf-8")
        titulo = re.search(r"^# (.+)$", texto, re.M)
        docs.append(Documento(
            ruta=archivo.relative_to(carpeta).as_posix(),
            titulo=titulo.group(1).strip() if titulo else archivo.stem,
            texto=texto,
        ))
    return docs


def _secciones(texto: str) -> list[tuple[str, str]]:
    """Divide en pares (título de sección, cuerpo) usando los encabezados ##."""
    partes = re.split(r"^## (.+)$", texto, flags=re.M)
    intro = re.sub(r"^# .+$", "", partes[0], flags=re.M).strip()
    secciones = [("Introducción", intro)] if intro else []
    for i in range(1, len(partes), 2):
        secciones.append((partes[i].strip(), partes[i + 1].strip()))
    return secciones


def _partir_por_parrafos(cuerpo: str, maximo: int) -> list[str]:
    """Si una sección es muy larga, la corta entre párrafos, nunca a mitad de frase."""
    bloques, actual = [], ""
    for parrafo in cuerpo.split("\n\n"):
        if actual and len(actual) + len(parrafo) > maximo:
            bloques.append(actual)
            actual = parrafo
        else:
            actual = f"{actual}\n\n{parrafo}" if actual else parrafo
    if actual:
        bloques.append(actual)
    return bloques


def por_secciones(doc: Documento, maximo: int = 1200) -> list[Chunk]:
    chunks = []
    for seccion, cuerpo in _secciones(doc.texto):
        for bloque in _partir_por_parrafos(cuerpo, maximo):
            chunks.append(Chunk(
                id=f"{doc.ruta}#{len(chunks)}",
                ruta=doc.ruta,
                titulo=doc.titulo,
                seccion=seccion,
                texto=f"{doc.titulo} > {seccion}\n\n{bloque}",
            ))
    return chunks


def fijo(doc: Documento, tamano: int, solape: int = 0) -> list[Chunk]:
    texto = re.sub(r"\s+", " ", doc.texto).strip()
    chunks, inicio = [], 0
    while inicio < len(texto):
        chunks.append(Chunk(id=f"{doc.ruta}#{len(chunks)}", ruta=doc.ruta, titulo=doc.titulo,
                            seccion="", texto=texto[inicio:inicio + tamano]))
        inicio += tamano - solape
    return chunks


def dividir(docs: list[Documento], estrategia: str = "secciones", maximo: int = 1200) -> list[Chunk]:
    if estrategia == "secciones":
        return [c for d in docs for c in por_secciones(d, maximo)]
    if estrategia == "fijo-300":
        return [c for d in docs for c in fijo(d, 300)]
    if estrategia == "fijo-800":
        return [c for d in docs for c in fijo(d, 800, solape=120)]
    raise ValueError(f"Estrategia de chunking desconocida: {estrategia}")
