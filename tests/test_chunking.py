from asistente_andes.chunking import Documento, dividir, fijo, por_secciones

DOC = Documento(
    ruta="politicas/demo.md",
    titulo="Despachos",
    texto="# Despachos\n\nIntro breve.\n\n## Plazos\n\nUno.\n\nDos.\n\n## Costos\n\nTres.",
)


def test_secciones_antepone_documento_y_seccion():
    chunks = por_secciones(DOC)
    assert [c.seccion for c in chunks] == ["Introducción", "Plazos", "Costos"]
    assert chunks[1].texto.startswith("Despachos > Plazos\n\n")
    assert chunks[1].id == "politicas/demo.md#1"


def test_seccion_larga_se_corta_entre_parrafos():
    largo = Documento("a.md", "A", "# A\n\n## S\n\n" + "\n\n".join(["x" * 50] * 6))
    chunks = por_secciones(largo, maximo=120)
    assert len(chunks) == 3
    assert all("x" * 50 in c.texto for c in chunks)


def test_fijo_respeta_tamano_y_solape():
    chunks = fijo(DOC, tamano=20, solape=5)
    assert all(len(c.texto) <= 20 for c in chunks)
    assert chunks[0].texto[-5:] == chunks[1].texto[:5]


def test_estrategia_desconocida_falla():
    try:
        dividir([DOC], "otra")
    except ValueError:
        return
    raise AssertionError("debió fallar")
