from asistente_andes.busqueda import consulta_fts, rrf
from asistente_andes.costos import costo_usd


def test_rrf_premia_lo_que_aparece_en_ambas_listas():
    bm25 = [1, 2, 3]
    vectorial = [3, 4, 1]
    orden = [n for n, _ in rrf([bm25, vectorial])]
    assert orden[:2] == [1, 3]  # están en las dos listas
    assert set(orden) == {1, 2, 3, 4}


def test_consulta_fts_quita_tildes_y_palabras_vacias():
    assert consulta_fts("¿Cuánto cuesta el despacho a Punta Arenas?") == '"cuesta" OR "despacho" OR "punta" OR "arenas"'
    assert consulta_fts("¿y?") == ""


def test_costo_por_peticion():
    # 3.000 tokens de entrada y 200 de salida con Claude Opus 5.5 (4 y 20 dólares por millón)
    assert costo_usd("claude-opus-5-5", 3000, 200) == 0.016
    assert costo_usd("modelo-desconocido", 1, 1) is None
