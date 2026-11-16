"""BM25 paso a paso: reproduce a mano el puntaje bm25() de SQLite FTS5 para una pregunta.

Muestra las palabras que quedan, en cuántos fragmentos está cada una, su IDF y el peso
de cada palabra en los 10 primeros fragmentos. Los números deben calzar con FTS5.

Uso: uv run python evals/bm25_paso_a_paso.py "¿Puedo devolver un cartucho de gas que no usé?"
"""
import math
import re
import sqlite3
import sys
import unicodedata

from asistente_andes.config import CONFIG

db = sqlite3.connect(f"file:{CONFIG.indice}?mode=ro", uri=True)
STOP = set("""
a al algo como con cual cuales cuando cuanto cuanta cuantos cuantas de del donde e el ella en es esta
este esto hay la las le les lo los me mi mis no o para pero por que quien se si sin su sus te tengo
tiene tienen un una uno unos unas y ya yo puedo puede hacer hago
""".split())
def norm(t):
    s = unicodedata.normalize("NFD", t.lower()); return "".join(c for c in s if unicodedata.category(c) != "Mn")
def toks(t):  # aprox unicode61: separa en no alfanumericos
    return re.findall(r"[0-9a-z]+", norm(t))
preg = sys.argv[1]
terms = list(dict.fromkeys([t for t in re.findall(r"[\w-]+", norm(preg)) if t not in STOP and len(t) > 1]))
q = " OR ".join(f'"{t}"' for t in terms)
print("consulta FTS:", q)
rows = db.execute("SELECT n, ruta, seccion, texto FROM chunks ORDER BY n").fetchall()
N = len(rows)
docs = {n: toks(tx) for n, _, _, tx in rows}
avgdl = sum(len(v) for v in docs.values()) / N
print("N =", N, "avgdl =", round(avgdl, 1))
k1, b = 1.2, 0.75
df = {t: sum(1 for v in docs.values() if t in v) for t in terms}
idf = {}
for t in terms:
    x = math.log((N - df[t] + 0.5) / (df[t] + 0.5))
    idf[t] = x if x > 0 else 1e-6
    print(f"  {t}: n(t)={df[t]}  IDF={idf[t]:.3f}  (Lucene log(1+..)={math.log(1+(N-df[t]+0.5)/(df[t]+0.5)):.3f})")
fts = db.execute("SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT 10", (q,)).fetchall()
meta = {n: (r, s) for n, r, s, _ in rows}
for pos, (n, sc) in enumerate(fts, 1):
    d = docs[n]; dl = len(d); parts = []; tot = 0
    for t in terms:
        f = d.count(t)
        if f:
            w = idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
            parts.append(f"{t}: f={f} -> {w:.2f}"); tot += w
    print(f"{pos}. fts={-sc:.3f} mio={tot:.3f} |d|={dl} {meta[n][0]} > {meta[n][1]} :: {'; '.join(parts)}")
