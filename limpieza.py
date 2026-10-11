
import re
import sys
from collections import Counter

import spacy

ARCHIVO = sys.argv[1] if len(sys.argv) > 1 else "libro.txt"
SALIDA = "libro_limpio.txt"

# ---------------------------------------------------------------- carga
try:
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

nlp.max_length = 5_000_000  # libros largos superan el límite por defecto

with open(ARCHIVO, "r", encoding="utf-8") as f:
    texto = f.read()
print(f"Texto cargado: {len(texto):,} caracteres")


# ---------------------------------------------------------- limpieza base
def quitar_boilerplate_gutenberg(t):
    """Elimina encabezado y pie de licencia de Project Gutenberg, si existen."""
    inicio = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", t, re.I | re.S)
    fin = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t, re.I)
    if inicio:
        t = t[inicio.end():]
    if fin:
        t = t[:fin.start()] if not inicio else t[:re.search(
            r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t, re.I).start()]
    return t


def normalizar(t):
    """Normalización: minúsculas, sin números ni símbolos raros, espacios limpios."""
    t = t.lower()
    t = re.sub(r"\d+", " ", t)                      # números
    t = re.sub(r"[^a-záéíóúüñ\s]", " ", t)          # solo letras (con acentos y ñ)
    t = re.sub(r"\s+", " ", t).strip()              # espacios múltiples
    return t


texto = quitar_boilerplate_gutenberg(texto)
texto_norm = normalizar(texto)
print(f"Después de normalizar: {len(texto_norm):,} caracteres")

# ---------------------------------------------------- lematización
doc = nlp(texto_norm)

tokens_totales = [t for t in doc if not t.is_space]
lemas = [
    t.lemma_.lower()
    for t in doc
    if not t.is_stop and not t.is_punct and not t.is_space and len(t.text) > 1
]

print(f"Tokens totales:        {len(tokens_totales):,}")
print(f"Lemas tras filtrado:   {len(lemas):,}")
print(f"Vocabulario único:     {len(set(lemas)):,}")

print("\nEjemplos (original -> lema):")
vistos = set()
for t in doc:
    if not t.is_stop and t.text != t.lemma_.lower() and t.text not in vistos:
        vistos.add(t.text)
        print(f"  {t.text} -> {t.lemma_.lower()}")
    if len(vistos) >= 10:
        break

print("\n15 lemas más frecuentes:")
for lema, n in Counter(lemas).most_common(15):
    print(f"  {lema}: {n}")

with open(SALIDA, "w", encoding="utf-8") as f:
    f.write(" ".join(lemas))
print(f"\nTexto limpio guardado en {SALIDA}")
