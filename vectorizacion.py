import re
import sys
from collections import Counter

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ARCHIVO = sys.argv[1] if len(sys.argv) > 1 else "libro.txt"

# ------------------------------------------------------------ 1. LIMPIEZA
# (misma lógica que limpieza.py, pero conservando la división por oraciones)
try:
    nlp = spacy.load("es_core_news_sm", disable=["ner"])  # el parser da doc.sents
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm", disable=["ner"])
nlp.max_length = 5_000_000

with open(ARCHIVO, "r", encoding="utf-8") as f:
    texto = f.read()

# Quitar licencia de Project Gutenberg si existe
ini = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", texto, re.I | re.S)
if ini:
    texto = texto[ini.end():]
fin = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", texto, re.I)
if fin:
    texto = texto[:fin.start()]

texto = re.sub(r"\s+", " ", texto).strip()
doc = nlp(texto)

corpus = []  # una "oración lematizada" = un documento del corpus
for oracion in doc.sents:
    lemas = [
        t.lemma_.lower()
        for t in oracion
        if t.is_alpha and not t.is_stop and len(t.text) > 1
    ]
    if len(lemas) >= 3:  # descartamos oraciones casi vacías
        corpus.append(" ".join(lemas))

print(f"Oraciones en el corpus: {len(corpus)}")
print(f"Ejemplo: {corpus[0]}\n")

# ------------------------------------------------------- 2. BAG OF WORDS
bow = CountVectorizer(max_features=5000)
X_bow = bow.fit_transform(corpus)
vocab_bow = bow.get_feature_names_out()
densidad = X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1])

print("=== BAG OF WORDS ===")
print(f"Matriz: {X_bow.shape} (oraciones x términos)")
print(f"Densidad: {densidad:.4%} (el resto son ceros)")
frec_total = X_bow.sum(axis=0).A1
top_bow = pd.DataFrame({"termino": vocab_bow, "frecuencia": frec_total}) \
    .sort_values("frecuencia", ascending=False).head(10)
print(top_bow.to_string(index=False), "\n")

# ----------------------------------------------------------- 3. TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(corpus)
vocab_tfidf = tfidf.get_feature_names_out()

print("=== TF-IDF ===")
print(f"Matriz: {X_tfidf.shape}")
peso_medio = X_tfidf.mean(axis=0).A1
top_tfidf = pd.DataFrame({"termino": vocab_tfidf, "peso_tfidf_medio": peso_medio}) \
    .sort_values("peso_tfidf_medio", ascending=False).head(10)
print(top_tfidf.to_string(index=False))
print("TF-IDF reduce el peso de términos que aparecen en casi todas las "
      "oraciones y resalta los más distintivos.\n")

# ----------------------------------------------------------- 4. N-GRAMAS
ngr = CountVectorizer(ngram_range=(2, 2), min_df=2)
X_ngr = ngr.fit_transform(corpus)
frec_ngr = X_ngr.sum(axis=0).A1
top_ngr = pd.DataFrame({"bigrama": ngr.get_feature_names_out(), "frecuencia": frec_ngr}) \
    .sort_values("frecuencia", ascending=False).head(10)

print("=== BIGRAMAS (n=2) ===")
print(top_ngr.to_string(index=False), "\n")

# ---------------------------------------------------- 5. SIMILITUD COSENO
idx = 0
sims = cosine_similarity(X_tfidf[idx], X_tfidf).ravel()
mas_parecidas = sims.argsort()[::-1][1:4]  # [0] es la oración consigo misma

print("=== SIMILITUD COSENO (TF-IDF) ===")
print(f"Oración base: {corpus[idx]}")
for i in mas_parecidas:
    print(f"  sim={sims[i]:.3f} -> {corpus[i]}")
print()

# ------------------------------------------------- 6. WORD2VEC (opcional)
try:
    from gensim.models import Word2Vec

    sentencias = [o.split() for o in corpus]
    w2v = Word2Vec(sentencias, vector_size=100, window=5, min_count=2, workers=1, seed=42)

    palabra = Counter(w for s in sentencias for w in s if w in w2v.wv).most_common(1)[0][0]
    print("=== WORD2VEC ===")
    print(f"Vocabulario: {len(w2v.wv)} palabras, vectores de 100 dimensiones")
    print(f"Palabras más cercanas a '{palabra}':")
    for p, s in w2v.wv.most_similar(palabra, topn=5):
        print(f"  {p}: {s:.3f}")
except ImportError:
    print("=== WORD2VEC === omitido (instala gensim para habilitarlo)")

# --------------------------------------------------------- 7. RESULTADOS
top_bow.to_csv("top_terminos_bow.csv", index=False)
top_tfidf.to_csv("top_terminos_tfidf.csv", index=False)
top_ngr.to_csv("top_bigramas.csv", index=False)
print("\nCSV guardados: top_terminos_bow.csv, top_terminos_tfidf.csv, top_bigramas.csv")
