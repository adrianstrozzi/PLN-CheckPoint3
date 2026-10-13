import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # genera PNG sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE

ARCHIVO = sys.argv[1] if len(sys.argv) > 1 else "libro.txt"

# =========================================================== 1. CARGA
print("1. CARGA DE DATOS")
with open(ARCHIVO, "r", encoding="utf-8") as f:
    texto = f.read()
print(f"   {len(texto):,} caracteres leídos de {ARCHIVO}")

# ========================================================== 2. LIMPIEZA
print("2. LIMPIEZA")
try:
    nlp = spacy.load("es_core_news_sm", disable=["ner"])  # el parser da doc.sents
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm", disable=["ner"])
nlp.max_length = 5_000_000

# 2.1 Quitar licencia de Project Gutenberg (si existe)
ini = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", texto, re.I | re.S)
if ini:
    texto = texto[ini.end():]
fin = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", texto, re.I)
if fin:
    texto = texto[:fin.start()]

# 2.2 Normalización de espacios (la minúscula se aplica sobre los lemas)
texto = re.sub(r"\s+", " ", texto).strip()

# 2.3 Tokenización, filtrado de stop words/puntuación y lematización
doc = nlp(texto)
corpus = []
for oracion in doc.sents:
    lemas = [
        t.lemma_.lower()
        for t in oracion
        if t.is_alpha and not t.is_stop and len(t.text) > 1
    ]
    if len(lemas) >= 3:  # descartamos oraciones casi vacías
        corpus.append(" ".join(lemas))

with open("libro_procesado.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus))
print(f"   {len(corpus)} oraciones procesadas -> libro_procesado.txt")

# ============================================ 3. VECTORIZACIÓN CLÁSICA
print("3. VECTORIZACIÓN CLÁSICA")
bow = CountVectorizer(max_features=5000)
X_bow = bow.fit_transform(corpus)
print(f"   Bag of Words: matriz {X_bow.shape}")

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(corpus)
print(f"   TF-IDF:       matriz {X_tfidf.shape}")

# ===================================== 4. VECTORIZACIÓN DISTRIBUCIONAL
print("4. VECTORIZACIÓN DISTRIBUCIONAL (Word2Vec)")
sentencias = [o.split() for o in corpus]
w2v = Word2Vec(sentencias, vector_size=100, window=5, min_count=2,
               sg=1, epochs=50, workers=1, seed=42)
print(f"   Vocabulario: {len(w2v.wv)} palabras x 100 dimensiones")

frecuentes = [p for p, _ in Counter(w for s in sentencias for w in s).most_common(200)
              if p in w2v.wv]
for p in frecuentes[:3]:
    cercanas = ", ".join(f"{w} ({s:.2f})" for w, s in w2v.wv.most_similar(p, topn=4))
    print(f"   Cercanas a '{p}': {cercanas}")

# ============================================= 5. REPRESENTACIÓN GRÁFICA
print("5. GRÁFICAS")

# 5.1 Espacio TF-IDF: cada punto es una oración, color = posición en el libro
coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(X_tfidf)
plt.figure(figsize=(10, 7))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(corpus)),
                 cmap="viridis", s=12, alpha=0.7)
plt.colorbar(sc, label="Posición de la oración en el libro")
plt.title("Espacio vectorial TF-IDF (oraciones, SVD 2D)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.tight_layout()
plt.savefig("img_tfidf_oraciones.png", dpi=150)
plt.close()


def graficar_palabras(coords2d, palabras, titulo, archivo):
    plt.figure(figsize=(12, 9))
    plt.scatter(coords2d[:, 0], coords2d[:, 1], s=25, alpha=0.6)
    for (x, y), palabra in zip(coords2d, palabras):
        plt.annotate(palabra, (x, y), fontsize=8, alpha=0.85,
                     xytext=(3, 3), textcoords="offset points")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


# 5.2 Espacio Word2Vec con PCA (palabras más frecuentes)
palabras = frecuentes[:80]
vectores = np.array([w2v.wv[p] for p in palabras])
graficar_palabras(PCA(n_components=2, random_state=42).fit_transform(vectores),
                  palabras, "Espacio vectorial Word2Vec (PCA 2D)",
                  "img_word2vec_pca.png")

# 5.3 Espacio Word2Vec con t-SNE (más palabras, agrupa por contexto)
palabras = frecuentes[:150]
vectores = np.array([w2v.wv[p] for p in palabras])
perplexity = max(2, min(30, len(palabras) - 1))
graficar_palabras(
    TSNE(n_components=2, perplexity=perplexity, random_state=42,
         init="pca").fit_transform(vectores),
    palabras, "Espacio vectorial Word2Vec (t-SNE 2D)", "img_word2vec_tsne.png")

print("   Guardadas: img_tfidf_oraciones.png, img_word2vec_pca.png, img_word2vec_tsne.png")
print("\nListo.")
