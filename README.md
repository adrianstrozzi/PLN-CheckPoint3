# Limpieza de texto de un libro (PLN)

Proyecto de Procesamiento de Lenguaje Natural: normalización y lematización
de un libro en español usando spaCy.

## Contenido

- `limpieza.py`: script con la normalización y lematización.
- `libro.txt`: libro a procesar.
- `requirements.txt`: dependencias congeladas.

## Cómo ejecutarlo

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python limpieza.py
```

Genera `libro_limpio.txt` y muestra estadísticas en consola.

## Proceso de limpieza

1. Eliminación del encabezado/pie de licencia de Project Gutenberg.
2. Normalización: minúsculas, sin números ni símbolos, espacios unificados.
3. Tokenización con spaCy.
4. Filtrado de stop words y puntuación.
5. Lematización (por ejemplo, `fui` -> `ir`).
