"""Embedding helper using sentence-transformers.

Default model: BAAI/bge-base-en-v1.5 (768-d, good quality/speed trade-off).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

# Module-level cache so the model is loaded only once per process.
_model_cache: dict[str, object] = {}


def _get_model(model_name: str = DEFAULT_MODEL):
    """Return a cached SentenceTransformer instance."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s' …", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Model loaded.")
    return _model_cache[model_name]


def build_embed_text(record: dict) -> str:
    """Build the dense text representation for a single course record.

    Format: "[course_code] [title] – [department]. [description].
             Prerequisites: [prerequisites]."
    """
    code = record.get("course_code", "")
    title = record.get("title", "")
    dept = record.get("department", "")
    desc = record.get("description", "")
    prereq = record.get("prerequisites") or ""

    parts = [f"{code} {title} – {dept}.", desc + "."]
    if prereq:
        parts.append(f"Prerequisites: {prereq}.")
    return " ".join(parts)


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed a list of strings and return an (N, dim) float32 array.

    Vectors are L2-normalised so that inner-product == cosine similarity.
    """
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-norm → IP == cosine
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(
    query: str, model_name: str = DEFAULT_MODEL
) -> np.ndarray:
    """Embed a single query string and return a (1, dim) float32 array."""
    return embed_texts([query], model_name=model_name, show_progress=False)
