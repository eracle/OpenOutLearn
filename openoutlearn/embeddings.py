# openoutlearn/embeddings.py
"""Fastembed text embedding utilities, shared by every ``GPBaldQualifier`` consumer.

Extracted from OpenOutFind's ``core/ml/embeddings.py`` (OpenOutNews had already
copied it forward verbatim, dropping only the Django ``settings`` coupling). The
model is loaded lazily and cached per process; caller supplies the model name and
cache directory, since each domain owns its own config for both.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Texts per batch. Measured on 130 real profiles, sorted by length: 16 took 28s, 32 35s,
# 64 49s; unsorted at fastembed's default of 256, 90s.
EMBED_BATCH = 16

_model = None
_model_name = None


def _get_model(model_name: str, cache_dir: Path):
    """Lazy-load the fastembed model singleton.

    Re-loads if a caller asks for a different model name than what's cached — a
    single process is expected to use one model, but nothing here enforces that.
    """
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from fastembed import TextEmbedding

        logger.debug("Loading embedding model %s from %s", model_name, cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        _model_name = model_name
    return _model


def embed_text(text: str, *, model_name: str = DEFAULT_MODEL, cache_dir: Path) -> np.ndarray:
    """Embed a single text string → 384-dim numpy array."""
    model = _get_model(model_name, cache_dir)
    embeddings = list(model.embed([text]))
    return np.array(embeddings[0], dtype=np.float32)


def embed_texts(texts: list[str], *, model_name: str = DEFAULT_MODEL, cache_dir: Path) -> np.ndarray:
    """Embed multiple texts → (N, 384) numpy array, in the order given.

    A batch is padded to its longest text, so the texts are embedded shortest first in
    small batches and put back in order: the same vectors, a third of the time on a page
    of Lead Finder profiles (one 2,681-character profile no longer pads 129 short ones).
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = _get_model(model_name, cache_dir)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = np.array(list(model.embed([texts[i] for i in order], batch_size=EMBED_BATCH)),
                       dtype=np.float32)
    in_order = np.empty_like(vectors)
    in_order[order] = vectors
    return in_order
