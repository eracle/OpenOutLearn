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
    """Embed multiple texts → (N, 384) numpy array."""
    model = _get_model(model_name, cache_dir)
    embeddings = list(model.embed(texts))
    return np.array(embeddings, dtype=np.float32)
