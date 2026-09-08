import numpy as np
import pytest

from openoutlearn import embeddings


class _FakeModel:
    """Stands in for fastembed's TextEmbedding — returns a fixed-width vector per text."""

    def embed(self, texts):
        return [np.full(4, len(t), dtype=np.float32) for t in texts]


@pytest.fixture(autouse=True)
def _reset_model_cache():
    embeddings._model = None
    embeddings._model_name = None
    yield
    embeddings._model = None
    embeddings._model_name = None


def test_embed_text_returns_a_vector(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "_get_model", lambda model_name, cache_dir: _FakeModel())

    vec = embeddings.embed_text("hello", cache_dir=tmp_path)

    assert vec.shape == (4,)


def test_embed_texts_returns_one_vector_per_text(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "_get_model", lambda model_name, cache_dir: _FakeModel())

    vecs = embeddings.embed_texts(["a", "bb", "ccc"], cache_dir=tmp_path)

    assert vecs.shape == (3, 4)


def test_get_model_reloads_when_model_name_changes(monkeypatch, tmp_path):
    calls = []

    class _StubTextEmbedding:
        def __init__(self, model_name, cache_dir):
            calls.append(model_name)

        def embed(self, texts):
            return [np.zeros(4, dtype=np.float32) for _ in texts]

    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)

    embeddings._get_model("model-a", tmp_path)
    embeddings._get_model("model-a", tmp_path)
    embeddings._get_model("model-b", tmp_path)

    assert calls == ["model-a", "model-b"]
