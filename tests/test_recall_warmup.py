from types import SimpleNamespace

import pytest

from serein.recall import warmup


def test_warm_recall_models_calls_only_configured_providers(monkeypatch):
    calls = []

    selected = SimpleNamespace(
        database="memory.db",
        index="index.db",
        embedding={"endpoint": "http://embedding.local/v1/embeddings", "api_key": "embed-key"},
        reranker={"endpoint": "http://reranker.local/v1/rerank", "model": "reranker", "api_key": "rerank-key"},
    )
    monkeypatch.setattr(warmup, "effective_settings", lambda _settings: selected)

    class Embedding:
        dimension = 3

        def __init__(self, database, index, **config):
            calls.append(("embedding_init", database, index, config))

        def query(self, text, *, client=None):
            calls.append(("embedding_query", text, client))
            return {"embedding": [1.0, 0.0, 0.0]}

    class Reranker:
        def __init__(self, **config):
            calls.append(("reranker_init", config))

        def __call__(self, text, documents, *, client=None):
            calls.append(("reranker_call", text, documents, client))
            return {"warmup:probe": 0.5}

    class Client:
        def __init__(self, **kwargs):
            calls.append(("client_init", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append(("client_close",))

    monkeypatch.setattr(warmup, "EmbeddingClient", Embedding)
    monkeypatch.setattr(warmup, "RerankerClient", Reranker)

    import httpx
    monkeypatch.setattr(httpx, "Client", Client)

    result = warmup.warm_recall_models(object(), timeout_seconds=45)

    assert result["status"] == "ready"
    assert result["embedding_dimension"] == 3
    assert result["embedding_seconds"] >= 0
    assert result["reranker_seconds"] >= 0
    assert calls[0] == ("client_init", {"timeout": 45.0, "follow_redirects": False})
    assert calls[1][0] == "embedding_init"
    assert calls[2][0] == "embedding_query"
    assert calls[3][0] == "reranker_init"
    assert calls[4][0] == "reranker_call"
    assert calls[-1] == ("client_close",)


@pytest.mark.parametrize("timeout", [0, 4.9, 301])
def test_warm_recall_models_rejects_unbounded_timeout(timeout):
    with pytest.raises(ValueError, match="between 5 and 300"):
        warmup.warm_recall_models(object(), timeout_seconds=timeout)


def test_warm_recall_models_requires_both_providers(monkeypatch):
    selected = SimpleNamespace(database="memory.db", index="index.db", embedding=None, reranker=None)
    monkeypatch.setattr(warmup, "effective_settings", lambda _settings: selected)
    with pytest.raises(ValueError, match="embedding and reranker"):
        warmup.warm_recall_models(object())
