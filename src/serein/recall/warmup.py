"""Explicit provider warmup for deployments that need low first-recall latency."""

import time

from ..adapters.embedding import EmbeddingClient
from ..adapters.reranker import RerankerClient
from ..configured_models import effective_settings


_WARMUP_QUERY = "Serein recall provider warmup"
_WARMUP_DOCUMENT = {
    "ref": "warmup:probe",
    "title": "Warmup probe",
    "body": "Synthetic provider warmup text. No user memory is involved.",
}


def warm_recall_models(settings, *, timeout_seconds=60):
    timeout = float(timeout_seconds)
    if not 5 <= timeout <= 300:
        raise ValueError("Recall warmup timeout must be between 5 and 300 seconds")

    selected = effective_settings(settings)
    if not selected.embedding or not selected.reranker:
        raise ValueError("Recall warmup requires configured embedding and reranker providers")

    import httpx

    timings = {}
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        embedding = EmbeddingClient(
            selected.database,
            selected.index,
            **selected.embedding,
        )
        started = time.monotonic()
        embedded = embedding.query(_WARMUP_QUERY, client=client)
        timings["embedding_seconds"] = round(time.monotonic() - started, 3)

        reranker = RerankerClient(**selected.reranker)
        started = time.monotonic()
        scores = reranker(_WARMUP_QUERY, [_WARMUP_DOCUMENT], client=client)
        timings["reranker_seconds"] = round(time.monotonic() - started, 3)

    if len(embedded.get("embedding") or []) != embedding.dimension:
        raise ValueError("Embedding warmup returned an unexpected vector dimension")
    if scores.get(_WARMUP_DOCUMENT["ref"]) is None:
        raise ValueError("Reranker warmup returned no score")

    return {
        "status": "ready",
        "embedding_dimension": embedding.dimension,
        **timings,
    }
