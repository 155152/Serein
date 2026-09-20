import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from serein.api.gateway import routes


def app_with_capture(captured):
    def recall(_query, **kwargs):
        captured.append(kwargs.get("deadline_at"))
        return {
            "cards": [],
            "context": "",
            "selected_refs": [],
            "routing": {},
            "status": "no_match",
        }

    app = FastAPI()
    app.include_router(routes(SimpleNamespace(recall=recall), []))
    return app


def test_hook_keeps_nine_second_default_but_accepts_bounded_host_budget():
    captured = []
    with TestClient(app_with_capture(captured)) as client:
        before = time.monotonic()
        assert client.post("/api/hook/recall", json={"query": "默认预算"}).status_code == 200
        default_delta = captured[-1] - before

        before = time.monotonic()
        assert client.post("/api/hook/recall", json={
            "query": "宿主预算",
            "deadline_seconds": 15,
        }).status_code == 200
        custom_delta = captured[-1] - before

        before = time.monotonic()
        assert client.post("/api/hook/recall", json={
            "query": "上限",
            "deadline_seconds": 99,
        }).status_code == 200
        capped_delta = captured[-1] - before

        before = time.monotonic()
        assert client.post("/api/hook/recall", json={
            "query": "下限",
            "deadline_seconds": 1,
        }).status_code == 200
        floored_delta = captured[-1] - before

    assert 8.5 <= default_delta <= 9.5
    assert 14.5 <= custom_delta <= 15.5
    assert 29.5 <= capped_delta <= 30.5
    assert 2.5 <= floored_delta <= 3.5


def test_hook_rejects_non_numeric_deadline_budget():
    with TestClient(app_with_capture([])) as client:
        response = client.post("/api/hook/recall", json={
            "query": "错误预算",
            "deadline_seconds": "15",
        })
    assert response.status_code == 400
    assert "deadline_seconds" in response.json()["detail"]
