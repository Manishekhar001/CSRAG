import pytest


CHAT_PAYLOAD = {
    "question": "What is the refund policy?",
    "thread_id": "thread-test-001",
    "user_id": "user-test-001",
    "include_sources": True,
}


def test_chat_returns_200(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    assert response.status_code == 200


def test_chat_returns_answer(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_chat_returns_question(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert data["question"] == CHAT_PAYLOAD["question"]


def test_chat_returns_sources_when_requested(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert data["sources"] is not None
    assert isinstance(data["sources"], list)


def test_chat_sources_have_origin_field(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    for source in data["sources"]:
        assert source["origin"] in ("internal", "web")


def test_chat_returns_no_sources_when_not_requested(client):
    payload = {**CHAT_PAYLOAD, "include_sources": False}
    response = client.post("/chat", json=payload)
    data = response.json()
    assert data["sources"] is None


def test_chat_returns_processing_time(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert "processing_time_ms" in data
    assert data["processing_time_ms"] >= 0


def test_chat_returns_crag_verdict(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert "crag_verdict" in data
    assert data["crag_verdict"] == "CORRECT"


def test_chat_returns_issup(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert "issup" in data
    assert data["issup"] == "fully_supported"


def test_chat_returns_isuse(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert "isuse" in data
    assert data["isuse"] == "useful"


def test_chat_returns_evidence_list(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert "evidence" in data
    assert isinstance(data["evidence"], list)


def test_chat_returns_retries(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert "retries" in data
    assert data["retries"] == 0


def test_chat_returns_rewrite_tries(client):
    response = client.post("/chat", json=CHAT_PAYLOAD)
    data = response.json()
    assert "rewrite_tries" in data
    assert data["rewrite_tries"] == 0


def test_chat_empty_question_returns_422(client):
    payload = {**CHAT_PAYLOAD, "question": ""}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_chat_missing_thread_id_returns_422(client):
    payload = {k: v for k, v in CHAT_PAYLOAD.items() if k != "thread_id"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_chat_missing_user_id_returns_422(client):
    payload = {k: v for k, v in CHAT_PAYLOAD.items() if k != "user_id"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_chat_question_too_long_returns_422(client):
    payload = {**CHAT_PAYLOAD, "question": "a" * 2001}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_chat_stream_returns_200(client):
    response = client.post("/chat/stream", json=CHAT_PAYLOAD)
    assert response.status_code == 200


def test_chat_stream_returns_text_plain(client):
    response = client.post("/chat/stream", json=CHAT_PAYLOAD)
    assert "text/plain" in response.headers["content-type"]


def test_chat_stream_returns_content(client):
    response = client.post("/chat/stream", json=CHAT_PAYLOAD)
    assert len(response.content) > 0


def test_chat_calls_engine_aquery(client, mock_engine):
    client.post("/chat", json=CHAT_PAYLOAD)
    mock_engine.aquery.assert_called_once_with(
        question=CHAT_PAYLOAD["question"],
        thread_id=CHAT_PAYLOAD["thread_id"],
        user_id=CHAT_PAYLOAD["user_id"],
    )
