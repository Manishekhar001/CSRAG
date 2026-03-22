from unittest.mock import MagicMock


def test_list_memories_returns_200(client):
    response = client.get("/memory/user-test-001")
    assert response.status_code == 200


def test_list_memories_returns_user_id(client):
    response = client.get("/memory/user-test-001")
    data = response.json()
    assert data["user_id"] == "user-test-001"


def test_list_memories_returns_memories_list(client):
    response = client.get("/memory/user-test-001")
    data = response.json()
    assert "memories" in data
    assert isinstance(data["memories"], list)


def test_list_memories_returns_count(client):
    response = client.get("/memory/user-test-001")
    data = response.json()
    assert "count" in data
    assert data["count"] == 0


def test_list_memories_count_matches_memories_length(client):
    response = client.get("/memory/user-test-001")
    data = response.json()
    assert data["count"] == len(data["memories"])


def test_list_memories_with_stored_data(client, mock_postgres_store):
    item = MagicMock()
    item.value = {"data": "User's name is Nitish."}
    item.key = "some-uuid"
    mock_postgres_store.search.return_value = [item]

    response = client.get("/memory/user-test-001")
    data = response.json()
    assert data["count"] == 1
    assert data["memories"][0]["data"] == "User's name is Nitish."


def test_delete_memories_returns_200(client):
    response = client.delete("/memory/user-test-001")
    assert response.status_code == 200


def test_delete_memories_returns_user_id(client):
    response = client.delete("/memory/user-test-001")
    data = response.json()
    assert data["user_id"] == "user-test-001"


def test_delete_memories_returns_message(client):
    response = client.delete("/memory/user-test-001")
    data = response.json()
    assert "message" in data


def test_delete_memories_calls_store_delete(client, mock_postgres_store):
    item = MagicMock()
    item.value = {"data": "Some fact."}
    item.key = "uuid-to-delete"
    mock_postgres_store.search.return_value = [item]

    client.delete("/memory/user-test-001")
    mock_postgres_store.delete.assert_called_once()


def test_delete_memories_when_empty(client, mock_postgres_store):
    mock_postgres_store.search.return_value = []
    response = client.delete("/memory/user-test-001")
    assert response.status_code == 200
    data = response.json()
    assert "Deleted 0" in data["message"]


def test_list_memories_different_user_ids(client):
    response_1 = client.get("/memory/user-001")
    response_2 = client.get("/memory/user-002")
    assert response_1.status_code == 200
    assert response_2.status_code == 200
    assert response_1.json()["user_id"] == "user-001"
    assert response_2.json()["user_id"] == "user-002"
