from app import __version__


def test_health_check_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_returns_healthy_status(client):
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"


def test_health_check_returns_correct_version(client):
    response = client.get("/health")
    data = response.json()
    assert data["version"] == __version__


def test_health_check_returns_timestamp(client):
    response = client.get("/health")
    data = response.json()
    assert "timestamp" in data
    assert data["timestamp"] is not None


def test_readiness_check_returns_200_when_healthy(client):
    response = client.get("/health/ready")
    assert response.status_code == 200


def test_readiness_check_returns_ready_status(client):
    response = client.get("/health/ready")
    data = response.json()
    assert data["status"] == "ready"


def test_readiness_check_qdrant_connected_true(client):
    response = client.get("/health/ready")
    data = response.json()
    assert data["qdrant_connected"] is True


def test_readiness_check_postgres_connected_true(client):
    response = client.get("/health/ready")
    data = response.json()
    assert data["postgres_connected"] is True


def test_readiness_check_includes_collection_info(client):
    response = client.get("/health/ready")
    data = response.json()
    assert "collection_info" in data
    assert data["collection_info"]["name"] == "csrag_documents"


def test_readiness_check_returns_503_when_qdrant_down(client_qdrant_down):
    response = client_qdrant_down.get("/health/ready")
    assert response.status_code == 503


def test_root_endpoint_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_root_endpoint_returns_service_name(client):
    response = client.get("/")
    data = response.json()
    assert "service" in data


def test_root_endpoint_returns_docs_link(client):
    response = client.get("/")
    data = response.json()
    assert data["docs"] == "/docs"
