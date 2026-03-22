"""Tests for health check endpoints."""

from app import __version__


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_health_check_returns_correct_version(self, client):
        """Test health check returns the correct application version."""
        response = client.get("/health")

        data = response.json()
        assert data["version"] == __version__

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data

    def test_root_endpoint_docs_link(self, client):
        """Test root endpoint returns correct docs URL."""
        response = client.get("/")

        data = response.json()
        assert data["docs"] == "/docs"

    def test_docs_available(self, client):
        """Test that Swagger docs are accessible."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_openapi_available(self, client):
        """Test that OpenAPI schema is accessible."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_readiness_check(self, client, mock_vector_store):
        """Test readiness check when all dependencies are healthy."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["qdrant_connected"] is True
        assert data["postgres_connected"] is True
        assert "collection_info" in data

    def test_readiness_check_collection_info(self, client):
        """Test readiness check includes correct collection info."""
        response = client.get("/health/ready")

        data = response.json()
        assert data["collection_info"]["name"] == "csrag_documents"

    def test_readiness_check_qdrant_down(self, client_qdrant_down):
        """Test readiness check returns 503 when Qdrant is unreachable."""
        response = client_qdrant_down.get("/health/ready")

        assert response.status_code == 503
