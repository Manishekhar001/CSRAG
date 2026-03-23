"""Tests for application settings and configuration."""

import pytest


class TestSettings:
    """Test suite for application Settings class."""

    def test_settings_loads_without_error(self):
        """Test that Settings loads successfully from environment."""
        from app.config import get_settings
        settings = get_settings()
        assert settings is not None

    def test_settings_llm_model_default(self):
        """Test default LLM model is set correctly."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.llm_model == "llama-3.3-70b-versatile"

    def test_settings_llm_temperature_default(self):
        """Test default LLM temperature is zero."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.llm_temperature == 0.0

    def test_settings_embedding_model_default(self):
        """Test default embedding model is nomic-embed-text (or env override)."""
        from app.config import get_settings
        settings = get_settings()
        # Environment variable may override the default
        assert settings.embedding_model in ("nomic-embed-text", "mxbai-embed-large")

    def test_settings_embedding_dimension_default(self):
        """Test default embedding dimension is set correctly."""
        from app.config import get_settings
        settings = get_settings()
        # Environment variable may override the default (768 for nomic, 1024 for mxbai)
        assert settings.embedding_dimension in (768, 1024)

    def test_settings_chunk_size_default(self):
        """Test default chunk size is configured correctly."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.chunk_size == 900

    def test_settings_chunk_overlap_default(self):
        """Test default chunk overlap is configured correctly."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.chunk_overlap == 150

    def test_settings_retrieval_k_default(self):
        """Test default retrieval k is configured correctly."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.retrieval_k == 4

    def test_settings_stm_threshold_default(self):
        """Test default STM message threshold is configured correctly."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.stm_message_threshold == 6

    def test_settings_crag_upper_threshold_default(self):
        """Test default CRAG upper threshold is 0.7."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.crag_upper_threshold == 0.7

    def test_settings_crag_lower_threshold_default(self):
        """Test default CRAG lower threshold is 0.3."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.crag_lower_threshold == 0.3

    def test_settings_srag_max_retries_default(self):
        """Test default SRAG max retries is 2."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.srag_max_retries == 2

    def test_settings_max_rewrite_tries_default(self):
        """Test default max rewrite tries is 2."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.max_rewrite_tries == 2

    def test_settings_api_port_default(self):
        """Test default API port is 8000."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.api_port == 8000

    def test_settings_api_host_default(self):
        """Test default API host binds to all interfaces."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.api_host == "0.0.0.0"

    def test_settings_groq_api_key_from_env(self):
        """Test that GROQ_API_KEY is read from environment."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.groq_api_key == "test-groq-key"

    def test_settings_qdrant_url_from_env(self):
        """Test that QDRANT_URL is read from environment."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.qdrant_url == "http://localhost:6333"

    def test_settings_tavily_max_results_default(self):
        """Test default Tavily max results is 5."""
        from app.config import get_settings
        settings = get_settings()
        assert settings.tavily_max_results == 5

    def test_settings_is_cached(self):
        """Test that get_settings returns the same object on repeated calls."""
        from app.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
