from app.config import get_settings


def test_settings_loads_without_error():
    settings = get_settings()
    assert settings is not None


def test_settings_llm_model_default():
    settings = get_settings()
    assert settings.llm_model == "llama-3.3-70b-versatile"


def test_settings_llm_temperature_default():
    settings = get_settings()
    assert settings.llm_temperature == 0.0


def test_settings_embedding_model_default():
    settings = get_settings()
    assert settings.embedding_model == "mxbai-embed-large"


def test_settings_embedding_dimension_default():
    settings = get_settings()
    assert settings.embedding_dimension == 1024


def test_settings_chunk_size_default():
    settings = get_settings()
    assert settings.chunk_size == 900


def test_settings_chunk_overlap_default():
    settings = get_settings()
    assert settings.chunk_overlap == 150


def test_settings_retrieval_k_default():
    settings = get_settings()
    assert settings.retrieval_k == 4


def test_settings_stm_threshold_default():
    settings = get_settings()
    assert settings.stm_message_threshold == 6


def test_settings_crag_upper_threshold_default():
    settings = get_settings()
    assert settings.crag_upper_threshold == 0.7


def test_settings_crag_lower_threshold_default():
    settings = get_settings()
    assert settings.crag_lower_threshold == 0.3


def test_settings_srag_max_retries_default():
    settings = get_settings()
    assert settings.srag_max_retries == 2


def test_settings_max_rewrite_tries_default():
    settings = get_settings()
    assert settings.max_rewrite_tries == 2


def test_settings_api_port_default():
    settings = get_settings()
    assert settings.api_port == 8000


def test_settings_api_host_default():
    settings = get_settings()
    assert settings.api_host == "0.0.0.0"


def test_settings_groq_api_key_from_env():
    settings = get_settings()
    assert settings.groq_api_key == "test-groq-key"


def test_settings_qdrant_url_from_env():
    settings = get_settings()
    assert settings.qdrant_url == "http://localhost:6333"


def test_settings_tavily_max_results_default():
    settings = get_settings()
    assert settings.tavily_max_results == 5


def test_settings_is_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
