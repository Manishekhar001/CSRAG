from unittest.mock import MagicMock, patch

import pytest

from app.core.crag.evaluator import CRAGEvaluator


@pytest.fixture
def mock_llm():
    with patch("app.core.crag.evaluator.ChatGroq") as mock_cls:
        llm = MagicMock()
        mock_cls.return_value = llm
        yield llm


@pytest.fixture
def evaluator(mock_llm):
    with patch("app.core.crag.evaluator.get_settings") as mock_settings:
        settings = MagicMock()
        settings.llm_model = "llama-3.3-70b-versatile"
        settings.llm_temperature = 0.0
        settings.groq_api_key = "test-key"
        settings.crag_upper_threshold = 0.7
        settings.crag_lower_threshold = 0.3
        mock_settings.return_value = settings
        return CRAGEvaluator()


def _make_doc(content="Test content about the refund policy."):
    from langchain_core.documents import Document
    return Document(page_content=content, metadata={"source": "test.pdf"})


def test_evaluate_returns_incorrect_when_no_docs(evaluator):
    verdict, reason, good_docs = evaluator.evaluate("What is the policy?", [])
    assert verdict == "INCORRECT"
    assert good_docs == []


def test_evaluate_returns_correct_when_high_score(evaluator):
    from app.core.crag.evaluator import DocEvalScore
    mock_result = MagicMock()
    mock_result.score = 0.85
    mock_result.reason = "Directly relevant."
    evaluator._eval_chain = MagicMock()
    evaluator._eval_chain.invoke.return_value = mock_result

    verdict, reason, good_docs = evaluator.evaluate("What is the policy?", [_make_doc()])
    assert verdict == "CORRECT"
    assert len(good_docs) == 1


def test_evaluate_returns_incorrect_when_all_low_scores(evaluator):
    mock_result = MagicMock()
    mock_result.score = 0.1
    mock_result.reason = "Not relevant."
    evaluator._eval_chain = MagicMock()
    evaluator._eval_chain.invoke.return_value = mock_result

    verdict, reason, good_docs = evaluator.evaluate("What is the policy?", [_make_doc()])
    assert verdict == "INCORRECT"
    assert good_docs == []


def test_evaluate_returns_ambiguous_for_mid_range_scores(evaluator):
    mock_result = MagicMock()
    mock_result.score = 0.5
    mock_result.reason = "Partially relevant."
    evaluator._eval_chain = MagicMock()
    evaluator._eval_chain.invoke.return_value = mock_result

    verdict, reason, good_docs = evaluator.evaluate("What is the policy?", [_make_doc()])
    assert verdict == "AMBIGUOUS"
    assert len(good_docs) == 1


def test_evaluate_filters_good_docs_above_lower_threshold(evaluator):
    results = [
        MagicMock(score=0.5, reason="Partial"),
        MagicMock(score=0.1, reason="Irrelevant"),
        MagicMock(score=0.8, reason="Highly relevant"),
    ]
    evaluator._eval_chain = MagicMock()
    evaluator._eval_chain.invoke.side_effect = results

    docs = [_make_doc() for _ in range(3)]
    verdict, reason, good_docs = evaluator.evaluate("What is the policy?", docs)
    assert verdict == "CORRECT"
    assert len(good_docs) == 2


def test_evaluate_handles_eval_chain_error_gracefully(evaluator):
    evaluator._eval_chain = MagicMock()
    evaluator._eval_chain.invoke.side_effect = Exception("LLM timeout")

    verdict, reason, good_docs = evaluator.evaluate("What is the policy?", [_make_doc()])
    assert verdict == "INCORRECT"


def test_evaluate_reason_contains_max_score(evaluator):
    mock_result = MagicMock()
    mock_result.score = 0.9
    mock_result.reason = "Directly answers."
    evaluator._eval_chain = MagicMock()
    evaluator._eval_chain.invoke.return_value = mock_result

    verdict, reason, good_docs = evaluator.evaluate("What is the policy?", [_make_doc()])
    assert "0.90" in reason
