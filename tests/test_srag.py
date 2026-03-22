"""Tests for SRAG verifier module."""

from unittest.mock import MagicMock, patch

import pytest


class TestSRAGVerifier:
    """Test suite for SRAGVerifier class."""

    @pytest.fixture
    def verifier(self):
        """Create SRAGVerifier with mocked LLM and settings."""
        with patch("app.core.srag.verifier.ChatGroq") as mock_cls:
            llm = MagicMock()
            mock_cls.return_value = llm
            with patch("app.core.srag.verifier.get_settings") as mock_settings:
                settings = MagicMock()
                settings.llm_model = "llama-3.3-70b-versatile"
                settings.llm_temperature = 0.0
                settings.groq_api_key = "test-key"
                mock_settings.return_value = settings
                from app.core.srag.verifier import SRAGVerifier
                return SRAGVerifier()

    def test_verify_support_returns_fully_supported(self, verifier):
        """Test that a well-grounded answer returns fully_supported."""
        mock_result = MagicMock()
        mock_result.verdict = "fully_supported"
        mock_result.evidence = ["The refund window is 30 days."]
        verifier._support_chain = MagicMock()
        verifier._support_chain.invoke.return_value = mock_result

        verdict, evidence = verifier.verify_support("What is the refund?", "Context text.", "30 days.")
        assert verdict == "fully_supported"
        assert len(evidence) == 1

    def test_verify_support_returns_partially_supported(self, verifier):
        """Test that a partially grounded answer returns partially_supported."""
        mock_result = MagicMock()
        mock_result.verdict = "partially_supported"
        mock_result.evidence = []
        verifier._support_chain = MagicMock()
        verifier._support_chain.invoke.return_value = mock_result

        verdict, evidence = verifier.verify_support("Question", "Context", "Answer")
        assert verdict == "partially_supported"

    def test_verify_support_defaults_to_fully_supported_on_error(self, verifier):
        """Test graceful fallback to fully_supported when verification fails."""
        verifier._support_chain = MagicMock()
        verifier._support_chain.invoke.side_effect = Exception("LLM error")

        verdict, evidence = verifier.verify_support("Q", "C", "A")
        assert verdict == "fully_supported"
        assert evidence == []

    def test_verify_usefulness_returns_useful(self, verifier):
        """Test that a relevant answer returns useful verdict."""
        mock_result = MagicMock()
        mock_result.verdict = "useful"
        mock_result.reason = "Directly answers the question."
        verifier._usefulness_chain = MagicMock()
        verifier._usefulness_chain.invoke.return_value = mock_result

        verdict, reason = verifier.verify_usefulness("What is the refund?", "30 day window.")
        assert verdict == "useful"
        assert "Directly" in reason

    def test_verify_usefulness_returns_not_useful(self, verifier):
        """Test that a vague answer returns not_useful verdict."""
        mock_result = MagicMock()
        mock_result.verdict = "not_useful"
        mock_result.reason = "Answer is too vague."
        verifier._usefulness_chain = MagicMock()
        verifier._usefulness_chain.invoke.return_value = mock_result

        verdict, reason = verifier.verify_usefulness("What is the refund?", "It depends.")
        assert verdict == "not_useful"

    def test_verify_usefulness_defaults_to_useful_on_error(self, verifier):
        """Test graceful fallback to useful when usefulness check fails."""
        verifier._usefulness_chain = MagicMock()
        verifier._usefulness_chain.invoke.side_effect = Exception("LLM error")

        verdict, reason = verifier.verify_usefulness("Q", "A")
        assert verdict == "useful"

    def test_revise_answer_returns_revised_string(self, verifier):
        """Test that revise_answer returns the revised answer text."""
        mock_result = MagicMock()
        mock_result.content = "Revised answer based only on context."
        verifier._revise_chain = MagicMock()
        verifier._revise_chain.invoke.return_value = mock_result

        revised = verifier.revise_answer("Question", "Context", "Original answer.")
        assert revised == "Revised answer based only on context."

    def test_revise_answer_returns_original_on_error(self, verifier):
        """Test that the original answer is returned when revision fails."""
        verifier._revise_chain = MagicMock()
        verifier._revise_chain.invoke.side_effect = Exception("LLM error")

        original = "Original answer."
        revised = verifier.revise_answer("Q", "C", original)
        assert revised == original
