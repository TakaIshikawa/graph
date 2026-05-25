from __future__ import annotations

import pytest

from graph.rag.query_persona_context import detect_query_persona_context


def test_persona_context_detects_roles_and_ownership():
    result = detect_query_persona_context("As a founder, should I adapt this for my team and our company?")

    assert result["has_persona_context"] is True
    assert result["persona_roles"] == ["founder", "team", "company"]
    assert result["ownership_cues"] == ["my", "our"]
    assert result["privacy_cues"] == ["internal"]


def test_persona_context_flags_privacy_separately():
    result = detect_query_persona_context("What should I ask my doctor about my family history and my portfolio?")

    assert result["persona_roles"] == ["doctor", "family"]
    assert result["ownership_cues"] == ["my", "portfolio"]
    assert result["privacy_cues"] == ["health", "family", "financial"]


def test_persona_context_no_cues_is_false():
    result = detect_query_persona_context("Explain vector databases.")

    assert result["has_persona_context"] is False
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", " ", None])
def test_persona_context_validates_query(query):
    with pytest.raises(ValueError):
        detect_query_persona_context(query)  # type: ignore[arg-type]
