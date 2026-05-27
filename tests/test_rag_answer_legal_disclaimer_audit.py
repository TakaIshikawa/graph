from __future__ import annotations

from graph.rag.answer_legal_disclaimer_audit import audit_answer_legal_disclaimer


def test_audit_answer_legal_disclaimer_scores_legal_content_higher():
    ordinary = audit_answer_legal_disclaimer("The deployment can use blue green rollout steps.")
    legal = audit_answer_legal_disclaimer("This contract may create liability under employment law.")

    assert ordinary["legal_sensitivity_score"] == 0.0
    assert legal["legal_sensitivity_score"] > ordinary["legal_sensitivity_score"]
    assert "add_professional_disclaimer" in legal["recommendations"]


def test_audit_answer_legal_disclaimer_detects_disclaimer_and_jurisdiction_caveat():
    result = audit_answer_legal_disclaimer(
        "This is not legal advice and laws vary by jurisdiction. Consult a qualified attorney before acting.",
        query="Can I terminate this contract?",
    )

    assert result["legal_sensitivity_score"] > 0
    assert result["has_jurisdiction_caveat"] is True
    assert result["has_professional_disclaimer"] is True
    assert result["recommendations"] == []


def test_audit_answer_legal_disclaimer_flags_specific_legal_advice_deterministically():
    result = audit_answer_legal_disclaimer(
        "You should sue the vendor and definitely file in court.",
        query="What are my legal options for breach of contract?",
    )

    assert result["has_specific_legal_advice_flags"] == ["you should sue", "definitely file"]
    assert "soften_over_specific_legal_advice" in result["recommendations"]


def test_audit_answer_legal_disclaimer_ordinary_answer_has_no_flags():
    result = audit_answer_legal_disclaimer("Use the product changelog to compare release notes.")

    assert result == {
        "legal_sensitivity_score": 0.0,
        "has_jurisdiction_caveat": False,
        "has_professional_disclaimer": False,
        "has_specific_legal_advice_flags": [],
        "recommendations": [],
    }
