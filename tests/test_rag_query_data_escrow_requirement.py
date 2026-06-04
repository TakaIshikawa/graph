from __future__ import annotations

from graph.rag.query_data_escrow_requirement import detect_query_data_escrow_requirements


def test_data_escrow_context_is_required_before_categories_are_emitted():
    result = detect_query_data_escrow_requirements("Describe release conditions for production access.")

    assert result["has_data_escrow_requirements"] is False
    assert result["requirements"] == []


def test_data_escrow_detects_agent_release_verification_and_beneficiary_access():
    result = detect_query_data_escrow_requirements(
        "Need data escrow with a third-party escrow agent, release conditions, "
        "verification of deposits, and beneficiary access."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "beneficiary_access",
        "deposit_scope",
        "escrow_agent",
        "release_conditions",
        "verification",
    ]
    assert all(row["matched_text"] and row["severity"] for row in result["requirements"])


def test_source_code_escrow_wording_is_recognized():
    result = detect_query_data_escrow_requirements("Does the vendor offer source-code escrow and escrow deposits?")

    assert result["has_data_escrow_requirements"] is True
    assert result["requirements"][0]["category"] == "deposit_scope"
    assert result["requirements"][0]["matched_text"] == "source-code escrow"
