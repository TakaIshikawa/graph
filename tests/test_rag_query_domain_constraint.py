from __future__ import annotations

from graph.rag.query_domain_constraint import detect_query_domain_constraint


def test_detect_query_domain_constraint_returns_no_domains_for_generic_query():
    assert detect_query_domain_constraint("Compare retrieval chunking strategies.") == {
        "domains": [],
        "matched_cues": [],
        "requires_domain_care": False,
    }


def test_detect_query_domain_constraint_detects_high_stakes_cues_case_insensitively():
    result = detect_query_domain_constraint("Review MEDICAL, Legal, finance, and CYBERSECURITY evidence.")

    assert result["domains"] == ["healthcare", "legal", "finance", "security"]
    assert result["requires_domain_care"] is True
    assert [cue["cue"] for cue in result["matched_cues"]] == ["MEDICAL", "Legal", "finance", "CYBERSECURITY"]
    assert all(cue["requires_care"] for cue in result["matched_cues"])


def test_detect_query_domain_constraint_detects_non_high_stakes_domains():
    result = detect_query_domain_constraint("Find education policy research with peer-reviewed studies.")

    assert result["domains"] == ["education", "policy", "scientific"]
    assert result["requires_domain_care"] is False
    assert [cue["cue"] for cue in result["matched_cues"]] == ["education", "policy", "research", "peer-reviewed"]
