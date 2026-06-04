from __future__ import annotations

from graph.rag.query_waf_requirement import detect_query_waf_requirements


def test_waf_context_gates_output():
    result = detect_query_waf_requirements("Tune logging and bot rules for an unrelated queue.")

    assert result["has_waf_requirements"] is False
    assert result["requirements"] == []


def test_waf_detects_common_categories_with_stable_sorting():
    result = detect_query_waf_requirements(
        "Need a managed WAF with managed rules, OWASP CRS, bot protection, request blocking, "
        "tuning for false positives, logging, and virtual patching."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "blocking_mode",
        "bot_protection",
        "logging",
        "managed_service",
        "rule_sets",
        "tuning",
        "virtual_patching",
    ]
    assert all(row["matched_text"] and row["severity"] for row in result["requirements"])


def test_waf_detects_rate_rules_as_rule_sets():
    result = detect_query_waf_requirements("Should the web application firewall include rate rules?")

    assert result["has_waf_requirements"] is True
    assert result["requirements"][0] == {"category": "rule_sets", "matched_text": "rate rules", "severity": "high"}
