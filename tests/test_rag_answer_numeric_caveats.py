from __future__ import annotations

from graph.rag.answer_numeric_caveats import analyze_answer_numeric_caveats


def test_low_risk_fully_qualified_numeric_claims():
    report = analyze_answer_numeric_caveats(
        "According to the 2025 survey, 42% of 1,000 respondents changed plans during 2025. "
        "The median of $50 among surveyed users was reported by the dataset in 2025."
    )

    assert report["risk_level"] == "low"
    assert len(report["numeric_claims"]) == 2
    assert report["caveat_gaps"] == {
        "missing_denominator": [],
        "missing_timeframe": [],
        "missing_source": [],
    }


def test_high_risk_unsupported_numeric_claims_flag_separate_gaps():
    report = analyze_answer_numeric_caveats("Costs rose 25%. Average spend was $123.45. There were 400 users.")

    assert report["risk_level"] == "high"
    assert {claim["claim_type"] for claim in report["numeric_claims"]} >= {"percent", "average", "count"}
    assert "25%" in report["caveat_gaps"]["missing_denominator"]
    assert "25%" in report["caveat_gaps"]["missing_timeframe"]
    assert "25%" in report["caveat_gaps"]["missing_source"]


def test_detects_range_currency_count_average_median_and_percent():
    report = analyze_answer_numeric_caveats(
        "Reported by the 2024 study, 10 to 20 cases out of 200 appeared in 2024. "
        "According to the report, median 15% of the sample occurred during 2024."
    )

    types = [claim["claim_type"] for claim in report["numeric_claims"]]
    assert "range" in types
    assert "average" in types
