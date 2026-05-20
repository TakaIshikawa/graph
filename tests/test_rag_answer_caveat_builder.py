from __future__ import annotations

from graph.rag.answer_caveat_builder import build_answer_caveats


def test_answer_caveats_report_empty_and_low_result_counts():
    assert build_answer_caveats("query", []) == [
        "No retrieved results were available, so the answer may be incomplete."
    ]

    assert build_answer_caveats("query", [{"id": "a"}]) == [
        "Only 1 retrieved result supported the answer."
    ]


def test_answer_caveats_use_optional_diagnostics_in_priority_order():
    caveats = build_answer_caveats(
        "python package release",
        [{"id": "a"}, {"id": "b"}],
        {
            "source_gaps": {"totals": {"missing_url": 1}},
            "freshness": [{"result_id": "a", "age_days": 900, "freshness_score": 0.1}],
            "conflicts": {"conflict_count": 2},
            "coverage": [{"result_id": "b", "coverage_score": 0.25, "missing_terms": ["release"]}],
            "date_gaps": {"result_gaps": [{"missing_fields": ["published_date"]}]},
        },
    )

    assert caveats == [
        "Some retrieved evidence is missing source attribution.",
        "Some retrieved evidence may be stale.",
        "Retrieved evidence contains possible conflicts.",
        "Retrieved evidence has weak coverage of the query.",
        "Some retrieved evidence is missing usable dates.",
    ]


def test_answer_caveats_collapse_duplicate_signals():
    caveats = build_answer_caveats(
        "deployment status",
        [{"id": "a"}, {"id": "b"}],
        [
            {"missing_fields": ["source", "url"], "missing_source": 1},
            {"missing_attribution": True},
            {"conflict_count": 1, "flags": [{"term": "deployment"}]},
            {"has_conflicts": "true"},
        ],
    )

    assert caveats == [
        "Some retrieved evidence is missing source attribution.",
        "Retrieved evidence contains possible conflicts.",
    ]


def test_answer_caveats_can_trigger_missing_dates_without_marking_stale():
    caveats = build_answer_caveats(
        "roadmap timing",
        [{"id": "a"}, {"id": "b"}],
        {"freshness": [{"result_id": "a", "freshness_score": 0.0, "reason": "missing date metadata"}]},
    )

    assert caveats == ["Some retrieved evidence is missing usable dates."]
