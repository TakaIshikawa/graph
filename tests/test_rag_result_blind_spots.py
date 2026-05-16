from __future__ import annotations

from graph.rag.result_blind_spots import detect_result_blind_spots


def codes(plan):
    return [spot["code"] for spot in plan["blind_spots"]]


def test_result_blind_spots_empty_results():
    plan = detect_result_blind_spots("alpha beta", [])

    assert codes(plan) == ["no_results"]
    assert plan["query_terms"] == ["alpha", "beta"]
    assert plan["blind_spots"][0]["severity"] == "high"


def test_result_blind_spots_flags_metadata_gaps():
    plan = detect_result_blind_spots(
        "alpha beta gamma delta",
        [
            {
                "id": "a",
                "content": "alpha only",
                "source_id": "one",
                "entity_type": "note",
                "published_at": "2020-01-01",
            },
            {
                "id": "b",
                "content": "unrelated",
                "source_id": "one",
                "entity_type": "note",
                "published_at": "2020-02-01",
            },
        ],
    )

    assert codes(plan) == [
        "missing_recent_evidence",
        "single_entity_type",
        "narrow_source_set",
        "no_primary_or_url_evidence",
        "weak_query_term_coverage",
    ]
    assert all({"code", "severity", "reason", "suggested_action"} <= set(spot) for spot in plan["blind_spots"])


def test_result_blind_spots_does_not_flag_well_covered_results():
    plan = detect_result_blind_spots(
        "alpha beta",
        [
            {
                "id": "a",
                "content": "alpha evidence",
                "source_id": "one",
                "entity_type": "note",
                "url": "https://example.com/a",
                "published_at": "2026-01-01",
            },
            {
                "id": "b",
                "content": "beta evidence",
                "source_id": "two",
                "entity_type": "article",
                "url": "https://example.org/b",
                "published_at": "2026-02-01",
            },
        ],
    )

    assert plan["blind_spots"] == []
    assert plan["counts"] == {"result_count": 2, "blind_spot_count": 0}
