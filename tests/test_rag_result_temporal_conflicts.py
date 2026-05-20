from __future__ import annotations

from dataclasses import dataclass

from graph.rag.result_temporal_conflicts import detect_result_temporal_conflicts


@dataclass
class ResultObject:
    id: str
    title: str
    date: str


def test_temporal_conflicts_detects_planned_to_completed_status_change():
    rows = detect_result_temporal_conflicts(
        [
            {"id": "old", "title": "Migration alpha planned", "published_at": "2024-01-10"},
            {"id": "new", "title": "Migration alpha completed", "published_at": "2024-03-10"},
        ]
    )

    assert rows == [
        {
            "topic": "alpha migration",
            "older_result_id": "old",
            "newer_result_id": "new",
            "older_date": "2024-01-10",
            "newer_date": "2024-03-10",
            "conflict_type": "planned_vs_completed",
            "reason": "planned status on older evidence conflicts with completed status on newer evidence",
        }
    ]


def test_temporal_conflicts_detects_other_status_pairs_and_nested_dates():
    rows = detect_result_temporal_conflicts(
        [
            {"id": "a", "claim": "API v2 active", "metadata": {"date": "2023-05-01"}},
            {"id": "b", "claim": "API v2 deprecated", "metadata": {"updated_at": "2024-05-01T09:00:00Z"}},
            {"id": "c", "claim": "Portal beta available", "metadata": {"created_at": "2024-01-01"}},
            {"id": "d", "claim": "Portal beta unavailable", "metadata": {"created_at": "2024-02-01"}},
        ]
    )

    assert [row["conflict_type"] for row in rows] == [
        "active_vs_deprecated",
        "available_vs_unavailable",
    ]
    assert rows[0]["topic"] == "api v2"
    assert rows[1]["topic"] == "beta portal"


def test_temporal_conflicts_supports_objects_and_tuple_wrapped_payloads():
    rows = detect_result_temporal_conflicts(
        [
            ResultObject("object-old", "Issue 42 open", "2024-01-01"),
            ({"result_id": "tuple-new", "title": "Issue 42 closed", "date": "2024-01-02"}, 0.7),
        ]
    )

    assert rows[0]["topic"] == "42 issue"
    assert rows[0]["conflict_type"] == "open_vs_closed"
    assert rows[0]["older_result_id"] == "object-old"
    assert rows[0]["newer_result_id"] == "tuple-new"


def test_temporal_conflicts_ignores_undated_or_non_conflicting_results():
    assert detect_result_temporal_conflicts(None) == []
    assert (
        detect_result_temporal_conflicts(
            [
                {"id": "a", "title": "Metric increased", "date": "2024-01-01"},
                {"id": "b", "title": "Metric increased again", "date": "2024-02-01"},
                {"id": "c", "title": "Metric decreased"},
            ]
        )
        == []
    )
