from __future__ import annotations

from dataclasses import dataclass

from graph.store.unit_reference_resolution_summary import summarize_unit_reference_resolution


@dataclass
class Unit:
    id: str
    metadata: dict[str, object]


def test_summarize_unit_reference_resolution_counts_strings_mappings_and_ratios():
    summary = summarize_unit_reference_resolution(
        [
            {
                "id": "u1",
                "metadata": {
                    "references": [
                        "unit:resolved",
                        {"target": "unit:also-resolved", "resolved": True},
                        {"target": "unit:missing", "resolved": False},
                    ]
                },
            },
            Unit(
                id="u2",
                metadata={
                    "citations": [
                        {"url": "https://ok.test/ref", "status_code": 200},
                        {"url": "https://gone.test/ref", "status": 404},
                    ]
                },
            ),
            {"id": "u3", "links": [{"href": "https://error.test/ref", "status": "error"}]},
        ]
    )

    assert summary == {
        "total_units": 3,
        "reference_count": 6,
        "resolved_reference_count": 3,
        "unresolved_reference_count": 3,
        "broken_url_count": 2,
        "units": [
            {
                "unit_id": "u1",
                "reference_count": 3,
                "resolved_reference_count": 2,
                "unresolved_reference_count": 1,
                "broken_url_count": 0,
                "resolution_ratio": 0.67,
            },
            {
                "unit_id": "u2",
                "reference_count": 2,
                "resolved_reference_count": 1,
                "unresolved_reference_count": 1,
                "broken_url_count": 1,
                "resolution_ratio": 0.5,
            },
            {
                "unit_id": "u3",
                "reference_count": 1,
                "resolved_reference_count": 0,
                "unresolved_reference_count": 1,
                "broken_url_count": 1,
                "resolution_ratio": 0.0,
            },
        ],
    }


def test_summarize_unit_reference_resolution_marks_missing_targets_unresolved():
    summary = summarize_unit_reference_resolution(
        [
            {"id": "empty", "metadata": {}},
            {"id": "missing", "metadata": {"links": [{"status": 200}, {"url": ""}]}},
            {"id": "ok", "metadata": {"references": [{"id": "target", "status": "ok"}]}},
        ]
    )

    assert summary["reference_count"] == 3
    assert summary["resolved_reference_count"] == 1
    assert summary["unresolved_reference_count"] == 2
    assert summary["units"] == [
        {
            "unit_id": "empty",
            "reference_count": 0,
            "resolved_reference_count": 0,
            "unresolved_reference_count": 0,
            "broken_url_count": 0,
            "resolution_ratio": 0.0,
        },
        {
            "unit_id": "missing",
            "reference_count": 2,
            "resolved_reference_count": 0,
            "unresolved_reference_count": 2,
            "broken_url_count": 0,
            "resolution_ratio": 0.0,
        },
        {
            "unit_id": "ok",
            "reference_count": 1,
            "resolved_reference_count": 1,
            "unresolved_reference_count": 0,
            "broken_url_count": 0,
            "resolution_ratio": 1.0,
        },
    ]
