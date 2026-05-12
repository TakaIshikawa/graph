from __future__ import annotations

import pytest

from graph.rag import audit_source_diversity


def distribution(payload: dict, facet: str, value: str) -> dict:
    return next(item for item in payload["distributions"][facet] if item["value"] == value)


def test_source_diversity_audit_reports_distributions_and_examples():
    payload = audit_source_diversity(
        [
            {
                "id": "a",
                "source_project": "Notes",
                "source_entity_type": "note",
                "content_type": "text/markdown",
                "url": "https://www.example.com/a",
            },
            {
                "id": "b",
                "source_project": "Papers",
                "source_entity_type": "paper",
                "content_type": "application/pdf",
                "metadata": {"source_url": "http://journal.example/paper"},
            },
            {
                "id": "c",
                "source_project": "Notes",
                "source_entity_type": "note",
                "content_type": "text/markdown",
                "link": "example.com/c",
            },
        ],
        max_examples=1,
    )

    assert payload["totals"] == {
        "result_count": 3,
        "facet_count": 4,
        "dominance_threshold": 0.6,
    }
    assert distribution(payload, "source_project", "notes") == {
        "value": "notes",
        "count": 2,
        "ratio": 0.667,
        "representative_result_ids": ["a"],
    }
    assert distribution(payload, "citation_host", "example.com")["count"] == 2


def test_source_diversity_audit_warns_deterministically_for_dominance():
    payload = audit_source_diversity(
        [
            {"id": "a", "source_project": "notes", "content_type": "text"},
            {"id": "b", "source_project": "notes", "content_type": "text"},
            {"id": "c", "source_project": "web", "content_type": "text"},
        ]
    )

    assert payload["dominance_warnings"] == [
        {
            "facet": "citation_host",
            "value": "unknown",
            "ratio": 1.0,
            "message": "citation_host value unknown represents 3 of 3 results",
        },
        {
            "facet": "content_type",
            "value": "text",
            "ratio": 1.0,
            "message": "content_type value text represents 3 of 3 results",
        },
        {
            "facet": "source_entity_type",
            "value": "unknown",
            "ratio": 1.0,
            "message": "source_entity_type value unknown represents 3 of 3 results",
        },
        {
            "facet": "source_project",
            "value": "notes",
            "ratio": 0.667,
            "message": "source_project value notes represents 2 of 3 results",
        },
    ]


def test_source_diversity_audit_supports_nested_units():
    payload = audit_source_diversity(
        [
            {
                "unit": {
                    "id": "nested",
                    "metadata": {
                        "source_project": "archive",
                        "source_entity_type": "bookmark",
                        "content_type": "html",
                        "canonical_url": "https://Archive.Example/path",
                    },
                }
            }
        ]
    )

    assert distribution(payload, "citation_host", "archive.example")[
        "representative_result_ids"
    ] == ["nested"]


@pytest.mark.parametrize(
    "kwargs",
    [{"max_examples": 0}, {"max_examples": True}, {"dominance_threshold": 0}, {"dominance_threshold": 1.1}],
)
def test_source_diversity_audit_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        audit_source_diversity([], **kwargs)
