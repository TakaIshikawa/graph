from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag import analyze_result_provenance_completeness


@dataclass
class UnitStub:
    id: str
    source_id: str
    source_project: str
    source_entity_type: str
    metadata: dict


@dataclass
class ResultStub:
    unit_id: str
    unit: UnitStub
    metadata: dict


def test_analyze_result_provenance_completeness_handles_empty_input():
    assert analyze_result_provenance_completeness([]) == {
        "total_results": 0,
        "complete_result_count": 0,
        "incomplete_result_count": 0,
        "missing_field_counts": {
            "citation": 0,
            "source_entity_type": 0,
            "source_id": 0,
            "source_project": 0,
        },
        "per_result_gaps": [],
        "completeness_percent": 0.0,
    }


def test_analyze_result_provenance_completeness_reads_top_level_and_metadata():
    payload = analyze_result_provenance_completeness(
        [
            {
                "id": "top",
                "source_id": "source-1",
                "source_project": "notes",
                "source_entity_type": "note",
                "url": "https://example.test/top",
            },
            {
                "id": "metadata",
                "metadata": {
                    "source_id": "source-2",
                    "source_project": "bookmarks",
                    "source_entity_type": "bookmark",
                    "canonical_url": "https://example.test/metadata",
                },
            },
        ]
    )

    assert payload == {
        "total_results": 2,
        "complete_result_count": 2,
        "incomplete_result_count": 0,
        "missing_field_counts": {
            "citation": 0,
            "source_entity_type": 0,
            "source_id": 0,
            "source_project": 0,
        },
        "per_result_gaps": [],
        "completeness_percent": 100.0,
    }


def test_analyze_result_provenance_completeness_reads_objects_tuples_and_nested_units():
    unit = UnitStub(
        id="unit-1",
        source_id="source-1",
        source_project="readwise",
        source_entity_type="highlight",
        metadata={"doi": "10.1234/example"},
    )
    result = ResultStub(unit_id="wrapper-1", unit=unit, metadata={})

    payload = analyze_result_provenance_completeness(
        [
            (result, 0.92),
            {
                "unit_id": "wrapped-missing",
                "unit": {
                    "source_id": "source-2",
                    "metadata": {
                        "source_project": "archive",
                        "source_entity_type": "page",
                    },
                },
            },
        ]
    )

    assert payload["complete_result_count"] == 1
    assert payload["incomplete_result_count"] == 1
    assert payload["missing_field_counts"] == {
        "citation": 1,
        "source_entity_type": 0,
        "source_id": 0,
        "source_project": 0,
    }
    assert payload["per_result_gaps"] == [
        {"result_id": "wrapped-missing", "missing_fields": ["citation"]}
    ]
    assert payload["completeness_percent"] == 50.0


def test_analyze_result_provenance_completeness_accepts_citation_alternatives():
    payload = analyze_result_provenance_completeness(
        [
            {
                "source_id": "a",
                "source_project": "papers",
                "source_entity_type": "paper",
                "citations": ["Smith 2026"],
            },
            {
                "source_id": "b",
                "source_project": "web",
                "source_entity_type": "page",
                "link": "https://example.test/page",
            },
        ]
    )

    assert payload["complete_result_count"] == 2
    assert payload["missing_field_counts"]["citation"] == 0


def test_analyze_result_provenance_completeness_custom_required_fields_override_defaults():
    payload = analyze_result_provenance_completeness(
        [
            {"id": "a", "title": "A", "metadata": {"author": "Ada"}},
            {"id": "b", "title": "B"},
        ],
        required_fields=["title", "author"],
    )

    assert payload["missing_field_counts"] == {"author": 1, "title": 0}
    assert payload["per_result_gaps"] == [
        {"result_id": "b", "missing_fields": ["author"]}
    ]


def test_analyze_result_provenance_completeness_uses_stable_identifiers():
    payload = analyze_result_provenance_completeness(
        [
            {"id": "id-1"},
            {"unit_id": "unit-1"},
            {"source_id": "source-1"},
            {},
        ],
        required_fields=["source_project"],
    )

    assert [row["result_id"] for row in payload["per_result_gaps"]] == [
        "id-1",
        "unit-1",
        "source-1",
        "result-4",
    ]


@pytest.mark.parametrize("required_fields", [[], [""], [None]])
def test_analyze_result_provenance_completeness_validates_required_fields(required_fields):
    with pytest.raises(ValueError, match="required_fields"):
        analyze_result_provenance_completeness([], required_fields=required_fields)  # type: ignore[list-item]
