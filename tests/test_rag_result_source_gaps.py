from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_source_gaps import detect_result_source_gaps
from graph.types.models import KnowledgeUnit


def test_source_gaps_reports_complete_attribution():
    payload = detect_result_source_gaps(
        [
            {
                "id": "a",
                "source": "Journal",
                "url": "https://example.test/a",
                "title": "Study",
                "author": "Ada",
            }
        ]
    )

    assert payload["totals"] == {
        "result_count": 1,
        "missing_source": 0,
        "missing_url": 0,
        "missing_title": 0,
        "missing_author": 0,
        "complete_attribution": 1,
    }
    assert payload["result_gaps"][0]["missing_fields"] == []


def test_source_gaps_reports_missing_urls_and_unknown_sources():
    payload = detect_result_source_gaps(
        [
            {"id": "a", "source": "unknown", "title": "Untitled", "author": "Ada"},
            {"id": "b", "source_project": "notes", "url": "https://example.test/b"},
        ]
    )

    assert payload["totals"]["missing_source"] == 1
    assert payload["totals"]["missing_url"] == 1
    assert payload["totals"]["missing_title"] == 1
    assert payload["totals"]["missing_author"] == 1
    assert payload["result_gaps"][0]["missing_fields"] == ["source", "url"]
    assert payload["result_gaps"][1]["missing_fields"] == ["title", "author"]


def test_source_gaps_uses_nested_unit_metadata_before_marking_missing():
    unit = KnowledgeUnit.model_construct(
        id="nested",
        source_project="readwise",
        source_id="source-nested",
        source_entity_type="highlight",
        title="Nested title",
        content="Text",
        metadata={"canonical_url": "https://book.test", "author": "Grace"},
        tags=[],
    )
    payload = detect_result_source_gaps([SimpleNamespace(id="wrapper", unit=unit)])

    assert payload["result_gaps"] == [
        {
            "result_id": "wrapper",
            "title": "Nested title",
            "source": "readwise",
            "url": "https://book.test",
            "author": "Grace",
            "missing_fields": [],
            "gap_count": 0,
        }
    ]


def test_source_gaps_accepts_tuple_results_and_empty_inputs():
    assert detect_result_source_gaps([]) == {
        "totals": {
            "result_count": 0,
            "missing_source": 0,
            "missing_url": 0,
            "missing_title": 0,
            "missing_author": 0,
            "complete_attribution": 0,
        },
        "result_gaps": [],
    }

    payload = detect_result_source_gaps([({"unit_id": "tuple", "metadata": {"publisher": "Archive"}}, 0.8)])
    assert payload["result_gaps"][0]["result_id"] == "tuple"
    assert payload["result_gaps"][0]["source"] == "Archive"
