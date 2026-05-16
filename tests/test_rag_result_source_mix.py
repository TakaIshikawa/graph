from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_source_mix import analyze_result_source_mix


def test_result_source_mix_summarizes_mixed_metadata():
    mix = analyze_result_source_mix(
        [
            {
                "id": "a",
                "source_project": "docs",
                "url": "https://example.com/a",
                "entity_type": "note",
            },
            {
                "id": "b",
                "metadata": {
                    "source_project": "docs",
                    "source_url": "https://example.com/b",
                    "entity_type": "task",
                },
            },
            {
                "id": "c",
                "source_project": "web",
                "url": "https://other.test/c",
                "entity_type": "note",
            },
        ]
    )

    assert mix["total_results"] == 3
    assert mix["source_project"] == [
        {"value": "docs", "count": 2, "percentage": 0.6667},
        {"value": "web", "count": 1, "percentage": 0.3333},
    ]
    assert mix["source_domain"][0] == {
        "value": "example.com",
        "count": 2,
        "percentage": 0.6667,
    }
    assert mix["entity_type"][0]["value"] == "note"
    assert mix["provenance"] == [
        {"value": "with_provenance", "count": 3, "percentage": 1.0}
    ]


def test_result_source_mix_groups_missing_values_and_objects():
    mix = analyze_result_source_mix(
        [
            SimpleNamespace(id="obj", content="No provenance"),
            {"id": "mapped", "domain": "Example.com", "type": "memo"},
        ]
    )

    assert mix["source_project"][0] == {
        "value": "unknown_project",
        "count": 2,
        "percentage": 1.0,
    }
    assert {"value": "unknown_domain", "count": 1, "percentage": 0.5} in mix[
        "source_domain"
    ]
    assert {"value": "missing_provenance", "count": 1, "percentage": 0.5} in mix[
        "provenance"
    ]
    assert mix["results"][0]["result_id"] == "obj"


def test_result_source_mix_empty_input():
    assert analyze_result_source_mix([]) == {
        "total_results": 0,
        "source_project": [],
        "source_domain": [],
        "entity_type": [],
        "provenance": [],
        "results": [],
    }
