from __future__ import annotations

from graph.rag.result_scope import classify_result_scope


def test_classify_result_scope_counts_personal_external_mixed_and_unknown():
    report = classify_result_scope(
        [
            {"id": "personal", "source_project": "notes", "source_entity_type": "note"},
            {"id": "external", "url": "https://example.test/a", "metadata": {"publisher": "Example"}},
            {"id": "mixed", "source_project": "tasks", "url": "https://vendor.test/a"},
            {"id": "unknown", "content": "No provenance"},
        ]
    )

    assert report["scope_counts"] == {"personal": 1, "external": 1, "mixed": 1, "unknown": 1}
    assert report["scope_percentages"] == {
        "personal": 25.0,
        "external": 25.0,
        "mixed": 25.0,
        "unknown": 25.0,
    }
    assert [row["scope"] for row in report["results"]] == ["personal", "external", "mixed", "unknown"]


def test_classify_result_scope_detects_file_urls_and_identifiers():
    report = classify_result_scope(
        [
            {"id": "file", "url": "file:///Users/me/note.md"},
            {"id": "path", "path": "/Users/me/task.md"},
            {"id": "doi", "metadata": {"doi": "10.123/example"}},
        ]
    )

    assert [row["scope"] for row in report["results"]] == ["personal", "personal", "external"]
    assert report["results"][2]["signals"] == ["publisher-or-identifier"]


def test_classify_result_scope_supports_nested_unit_values():
    report = classify_result_scope(
        [
            {
                "unit": {
                    "id": "unit",
                    "source_project": "calendar",
                    "metadata": {"source_url": "https://calendar-provider.test"},
                }
            }
        ]
    )

    assert report["results"] == [
        {
            "result_id": "unit",
            "scope": "mixed",
            "signals": ["http-domain", "personal-source-project"],
        }
    ]
