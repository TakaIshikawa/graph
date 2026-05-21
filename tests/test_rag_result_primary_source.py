from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_primary_source import classify_result_primary_sources


def test_classifies_primary_secondary_tertiary_and_unknown():
    report = classify_result_primary_sources(
        [
            {"id": "p", "source_type": "dataset"},
            {"id": "s", "publication_type": "analysis"},
            {"id": "t", "document_type": "encyclopedia"},
            {"id": "u", "title": "Unlabeled page"},
        ]
    )

    assert report["category_counts"] == {"primary": 1, "secondary": 1, "tertiary": 1, "unknown": 1}
    assert report["unknown_count"] == 1


def test_metadata_source_type_takes_precedence_over_title_heuristics():
    report = classify_result_primary_sources([{"id": "x", "source_type": "secondary", "title": "Official dataset"}])

    assert report["results"][0]["category"] == "secondary"
    assert report["results"][0]["reasons"] == ["metadata_source_type_secondary"]


def test_supports_tuple_wrapped_and_object_style_results():
    report = classify_result_primary_sources([(SimpleNamespace(id="obj", publisher="Official Standards Body"), 0.7)])

    assert report["primary_count"] == 1
    assert report["results"][0]["result_id"] == "obj"
