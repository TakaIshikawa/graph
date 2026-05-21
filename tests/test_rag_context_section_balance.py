from __future__ import annotations

from types import SimpleNamespace

from graph.rag.context_section_balance import analyze_context_section_balance


def test_counts_normalized_sections_and_flags_dominance():
    report = analyze_context_section_balance(
        [
            {"id": "a", "metadata": {"section": "Methods"}},
            {"id": "b", "section": "Methods"},
            {"id": "c", "heading": "Results"},
        ]
    )

    assert report["section_counts"] == {"methods": 2, "results": 1}
    assert report["dominant_section"] == "methods"
    assert report["reason_counts"]["dominant_section_overrepresented"] == 2


def test_unknown_section_is_explicitly_counted():
    report = analyze_context_section_balance([{"id": "a", "text": "No metadata"}])

    assert report["section_counts"] == {"unknown": 1}
    assert report["reason_counts"]["missing_section_metadata"] == 1


def test_supports_object_and_tuple_wrapped_results():
    report = analyze_context_section_balance([(SimpleNamespace(id="obj", category="FAQ"), 0.5)])

    assert report["results"][0]["result_id"] == "obj"
    assert report["results"][0]["section"] == "faq"
