from types import SimpleNamespace

from graph.rag.context_provenance_signal import analyze_context_provenance_signals


def test_accepts_dict_object_and_raw_string_context_items():
    result = analyze_context_provenance_signals([
        {"source_url": "https://example.test", "text": "Body"},
        SimpleNamespace(author="Ada", text="Body"),
        "Citation: Example 2024. Retrieved 2024-01-01.",
    ])

    assert result["missing_provenance_items"] == []
    assert result["provenance_score"] == 1.0


def test_counts_metadata_fields_separately_from_text_cues():
    result = analyze_context_provenance_signals([{"source_url": "https://example.test", "text": "Source: report. license: CC."}])

    assert result["provenance_counts"]["metadata_source"] == 1
    assert result["provenance_counts"]["text_source"] == 1
    assert result["provenance_counts"]["text_citation"] == 1


def test_reports_missing_items_without_source_author_date_or_citation():
    result = analyze_context_provenance_signals([{"text": "Unattributed context."}, {"date": "2024-01-01"}])

    assert result["missing_provenance_items"] == [0]
    assert result["provenance_score"] == 0.5
