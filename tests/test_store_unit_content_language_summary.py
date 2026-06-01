from __future__ import annotations

from graph.store import summarize_unit_content_languages


def test_unit_content_language_summary_normalizes_languages_and_examples():
    summary = summarize_unit_content_languages(
        [
            {"id": "a", "language": "EN-US"},
            {"id": "b", "metadata": {"lang": "en_GB"}},
            {"id": "c", "metadata": {"language": "ja"}},
            {"id": "d", "metadata": {"language": "unknown"}},
            {"id": "e"},
        ]
    )

    assert summary["total_units"] == 5
    assert summary["units_with_language"] == 3
    assert summary["units_missing_language"] == 2
    assert summary["language_counts"] == {"en": 2, "ja": 1}
    assert summary["examples"] == [{"language": "en", "unit_ids": ["a", "b"]}, {"language": "ja", "unit_ids": ["c"]}]


def test_unit_content_language_summary_handles_empty_units():
    assert summarize_unit_content_languages([])["language_counts"] == {}
