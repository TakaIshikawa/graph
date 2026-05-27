from __future__ import annotations

from graph.rag.result_language_mismatch import analyze_result_language_mismatch


def test_result_language_mismatch_uses_explicit_query_language():
    report = analyze_result_language_mismatch(
        [
            {"id": "1", "language": "en-US", "title": "One"},
            {"id": "2", "metadata": {"locale": "ja_JP"}, "title": "Two"},
            {"id": "3", "metadata": {}},
        ],
        query_language="en",
    )

    assert report["matching_language_count"] == 1
    assert report["mismatched_language_count"] == 1
    assert report["missing_language_count"] == 1
    assert report["mismatch_examples"] == [{"id": "2", "language": "ja", "title": "Two"}]


def test_result_language_mismatch_infers_dominant_language_deterministically():
    report = analyze_result_language_mismatch(
        [
            {"id": "1", "lang": "fr"},
            {"id": "2", "metadata": {"detected_language": "en"}},
            {"id": "3", "locale": "en-GB"},
            {"id": "4", "language": "FR"},
        ]
    )

    assert report["query_language"] == "en"
    assert report["languages"] == [{"language": "en", "count": 2}, {"language": "fr", "count": 2}]
    assert report["mismatched_language_count"] == 2
