from __future__ import annotations

from graph.rag import analyze_result_format_coverage


def test_result_format_coverage_normalizes_mime_extensions_and_urls():
    result = analyze_result_format_coverage([
        {"mime_type": "application/pdf"},
        {"content_type": "text/html; charset=utf-8"},
        {"metadata": {"file_extension": ".md"}},
        {"url": "https://example.com/data.csv"},
        {"metadata": {"url": "https://example.com/image.png"}},
        {"format": "transcript"},
        {},
    ])

    counts = {item["format"]: item["count"] for item in result["formats"]}
    assert counts == {"html": 1, "pdf": 1, "markdown": 1, "transcript": 1, "dataset": 1, "image": 1, "unknown": 1}
    assert result["dominant_format"] == "html"
    assert result["result_count"] == 7
