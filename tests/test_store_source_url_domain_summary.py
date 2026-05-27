from __future__ import annotations

from graph.store import summarize_source_url_domains


def test_summarize_source_url_domains_normalizes_and_classifies_urls():
    summary = summarize_source_url_domains([
        {"id": "a", "title": "A", "url": "HTTPS://Example.COM/a"},
        {"id": "b", "url": "example.com/b"},
        {"id": "c", "url": ""},
        {"id": "d", "url": "not a url"},
    ])

    assert summary["domains"][0]["domain"] == "example.com"
    assert summary["domains"][0]["count"] == 2
    assert summary["missing_url_count"] == 1
    assert summary["schemeless_url_count"] == 2
    assert summary["invalid_url_count"] == 1
