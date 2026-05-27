from __future__ import annotations

from types import SimpleNamespace

from graph.store.unit_canonical_url_summary import summarize_unit_canonical_urls


def test_summarize_unit_canonical_urls_groups_and_counts_duplicates():
    units = [
        {"id": "1", "source_project": "web", "canonical_url": "HTTPS://Example.com/A"},
        {"id": "2", "source_project": "web", "metadata": {"url": "https://example.com/A"}},
        {"id": "3", "source_project": "web", "source_url": "http://example.com/b"},
        {"id": "4", "source_project": "web", "canonical_url": "not a url"},
        {"id": "5", "source_project": "web", "canonical_url": ""},
        SimpleNamespace(id="6", source_project="notes", metadata={"permalink": "https://notes.local/x"}),
    ]

    summary = summarize_unit_canonical_urls(units)

    assert summary["unit_count"] == 6
    assert summary["rows"] == [
        {
            "source": "notes",
            "unit_count": 1,
            "canonical_url_count": 1,
            "missing_canonical_url_count": 0,
            "http_url_count": 0,
            "https_url_count": 1,
            "duplicate_canonical_url_count": 0,
        },
        {
            "source": "web",
            "unit_count": 5,
            "canonical_url_count": 4,
            "missing_canonical_url_count": 1,
            "http_url_count": 1,
            "https_url_count": 2,
            "duplicate_canonical_url_count": 2,
        },
    ]
