from __future__ import annotations

from graph.store import summarize_unit_markdown_link_schemes


def test_summarize_unit_markdown_link_schemes_buckets_destinations():
    summary = summarize_unit_markdown_link_schemes(
        [{"id": "a", "content": "[h](http://x) [s](https://x) [m](mailto:a@b) [r](docs/a) [a](#top) [e]()"}]
    )
    assert summary["scheme_counts"] == [
        {"scheme": "anchor", "count": 1},
        {"scheme": "http", "count": 1},
        {"scheme": "https", "count": 1},
        {"scheme": "mailto", "count": 1},
        {"scheme": "relative", "count": 1},
        {"scheme": "unknown", "count": 1},
    ]
