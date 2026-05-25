from __future__ import annotations

from graph.store.source_sitemap_coverage_summary import summarize_source_sitemap_coverage


def test_summarize_source_sitemap_coverage_aggregates_duplicate_hosts():
    summary = summarize_source_sitemap_coverage(
        [
            {
                "host": "Example.COM",
                "metadata": {
                    "discovered_url_count": 10,
                    "ingested_url_count": 6,
                    "has_sitemap": True,
                },
            },
            {
                "url": "https://example.com/docs",
                "metadata": {
                    "sitemap_discovered_url_count": 5,
                    "sitemap_ingested_url_count": 3,
                    "missing_sitemap": True,
                },
            },
            {
                "metadata": {
                    "source_url": "https://beta.test/start",
                    "discovered_url_count": 4,
                    "ingested_url_count": 4,
                },
            },
        ],
        low_coverage_threshold=0.7,
    )

    assert summary == {
        "hosts": [
            {
                "host": "beta.test",
                "discovered_url_count": 4,
                "ingested_url_count": 4,
                "coverage_ratio": 1.0,
                "missing_sitemap_count": 0,
            },
            {
                "host": "example.com",
                "discovered_url_count": 15,
                "ingested_url_count": 9,
                "coverage_ratio": 0.6,
                "missing_sitemap_count": 1,
            },
        ],
        "low_coverage_hosts": ["example.com"],
    }


def test_summarize_source_sitemap_coverage_handles_zero_discovered_urls():
    summary = summarize_source_sitemap_coverage(
        [
            {
                "host": "empty.test",
                "metadata": {"discovered_url_count": 0, "ingested_url_count": 0, "sitemap_found": False},
            }
        ],
        low_coverage_threshold=1.0,
    )

    assert summary["hosts"] == [
        {
            "host": "empty.test",
            "discovered_url_count": 0,
            "ingested_url_count": 0,
            "coverage_ratio": 0.0,
            "missing_sitemap_count": 1,
        }
    ]
    assert summary["low_coverage_hosts"] == []
