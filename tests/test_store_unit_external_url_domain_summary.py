from __future__ import annotations

from graph.store.unit_external_url_domain_summary import summarize_unit_external_url_domains


def test_summarize_unit_external_url_domains_counts_urls_and_units_by_hostname():
    summary = summarize_unit_external_url_domains(
        [
            {
                "id": "u1",
                "content": "Read https://Example.com/a?x=1 and https://example.com/b#part.",
                "metadata": {"canonical": "https://docs.example.org/start?ref=1"},
            },
            {
                "id": "u2",
                "content": "Again https://EXAMPLE.com/a?x=2 plus ftp://ignored.test/file",
                "metadata": {"links": ["not a url", "http://example.com/path"]},
            },
        ],
        sample_limit=2,
    )

    assert summary["total_units"] == 2
    assert summary["domains"][0] == {
        "hostname": "example.com",
        "unit_count": 2,
        "url_count": 4,
        "source_counts": [{"source": "content", "count": 3}, {"source": "metadata", "count": 1}],
        "examples": [
            {"unit_id": "u1", "url": "https://Example.com/a?x=1"},
            {"unit_id": "u1", "url": "https://example.com/b#part"},
        ],
    }
    assert summary["domains"][1]["hostname"] == "docs.example.org"


def test_summarize_unit_external_url_domains_ignores_malformed_values_without_raising():
    summary = summarize_unit_external_url_domains(
        [{"id": "bad", "content": "https:// should not count", "metadata": {"url": "http:///missing-host"}}]
    )

    assert summary == {"total_units": 1, "domains": []}
