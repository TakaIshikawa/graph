from __future__ import annotations

from graph.store.unit_external_link_domain_summary import summarize_unit_external_link_domains


def test_external_link_domain_summary_counts_markdown_and_bare_urls():
    summary = summarize_unit_external_link_domains(
        [
            {
                "id": "u1",
                "content": "Read [Docs](https://WWW.Example.com/a) and https://example.com/b.",
            },
            {
                "id": "u2",
                "content": "See http://news.example.org/story and [again](https://example.com/c).",
            },
            {"id": "u3", "content": "No external links here."},
        ],
        sample_limit=3,
    )

    assert summary == {
        "unit_count": 3,
        "linked_unit_count": 2,
        "external_link_count": 4,
        "domain_counts": {"example.com": 3, "news.example.org": 1},
        "samples": [
            {"unit_id": "u1", "url": "https://WWW.Example.com/a", "domain": "example.com"},
            {"unit_id": "u1", "url": "https://example.com/b", "domain": "example.com"},
            {"unit_id": "u2", "url": "https://example.com/c", "domain": "example.com"},
        ],
    }


def test_external_link_domain_summary_ignores_internal_and_non_http_links():
    summary = summarize_unit_external_link_domains(
        [
            {
                "id": "u1",
                "content": "[wiki](Note) [[Internal]] [anchor](#part) [mail](mailto:a@example.com) ftp://x.test/file https://www.Valid.test/path",
            }
        ]
    )

    assert summary["external_link_count"] == 1
    assert summary["domain_counts"] == {"valid.test": 1}
