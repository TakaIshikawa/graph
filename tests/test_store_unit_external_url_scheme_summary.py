from __future__ import annotations

from graph.store.unit_external_url_scheme_summary import summarize_unit_external_url_schemes


def test_external_url_schemes_group_http_https_and_metadata_urls():
    summary = summarize_unit_external_url_schemes(
        [
            {"id": "u1", "content": "[A](http://example.com/a) https://example.com/b", "metadata": {"source_url": "mailto:a@example.com"}},
            {"id": "u2", "content": "[B](ftp://files.example.com/x) custom+v1:thing", "metadata": {"canonical_url": "file:///tmp/a.md"}},
        ]
    )

    rows = {row["scheme"]: row for row in summary["schemes"]}
    assert rows["http"]["url_count"] == 1
    assert rows["https"]["url_count"] == 1
    assert rows["mailto"]["url_count"] == 1
    assert rows["ftp"]["url_count"] == 1
    assert rows["file"]["url_count"] == 1
    assert rows["custom+v1"]["url_count"] == 1


def test_external_url_schemes_ignore_relative_links_wikilinks_and_deduplicate_unit_count():
    summary = summarize_unit_external_url_schemes(
        [
            {"id": "u1", "content": "[Relative](/docs/a) [[Wiki]] https://example.com/a https://example.com/a"},
            {"id": "u2", "content": "https://example.com/a"},
        ],
        sample_limit=1,
    )

    assert summary["schemes"] == [
        {
            "scheme": "https",
            "url_count": 3,
            "unit_count": 2,
            "examples": [{"unit_id": "u1", "url": "https://example.com/a", "source": "content"}],
        }
    ]
