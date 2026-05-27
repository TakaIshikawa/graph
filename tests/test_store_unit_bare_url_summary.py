from __future__ import annotations

from graph.store import summarize_unit_bare_urls


def test_bare_url_summary_counts_plain_urls():
    report = summarize_unit_bare_urls([{"id": "a", "content": "See https://Example.test/a."}])

    assert report["total_bare_urls"] == 1
    assert report["scheme_counts"] == {"https": 1}
    assert report["domain_counts"] == {"example.test": 1}
    assert report["examples"][0]["url"] == "https://Example.test/a"


def test_bare_url_summary_ignores_markdown_links_and_autolinks():
    report = summarize_unit_bare_urls([{"id": "a", "content": "[x](https://example.test)\n<https://example.test>"}])

    assert report["total_bare_urls"] == 0


def test_bare_url_summary_skips_fenced_code_and_strips_punctuation():
    report = summarize_unit_bare_urls([{"id": "a", "content": "```\nhttps://skip.test\n```\nVisit https://ok.test/path)."}])

    assert report["examples"] == [{"unit_id": "a", "line": 4, "url": "https://ok.test/path", "domain": "ok.test"}]
