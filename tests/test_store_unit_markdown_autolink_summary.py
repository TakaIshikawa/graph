from __future__ import annotations

from graph.store import summarize_unit_markdown_autolinks


def test_markdown_autolink_summary_counts_urls_and_emails():
    report = summarize_unit_markdown_autolinks([{"id": "a", "content": "<https://Example.test/a> <mailto:a@example.test> <user@example.test>"}])

    assert report["total_autolinks"] == 3
    assert report["url_autolink_count"] == 1
    assert report["email_autolink_count"] == 2
    assert report["scheme_counts"] == {"https": 1, "mailto": 1}
    assert report["domain_counts"] == {"example.test": 3}


def test_markdown_autolink_summary_ignores_plain_angle_text():
    assert summarize_unit_markdown_autolinks([{"content": "<not a link> <tag>"}])["total_autolinks"] == 0


def test_markdown_autolink_summary_ignores_fenced_code():
    report = summarize_unit_markdown_autolinks([{"content": "```\n<https://skip.test>\n```\n<https://keep.test>"}])

    assert report["total_autolinks"] == 1
    assert report["domain_counts"] == {"keep.test": 1}
