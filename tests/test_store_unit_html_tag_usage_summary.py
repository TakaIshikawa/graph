from __future__ import annotations

from graph.store import summarize_unit_html_tag_usage


def test_html_tag_usage_summary_counts_tags_and_attributes():
    report = summarize_unit_html_tag_usage([{"id": "a", "content": '<span class="x">Hi</span><br />'}])

    assert report["total_tags"] == 3
    assert report["unique_tag_names"] == 2
    assert report["self_closing_tag_count"] == 1
    assert report["closing_tag_count"] == 1
    assert report["attribute_tag_count"] == 1


def test_html_tag_usage_summary_ignores_comments():
    report = summarize_unit_html_tag_usage([{"content": "<!-- <span>skip</span> -->"}])

    assert report["total_tags"] == 0


def test_html_tag_usage_summary_ignores_fenced_code():
    report = summarize_unit_html_tag_usage([{"content": "```\n<div>x</div>\n```\n<p>ok</p>"}])

    assert report["total_tags"] == 2
    assert report["top_tags"][0]["tag"] == "p"
