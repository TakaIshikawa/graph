from __future__ import annotations

from graph.store import summarize_unit_markdown_footnote_backrefs


def test_counts_numeric_named_and_multiple_backrefs():
    content = '<a href="#fnref1">one</a>\n<a href="#fnref-note">note</a>\n<a href="#fnref1">again</a>'

    report = summarize_unit_markdown_footnote_backrefs([{"id": "u", "content": content}])

    assert report["footnote_backref_count"] == 3
    assert report["footnote_id_counts"] == {"-note": 1, "1": 2}


def test_ignores_unrelated_and_fenced_links_and_limits_samples():
    content = '<a href="#other">x</a>\n```\n<a href="#fnrefskip">skip</a>\n```\n<a href="#fnrefok">ok</a>\n<a href="#fnrefnext">next</a>'

    report = summarize_unit_markdown_footnote_backrefs([{"id": "u", "content": content}], sample_limit=1)

    assert report["footnote_backref_count"] == 2
    assert report["footnote_backref_samples"] == [{"unit_id": "u", "line_number": 5, "footnote_id": "ok", "href": "#fnrefok", "backref_text": "ok"}]
