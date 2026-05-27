from __future__ import annotations

from graph.store.unit_html_heading_anchor_summary import summarize_unit_html_heading_anchors


def test_summarize_unit_html_heading_anchors_includes_content_and_metadata():
    summary = summarize_unit_html_heading_anchors(
        [
            {"id": "u1", "source": "notes", "content": '<h2 id="intro">Intro</h2>', "metadata": {"html": '<h3 id="intro">Again</h3>'}},
            {"id": "u2", "content": '<h1 id="top">Top</h1>'},
        ]
    )

    assert summary["total_anchors"] == 3
    assert summary["duplicate_anchor_ids"] == [{"anchor_id": "intro", "count": 2}]
    assert summary["anchors_by_level"] == {"h1": 1, "h2": 1, "h3": 1}
    assert summary["examples"][0] == {"unit_id": "u1", "source": "notes", "anchor_id": "intro", "heading_text": "Intro"}
