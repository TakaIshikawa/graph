from __future__ import annotations

from graph.store import summarize_unit_markdown_setext_headings


def test_counts_heading_levels_and_sorts_samples_by_input_order():
    report = summarize_unit_markdown_setext_headings([{"id": "u", "content": "One\n===\n\nTwo\n---\n\n---"}])

    assert report["setext_heading_count"] == 2
    assert report["level_counts"] == {1: 1, 2: 1}
    assert [sample["text"] for sample in report["setext_heading_samples"]] == ["One", "Two"]


def test_fenced_code_and_sample_limit():
    content = "```\nSkip\n---\n```\nKeep\n---\nNext\n==="

    report = summarize_unit_markdown_setext_headings([{"id": "u", "content": content}], sample_limit=1)

    assert report["setext_heading_count"] == 2
    assert report["setext_heading_samples"] == [{"unit_id": "u", "line_number": 5, "level": 2, "text": "Keep"}]
