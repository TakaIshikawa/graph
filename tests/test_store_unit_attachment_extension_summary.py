from __future__ import annotations

from graph.store.unit_attachment_extension_summary import summarize_unit_attachment_extensions


def test_unit_attachment_extension_summary_extracts_metadata_and_markdown_extensions():
    summary = summarize_unit_attachment_extensions(
        [
            {"id": "a", "metadata": {"attachments": ["Report.PDF", "archive"]}, "content": "[local](docs/image.PNG) [web](https://x.test/a.jpg)"},
            {"id": "b", "metadata": {"file": {"path": "notes.md"}}},
            {"id": "c", "content": "none"},
        ]
    )

    assert summary["extension_counts"] == {"(none)": 1, "md": 1, "pdf": 1, "png": 1}
    assert summary["units_with_attachments"] == 2
    assert summary["units_without_attachments"] == 1
    assert summary["examples_by_extension"]["pdf"] == ["Report.PDF"]
