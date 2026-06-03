from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_callout_type_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_callout_type_csv_exports_nested_fold_markers_and_titles():
    text = export_units_to_markdown_callout_type_csv(
        [
            {
                "id": "u",
                "title": "Unit",
                "source": "note.md",
                "content": "> [!note]\n> > [!WARNING]- Risk\n>>> [!todo]+ Next step",
            }
        ]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Unit", "source": "note.md", "line_number": "1", "callout_type": "note", "fold_marker": "", "callout_title": "", "blockquote_depth": "1"},
        {"unit_id": "u", "title": "Unit", "source": "note.md", "line_number": "2", "callout_type": "warning", "fold_marker": "-", "callout_title": "Risk", "blockquote_depth": "2"},
        {"unit_id": "u", "title": "Unit", "source": "note.md", "line_number": "3", "callout_type": "todo", "fold_marker": "+", "callout_title": "Next step", "blockquote_depth": "3"},
    ]


def test_callout_type_csv_ignores_fenced_code_and_sorts():
    text = export_units_to_markdown_callout_type_csv(
        [
            {"id": "z", "content": "```md\n> [!note] Ignored\n```\n> [!tip] Real"},
            {"id": "a", "metadata": {"title": "Alpha", "source_id": "alpha.md", "content": "> [!question]- Ask"}},
        ]
    )

    assert _rows(text) == [
        {"unit_id": "a", "title": "Alpha", "source": "alpha.md", "line_number": "1", "callout_type": "question", "fold_marker": "-", "callout_title": "Ask", "blockquote_depth": "1"},
        {"unit_id": "z", "title": "", "source": "", "line_number": "4", "callout_type": "tip", "fold_marker": "", "callout_title": "Real", "blockquote_depth": "1"},
    ]


def test_callout_type_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "callouts.csv"
    units = [{"unit_id": "u1", "content": "> [!todo]+ Ship"}]

    expected = export_units_to_markdown_callout_type_csv(units)
    stats = export_units_to_markdown_callout_type_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
