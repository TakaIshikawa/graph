from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_reference_link_text_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_full_collapsed_and_shortcut_reference_usages():
    content = "[Visible][Label]\n[Label][]\n[Shortcut]\n[label]: /target"

    result = rows(export_units_to_markdown_reference_link_text_csv([{"id": "u", "title": "T", "source": "s", "content": content}]))

    assert [(row["visible_text"], row["reference_label"], row["normalized_label"], row["usage_style"], row["definition_exists"]) for row in result] == [
        ("Visible", "Label", "label", "full", "True"),
        ("Label", "Label", "label", "collapsed", "True"),
        ("Shortcut", "Shortcut", "shortcut", "shortcut", "False"),
    ]
    assert result[0]["source"] == "s"


def test_ignores_images_inline_links_definitions_and_code():
    content = "![alt][img] [inline](x) `[code]` [ok]\n[ok]: /ok"

    result = rows(export_units_to_markdown_reference_link_text_csv([{"id": "u", "content": content}]))

    assert [(row["visible_text"], row["usage_style"], row["definition_exists"]) for row in result] == [("ok", "shortcut", "True")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "reference_text.csv"

    result = export_units_to_markdown_reference_link_text_csv([{"id": "u", "content": "[x]\n[x]: /x"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
