from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_frontmatter_comment_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_full_line_and_inline_comments():
    content = "---\n# standalone\ntitle: Value # inline\n---"

    found = rows(export_unit_frontmatter_comment_csv([{"id": "u", "content": content}]))

    assert [(row["comment"], row["inline"], row["field_key"]) for row in found] == [("standalone", "false", ""), ("inline", "true", "title")]


def test_quoted_hash_values_are_not_comments():
    content = "---\ntitle: \"A # value\"\ntag: '#literal'\nplain: ok # yes\n---"

    found = rows(export_unit_frontmatter_comment_csv([{"id": "u", "content": content}]))

    assert [(row["field_key"], row["comment"]) for row in found] == [("plain", "yes")]


def test_path_writing(tmp_path):
    path = tmp_path / "comments.csv"

    result = export_unit_frontmatter_comment_csv([{"id": "u", "content": "---\nx: 1 # note\n---"}], path)

    assert rows(path.read_text())[0]["comment"] == "note"
    assert result["rows_exported"] == 1
