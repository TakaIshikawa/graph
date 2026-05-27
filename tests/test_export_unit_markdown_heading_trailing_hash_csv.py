from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_heading_trailing_hash_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_only_headings_with_valid_closing_markers():
    content = "# Title ##\n## No closer\n### Literal # hash"

    found = rows(export_unit_markdown_heading_trailing_hash_csv([{"id": "u", "content": content}]))

    assert [(row["level"], row["text"], row["closing_hash_count"]) for row in found] == [("1", "Title", "2")]


def test_fenced_code_exclusion_and_path_writing(tmp_path):
    path = tmp_path / "hashes.csv"
    content = "```\n# Skip ##\n```\n## Keep ###"

    result = export_unit_markdown_heading_trailing_hash_csv([{"id": "u", "content": content}], path)

    assert rows(path.read_text())[0]["text"] == "Keep"
    assert result["rows_exported"] == 1
