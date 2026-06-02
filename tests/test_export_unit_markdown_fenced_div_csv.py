from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_fenced_div_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_fenced_div_blocks_with_parsed_info():
    content = "::: note {.lead #callout}\nBody\n:::\n::: {.warning .wide #w}\nText\n:::"

    result = rows(export_units_to_markdown_fenced_div_csv([{"id": "u", "title": "T", "source": "docs", "content": content}]))

    assert [(row["unit_id"], row["title"], row["source"], row["line_number"], row["opening_marker"]) for row in result] == [
        ("u", "T", "docs", "1", ":::"),
        ("u", "T", "docs", "4", ":::"),
    ]
    assert result[0]["raw_info"] == "note {.lead #callout}"
    assert result[0]["div_type"] == "note"
    assert result[0]["class_names"] == "lead"
    assert result[0]["id_value"] == "callout"
    assert result[0]["closed"] == "True"
    assert result[1]["div_type"] == ""
    assert result[1]["class_names"] == "warning wide"
    assert result[1]["id_value"] == "w"


def test_distinguishes_unclosed_blocks_and_ignores_non_blocks():
    content = "    ::: note\nText ::: ordinary\n::: tip\n```md\n::: ignored\n```\n"

    result = rows(export_units_to_markdown_fenced_div_csv([{"id": "u", "content": content}]))

    assert [(row["raw_info"], row["line_number"], row["closed"]) for row in result] == [("tip", "3", "False")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "fenced_div.csv"

    result = export_units_to_markdown_fenced_div_csv([{"id": "u", "content": "::: note\n:::"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
