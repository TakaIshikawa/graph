from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_mention_handle_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_mentions_with_context_and_position():
    result = rows(export_units_to_markdown_mention_handle_csv([{"id": "u", "title": "T", "source": "s", "content": "Ask @Alice and @team_ops."}]))

    assert [(row["raw_handle"], row["normalized_handle"], row["position"]) for row in result] == [
        ("@Alice", "@alice", "5"),
        ("@team_ops", "@team_ops", "16"),
    ]
    assert result[0]["context"] == "Ask @Alice and @team_ops."
    assert result[0]["source"] == "s"


def test_ignores_email_url_userinfo_code_and_fences():
    content = "me@example.com http://u@example.com/path `@code` @real\n```md\n@skip\n```"

    result = rows(export_units_to_markdown_mention_handle_csv([{"id": "u", "content": content}]))

    assert [(row["raw_handle"], row["line_number"]) for row in result] == [("@real", "1")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "mentions.csv"

    result = export_units_to_markdown_mention_handle_csv([{"id": "u", "content": "@one"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
