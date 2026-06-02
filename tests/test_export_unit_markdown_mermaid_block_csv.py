from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_mermaid_block_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_mermaid_block_csv_exports_only_mermaid_blocks():
    text = export_units_to_markdown_mermaid_block_csv(
        [
            {"id": "a", "title": "Alpha", "content": "```mermaid\ngraph TD\nA-->B\n```\n```python\npass\n```"},
            {"id": "b", "title": "Beta", "content": "~~~ mermaid\n\nsequenceDiagram\n~~~"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "start_line": "1", "end_line": "4", "diagram_type": "graph", "line_count": "2"},
        {"unit_id": "b", "title": "Beta", "start_line": "1", "end_line": "4", "diagram_type": "sequencediagram", "line_count": "2"},
    ]


def test_markdown_mermaid_block_csv_tolerates_blank_body_and_path_mode(tmp_path):
    path = tmp_path / "mermaid.csv"
    units = [{"id": "a", "content": "```mermaid\n```"}]

    stats = export_units_to_markdown_mermaid_block_csv(units, path)

    assert rows(path.read_text(encoding="utf-8"))[0]["diagram_type"] == ""
    assert stats["rows_exported"] == 1
