from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_mermaid_diagram_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_mermaid_diagram_csv_exports_only_mermaid_fences_with_opening_line():
    text = export_units_to_markdown_mermaid_diagram_csv(
        [
            {"id": "b", "title": "Beta", "content": "```python\nprint(1)\n```\n```mermaid\nsequenceDiagram\nA->>B: Hi\n```"},
            {"id": "a", "title": "Alpha", "content": "Intro\n~~~ mermaid title\nflowchart TD\nA-->B\n~~~"},
        ]
    )

    assert _rows(text) == [
        {"unit_id": "a", "title": "Alpha", "line_number": "2", "diagram_type": "flowchart", "statement_count": "2", "first_statement": "flowchart TD"},
        {"unit_id": "b", "title": "Beta", "line_number": "4", "diagram_type": "sequenceDiagram", "statement_count": "2", "first_statement": "sequenceDiagram"},
    ]


def test_mermaid_diagram_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "mermaid.csv"
    units = [{"unit_id": "u1", "content": "```mermaid\nclassDiagram\n```\n```js\nx\n```"}]

    expected = export_units_to_markdown_mermaid_diagram_csv(units)
    stats = export_units_to_markdown_mermaid_diagram_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
