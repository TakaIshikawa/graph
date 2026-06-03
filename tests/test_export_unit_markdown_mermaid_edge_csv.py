from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_mermaid_edge_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_mermaid_edge_csv_exports_multiple_edges_with_labels_and_arrows():
    text = export_units_to_markdown_mermaid_edge_csv(
        [
            {
                "id": "u",
                "title": "Unit",
                "source": "notes.md",
                "content": "```mermaid\nflowchart TD\nA -->|label| B\nB --- C\nC -.-> D\nD ==> E\n```",
            }
        ]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Unit", "source": "notes.md", "line_number": "3", "diagram_type": "flowchart", "from_node": "A", "arrow": "-->", "to_node": "B", "edge_label": "label"},
        {"unit_id": "u", "title": "Unit", "source": "notes.md", "line_number": "4", "diagram_type": "flowchart", "from_node": "B", "arrow": "---", "to_node": "C", "edge_label": ""},
        {"unit_id": "u", "title": "Unit", "source": "notes.md", "line_number": "5", "diagram_type": "flowchart", "from_node": "C", "arrow": "-.->", "to_node": "D", "edge_label": ""},
        {"unit_id": "u", "title": "Unit", "source": "notes.md", "line_number": "6", "diagram_type": "flowchart", "from_node": "D", "arrow": "==>", "to_node": "E", "edge_label": ""},
    ]


def test_mermaid_edge_csv_ignores_non_mermaid_fences_and_malformed_lines_and_sorts():
    text = export_units_to_markdown_mermaid_edge_csv(
        [
            {"id": "z", "title": "Zed", "content": "```python\nA --> B\n```\n```mermaid\nflowchart LR\nnot an edge\nZ --> A\n```"},
            {"id": "a", "metadata": {"title": "Alpha", "source_id": "alpha.md"}, "content": "~~~ mermaid\nsequenceDiagram\nClient --> Server\n~~~"},
        ]
    )

    assert _rows(text) == [
        {"unit_id": "a", "title": "Alpha", "source": "alpha.md", "line_number": "3", "diagram_type": "sequenceDiagram", "from_node": "Client", "arrow": "-->", "to_node": "Server", "edge_label": ""},
        {"unit_id": "z", "title": "Zed", "source": "", "line_number": "7", "diagram_type": "flowchart", "from_node": "Z", "arrow": "-->", "to_node": "A", "edge_label": ""},
    ]


def test_mermaid_edge_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "edges.csv"
    units = [{"unit_id": "u1", "content": "```mermaid\ngraph TD\nA --> B\n```"}]

    expected = export_units_to_markdown_mermaid_edge_csv(units)
    stats = export_units_to_markdown_mermaid_edge_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
