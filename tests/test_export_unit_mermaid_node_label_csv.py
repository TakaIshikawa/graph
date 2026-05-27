from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_mermaid_node_label_csv import export_units_to_mermaid_node_label_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_mermaid_node_label_csv_extracts_mermaid_nodes_only():
    csv_text = export_units_to_mermaid_node_label_csv(
        [
            {
                "id": "u1",
                "content": "```python\nA[Ignore]\n```\n```mermaid\ngraph TD\nA[Start] --> B(Next)\nB --> C{Decision}\nD[\"Quoted\"]\n```",
            }
        ]
    )

    assert [(row["node_id"], row["label"], row["shape_hint"]) for row in _rows(csv_text)] == [
        ("A", "Start", "bracket"),
        ("B", "Next", "paren"),
        ("C", "Decision", "brace"),
        ("D", "Quoted", "quoted"),
    ]


def test_export_units_to_mermaid_node_label_csv_deduplicates_node_ids_per_diagram():
    rows = _rows(export_units_to_mermaid_node_label_csv([{"id": "u1", "content": "```mermaid\nA[First]\nA[Second]\n```"}]))

    assert [(row["node_id"], row["label"], row["fence_start_line"]) for row in rows] == [("A", "First", "1")]
