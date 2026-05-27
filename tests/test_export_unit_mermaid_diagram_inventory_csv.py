from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_mermaid_diagram_inventory_csv import export_units_to_mermaid_diagram_inventory_csv


def test_export_units_to_mermaid_diagram_inventory_csv_detects_multiple_diagrams():
    rows = list(csv.DictReader(StringIO(export_units_to_mermaid_diagram_inventory_csv([{"id": "u1", "content": "```mermaid title=x\nflowchart TD\nA-->B\n```\n~~~mermaid\nsequenceDiagram\nA->>B: hi\n~~~"}]))))

    assert [(row["diagram_type"], row["line_count"], row["starts_at_line"]) for row in rows] == [
        ("flowchart", "2", "1"),
        ("sequenceDiagram", "2", "5"),
    ]
