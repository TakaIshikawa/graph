from graph.store import summarize_unit_mermaid_diagrams


def test_mermaid_diagram_types():
    summary = summarize_unit_mermaid_diagrams([{"content": "```mermaid\nflowchart TD\nA-->B\n```\n```mermaid\nsequenceDiagram\n```\n```mermaid\nclassDiagram\n```"}])
    assert summary["total_diagrams"] == 3
    assert summary["units_with_diagrams"] == 1
    assert summary["diagram_type_counts"] == {"classdiagram": 1, "flowchart": 1, "sequencediagram": 1}


def test_unknown_empty_diagram():
    assert summarize_unit_mermaid_diagrams([{"content": "```mermaid\n\n```"}])["diagram_type_counts"] == {"unknown": 1}


def test_empty_input():
    assert summarize_unit_mermaid_diagrams([]) == {"total_diagrams": 0, "units_with_diagrams": 0, "diagram_type_counts": {}}
