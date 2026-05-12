from __future__ import annotations

import json

from graph.adapters.linear_issues_json import LinearIssuesJsonAdapter
from graph.adapters.registry import get_adapter


def test_linear_issues_json_ingests_wrapped_export_and_relationships(tmp_path):
    export = tmp_path / "linear.json"
    export.write_text(json.dumps({"issues": [{"id": "p", "identifier": "LIN-1", "title": "Parent", "state": {"name": "Open"}, "labels": [{"name": "import"}], "createdAt": "2026-05-01T00:00:00Z"}, {"id": "c", "identifier": "LIN-2", "title": "Child", "parent": {"id": "p"}, "relatedIssueIds": ["p"], "createdAt": "2026-05-02T00:00:00Z"}]}), encoding="utf-8")

    result = LinearIssuesJsonAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == ["linear_issues_json:c", "linear_issues_json:p"]
    assert result.units[1].metadata["labels"] == ["import"]
    assert {edge.metadata["kind"] for edge in result.edges} == {"parent", "related"}
    assert get_adapter("linear_issues_json", path=str(export)).name == "linear_issues_json"
