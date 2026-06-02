from __future__ import annotations

import json

from graph.adapters.linear_projects_json import LinearProjectsJsonAdapter
from graph.adapters.registry import get_adapter


def test_linear_projects_json_ingests_projects_wrappers(tmp_path):
    path = tmp_path / "projects.json"
    path.write_text(json.dumps({"data": {"projects": {"nodes": [{"id": "p1", "name": "Importer Pack", "team": {"key": "ENG"}, "state": "started", "lead": {"name": "Ada"}, "description": "Build adapters", "url": "https://linear.test/p1", "targetDate": "2026-06-01", "progress": 0.5, "issueCount": 8}]}}}), encoding="utf-8")

    unit = LinearProjectsJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "linear_projects_json"
    assert unit.source_id == "linear_projects_json:p1"
    assert unit.source_entity_type == "project"
    assert unit.metadata["team"] == "ENG"
    assert unit.metadata["state"] == "started"
    assert unit.metadata["issue_count"] == 8
    assert {"linear", "project", "ENG", "started"}.issubset(set(unit.tags))
    assert isinstance(get_adapter("linear-projects-json"), LinearProjectsJsonAdapter)
