from __future__ import annotations

import json

from graph.adapters.gitlab_pipelines_json import GitlabPipelinesJsonAdapter
from graph.adapters.registry import get_adapter


def test_gitlab_pipelines_json_ingests_wrapped_pipelines(tmp_path):
    path = tmp_path / "pipelines.json"
    path.write_text(json.dumps({"pipelines": [{"id": 99, "iid": 12, "project": {"path_with_namespace": "acme/repo"}, "ref": "main", "sha": "abc", "status": "success", "source": "push", "web_url": "https://gitlab.test/p/99", "duration": 42.5, "queued_duration": 2, "created_at": "2026-05-01T00:00:00Z", "user": {"username": "ada"}}]}), encoding="utf-8")

    unit = GitlabPipelinesJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "gitlab_pipelines_json"
    assert unit.source_id == "gitlab_pipelines_json:acme/repo:99"
    assert unit.source_entity_type == "pipeline"
    assert unit.metadata["project_path"] == "acme/repo"
    assert unit.metadata["duration"] == 42.5
    assert {"gitlab", "pipeline", "success", "main"}.issubset(set(unit.tags))
    assert isinstance(get_adapter("gitlab_pipelines_json"), GitlabPipelinesJsonAdapter)
