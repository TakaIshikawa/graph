from __future__ import annotations

import json

from graph.adapters.asana_stories_json import AsanaStoriesJsonAdapter
from graph.adapters.registry import get_adapter


def test_asana_stories_json_ingests_wrapped_stories(tmp_path):
    path = tmp_path / "stories.json"
    path.write_text(json.dumps({"data": [{"gid": "s1", "task": {"gid": "t1", "name": "Import task"}, "project": "Graph", "created_by": {"name": "Ada"}, "created_at": "2026-05-01T10:00:00Z", "resource_subtype": "comment_added", "text": "Done", "liked": True, "permalink_url": "https://asana.test/s1"}]}), encoding="utf-8")

    unit = AsanaStoriesJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "asana_stories_json"
    assert unit.source_id == "asana_stories_json:s1"
    assert unit.source_entity_type == "story"
    assert unit.metadata["task_gid"] == "t1"
    assert unit.metadata["creator"] == "Ada"
    assert unit.metadata["resource_subtype"] == "comment_added"
    assert unit.metadata["liked"] is True
    assert isinstance(get_adapter("asana_stories_json"), AsanaStoriesJsonAdapter)
