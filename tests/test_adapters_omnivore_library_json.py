from __future__ import annotations

import json

from graph.adapters import OmnivoreLibraryJsonAdapter


def test_omnivore_library_json_adapter_accepts_top_level_arrays(tmp_path):
    path = tmp_path / "omnivore.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "title": "One",
                    "url": "https://one.test",
                    "labels": [{"name": "Read Later"}, "AI"],
                    "highlights": [{"quote": "Important quote"}],
                    "savedAt": "2026-05-30T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    adapter = OmnivoreLibraryJsonAdapter(str(path))
    result = adapter.ingest()

    assert adapter.name == "omnivore_library_json"
    assert "bookmark" in adapter.entity_types
    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_entity_type == "bookmark"
    assert unit.tags == ["read later", "ai"]
    assert "Important quote" in unit.content
    assert unit.metadata["highlights"] == ["Important quote"]


def test_omnivore_library_json_adapter_accepts_nested_items_and_nodes(tmp_path):
    path = tmp_path / "omnivore.json"
    path.write_text(
        json.dumps(
            {
                "data": {
                    "nodes": [
                        {
                            "node": {
                                "slug": "nested",
                                "title": "Nested",
                                "url": "https://nested.test",
                                "description": "Saved page",
                                "labels": [{"name": "Research"}],
                                "createdAt": "2026-05-29T00:00:00Z",
                            }
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = OmnivoreLibraryJsonAdapter(str(path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id.endswith(":nested")
    assert result.units[0].tags == ["research"]
    assert "Saved page" in result.units[0].content
