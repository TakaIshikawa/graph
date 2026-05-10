from __future__ import annotations

import json
from pathlib import Path

from graph.adapters.are_na import AreNaAdapter
from graph.types.enums import ContentType, EdgeRelation, SourceProject


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_parses_channel_with_blocks(tmp_path):
    data = {
        "title": "Design Inspiration",
        "id": "ch1",
        "contents": [
            {"id": "b1", "title": "Cool image", "content": "A nice design", "class": "Image"},
            {"id": "b2", "title": "Article link", "content": "Interesting read", "class": "Link",
             "source": {"url": "https://example.com"}},
        ],
    }
    _write_json(tmp_path / "arena.json", data)

    result = AreNaAdapter(path=str(tmp_path / "arena.json")).ingest()

    channels = [u for u in result.units if u.source_entity_type == "channel"]
    blocks = [u for u in result.units if u.source_entity_type == "block"]
    assert len(channels) == 1
    assert channels[0].title == "Design Inspiration"
    assert len(blocks) == 2
    for b in blocks:
        assert b.source_project == SourceProject.ARE_NA


def test_blocks_inherit_channel_tag(tmp_path):
    data = {"title": "Research", "id": "ch1", "contents": [
        {"id": "b1", "title": "Note", "content": "text"},
    ]}
    _write_json(tmp_path / "arena.json", data)

    result = AreNaAdapter(path=str(tmp_path / "arena.json")).ingest()
    block = [u for u in result.units if u.source_entity_type == "block"][0]
    assert "research" in block.tags


def test_channel_membership_edges(tmp_path):
    data = {"title": "Art", "id": "ch1", "contents": [
        {"id": "b1", "title": "Painting", "content": "oil"},
        {"id": "b2", "title": "Sculpture", "content": "marble"},
    ]}
    _write_json(tmp_path / "arena.json", data)

    result = AreNaAdapter(path=str(tmp_path / "arena.json")).ingest()
    contains = [e for e in result.edges if e.relation == EdgeRelation.CONTAINS]
    assert len(contains) == 2


def test_empty_export(tmp_path):
    _write_json(tmp_path / "empty.json", {"title": "Empty", "id": "ch1", "contents": []})
    result = AreNaAdapter(path=str(tmp_path / "empty.json")).ingest()
    channels = [u for u in result.units if u.source_entity_type == "channel"]
    blocks = [u for u in result.units if u.source_entity_type == "block"]
    assert len(channels) == 1
    assert len(blocks) == 0


def test_no_path_returns_empty():
    result = AreNaAdapter().ingest()
    assert len(result.units) == 0


def test_entity_type_filtering(tmp_path):
    data = {"title": "Test", "id": "ch1", "contents": [
        {"id": "b1", "title": "Block", "content": "x"},
    ]}
    _write_json(tmp_path / "arena.json", data)

    result = AreNaAdapter(path=str(tmp_path / "arena.json")).ingest(entity_types=["block"])
    assert all(u.source_entity_type == "block" for u in result.units)


def test_registry_lookup():
    from graph.adapters.registry import get_adapter

    adapter = get_adapter("are_na", path="/tmp/fake")
    assert adapter.name == "are_na"


def test_block_metadata_has_source_url(tmp_path):
    data = {"title": "Links", "id": "ch1", "contents": [
        {"id": "b1", "title": "Link", "content": "x", "class": "Link",
         "source": {"url": "https://example.com/page"}},
    ]}
    _write_json(tmp_path / "arena.json", data)

    result = AreNaAdapter(path=str(tmp_path / "arena.json")).ingest()
    block = [u for u in result.units if u.source_entity_type == "block"][0]
    assert block.metadata["source_url"] == "https://example.com/page"
    assert block.metadata["block_type"] == "Link"


def test_directory_with_multiple_files(tmp_path):
    _write_json(tmp_path / "a.json", {"title": "A", "id": "ch1", "contents": [
        {"id": "b1", "title": "Block A", "content": "a"},
    ]})
    _write_json(tmp_path / "b.json", {"title": "B", "id": "ch2", "contents": [
        {"id": "b2", "title": "Block B", "content": "b"},
    ]})

    result = AreNaAdapter(path=str(tmp_path)).ingest()
    assert len([u for u in result.units if u.source_entity_type == "channel"]) == 2
    assert len([u for u in result.units if u.source_entity_type == "block"]) == 2
