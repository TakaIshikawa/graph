"""Focused tests for the MCP relation motif analysis tool."""

from __future__ import annotations

import asyncio
import json

import pytest

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
    )


def _populate_relation_motif_graph(store: Store) -> dict[str, str]:
    ids: dict[str, str] = {}
    for unit_id, title in [
        ("unit-a", "Alpha"),
        ("unit-b", "Beta"),
        ("unit-c", "Gamma"),
        ("unit-d", "Delta"),
        ("unit-e", "Epsilon"),
        ("unit-f", "Zeta"),
        ("unit-g", "Eta"),
        ("unit-h", "Theta"),
        ("unit-i", "Iota"),
    ]:
        ids[unit_id] = store.insert_unit(_unit(unit_id, title)).id

    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        _edge("edge-b-c", "unit-b", "unit-c", EdgeRelation.CHALLENGES),
        _edge("edge-d-e", "unit-d", "unit-e", EdgeRelation.BUILDS_ON),
        _edge("edge-e-f", "unit-e", "unit-f", EdgeRelation.CHALLENGES),
        _edge("edge-g-h", "unit-g", "unit-h", EdgeRelation.DISCOVERS),
        _edge("edge-h-i", "unit-h", "unit-i", EdgeRelation.INSPIRES),
    ]:
        store.insert_edge(edge)
    return ids


@pytest.fixture
def relation_motif_store(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    ids = _populate_relation_motif_graph(store)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    return ids


def test_relation_motif_tool_is_registered_with_schema(relation_motif_store):
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "analyze_relation_motifs")
    properties = tool.inputSchema["properties"]

    assert set(properties) >= {
        "path_length",
        "limit",
        "min_count",
        "relation_types",
        "relation_sequence",
    }
    assert properties["path_length"]["enum"] == [2]
    assert properties["limit"]["minimum"] == 0
    assert properties["min_count"]["minimum"] == 1
    assert properties["relation_types"]["items"]["enum"] == [
        relation.value for relation in EdgeRelation
    ]


def test_relation_motif_tool_returns_json_serializable_motifs(relation_motif_store):
    response = asyncio.run(
        mcp_server.call_tool("analyze_relation_motifs", {"limit": 5})
    )
    payload = json.loads(response[0].text)

    assert payload["path_length"] == 2
    assert payload["limit"] == 5
    assert payload["min_count"] == 1
    assert payload["motif_count"] == 2
    assert payload["stats"] == {
        "total_paths": 3,
        "unique_motifs": 2,
        "matching_motifs": 2,
        "returned_motifs": 2,
    }
    assert [
        (motif["relation_sequence"], motif["count"])
        for motif in payload["motifs"]
    ] == [
        (["builds_on", "challenges"], 2),
        (["discovers", "inspires"], 1),
    ]

    example = payload["motifs"][0]["examples"][0]
    assert example["unit_ids"] == ["unit-a", "unit-b", "unit-c"]
    assert example["edge_ids"] == ["edge-a-b", "edge-b-c"]
    assert example["relations"] == ["builds_on", "challenges"]
    assert [unit["title"] for unit in example["units"]] == [
        "Alpha",
        "Beta",
        "Gamma",
    ]
    assert example["edges"][0]["relation"] == "builds_on"
    assert example["edges"][0]["from_unit_id"] == "unit-a"


def test_relation_motif_tool_honors_limit_after_relation_filter(
    relation_motif_store,
):
    response = asyncio.run(
        mcp_server.call_tool(
            "analyze_relation_motifs",
            {
                "limit": 1,
                "relation_sequence": ["discovers", "inspires"],
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["limit"] == 1
    assert payload["motif_count"] == 1
    assert payload["filters"] == {
        "relation_types": [],
        "relation_sequence": ["discovers", "inspires"],
    }
    assert payload["stats"] == {
        "total_paths": 3,
        "unique_motifs": 2,
        "matching_motifs": 1,
        "returned_motifs": 1,
    }
    assert payload["motifs"][0]["relation_sequence"] == ["discovers", "inspires"]


def test_relation_motif_tool_empty_store_returns_empty_payload(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    Store(str(db_path)).close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("analyze_relation_motifs", {}))
    payload = json.loads(response[0].text)

    assert payload == {
        "path_length": 2,
        "limit": 20,
        "min_count": 1,
        "motif_count": 0,
        "motifs": [],
        "stats": {
            "total_paths": 0,
            "unique_motifs": 0,
            "matching_motifs": 0,
            "returned_motifs": 0,
        },
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"limit": -1}, "limit must be a non-negative integer."),
        ({"min_count": 0}, "min_count must be a positive integer."),
        (
            {"relation_types": ["unknown"]},
            "Invalid relation type(s): unknown",
        ),
        (
            {"relation_sequence": ["builds_on"]},
            "relation_sequence length must match path_length",
        ),
    ],
)
def test_relation_motif_tool_serializes_invalid_arguments(
    relation_motif_store,
    arguments,
    message,
):
    response = asyncio.run(mcp_server.call_tool("analyze_relation_motifs", arguments))
    payload = json.loads(response[0].text)

    assert payload["error"] == "invalid_relation_motifs_request"
    assert payload["message"] == message
    assert payload["arguments"]["limit"] == arguments.get("limit", 20)
