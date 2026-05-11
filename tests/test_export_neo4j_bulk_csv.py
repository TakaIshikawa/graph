from __future__ import annotations

import csv
import json
from io import StringIO

from graph.export.neo4j_bulk_csv import export_graph_neo4j_bulk_csv
from graph.types.enums import ContentType, EdgeRelation
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_export_graph_neo4j_bulk_csv_headers_and_serialization():
    units = [
        KnowledgeUnit(
            source_project="steam_library_csv",
            source_id="game:1",
            source_entity_type="game",
            title='Portal, "Two"',
            content="",
            content_type=ContentType.ARTIFACT,
            metadata={"nested": {"b": 2, "a": 1}},
            tags=["puzzle", "steam", "puzzle"],
        )
    ]
    edges = [
        KnowledgeEdge(
            from_unit_id="game:1",
            to_unit_id="game:2",
            relation=EdgeRelation.RELATES_TO,
            weight=0.75,
            metadata={"why": "same series"},
        )
    ]

    exported = export_graph_neo4j_bulk_csv(units, edges)

    assert set(exported) == {"nodes.csv", "relationships.csv"}
    assert exported["nodes.csv"].splitlines()[0] == ":ID,:LABEL,title,content_type,metadata,tags"
    assert exported["relationships.csv"].splitlines()[0] == ":START_ID,:END_ID,:TYPE,weight,metadata"
    node = _rows(exported["nodes.csv"])[0]
    assert node[":ID"] == "game:1"
    assert node[":LABEL"] == "KnowledgeUnit;SteamLibraryCsv;Game"
    assert node["title"] == 'Portal, "Two"'
    assert json.loads(node["metadata"]) == {"nested": {"a": 1, "b": 2}}
    assert json.loads(node["tags"]) == ["puzzle", "steam"]
    rel = _rows(exported["relationships.csv"])[0]
    assert rel[":START_ID"] == "game:1"
    assert rel[":END_ID"] == "game:2"
    assert rel[":TYPE"] == "RELATES_TO"
    assert rel["weight"] == "0.75"
    assert json.loads(rel["metadata"]) == {"why": "same series"}


def test_export_graph_neo4j_bulk_csv_is_deterministic_and_normalizes_relations():
    first = KnowledgeUnit(source_project="x", source_id="b", source_entity_type="note", title="B", content="")
    second = KnowledgeUnit(source_project="x", source_id="a", source_entity_type="note", title="A", content="")
    edge = KnowledgeEdge(from_unit_id="a", to_unit_id="b", relation=EdgeRelation.BUILDS_ON)

    exported = export_graph_neo4j_bulk_csv([first, second], [edge])

    rows = _rows(exported["nodes.csv"])
    assert [row[":ID"] for row in rows] == ["a", "b"]
    assert _rows(exported["relationships.csv"])[0][":TYPE"] == "BUILDS_ON"
