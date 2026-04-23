"""CLI tests for graph path lookup."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from graph.cli.main import app
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


runner = CliRunner()


@pytest.fixture
def graph_db(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node",
            utility_score=0.9,
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="b",
            source_entity_type="knowledge_node",
            title="Node B",
            content="Second node",
            utility_score=0.7,
        )
    )
    c = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="c",
            source_entity_type="insight",
            title="Node C",
            content="Third node",
            content_type=ContentType.INSIGHT,
            utility_score=0.5,
        )
    )
    d = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="d",
            source_entity_type="knowledge_item",
            title="Node D",
            content="Isolated node",
            content_type=ContentType.ARTIFACT,
            utility_score=0.8,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=a.id,
            to_unit_id=b.id,
            relation=EdgeRelation.BUILDS_ON,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=b.id,
            to_unit_id=c.id,
            relation=EdgeRelation.INSPIRES,
        )
    )
    store.close()
    return db_path, {"a": a.id, "b": b.id, "c": c.id, "d": d.id}


def test_path_prints_ordered_hops(monkeypatch: pytest.MonkeyPatch, graph_db: tuple[Path, dict[str, str]]) -> None:
    db_path, ids = graph_db
    monkeypatch.setattr("graph.cli.main._get_store", lambda: Store(str(db_path)))

    result = runner.invoke(app, ["path", ids["a"], ids["c"]])

    assert result.exit_code == 0
    assert "Shortest path (3 nodes):" in result.stdout
    assert f"1. [forty_two] Node A" in result.stdout
    assert f"   ID: {ids['a']}" in result.stdout
    assert f"2. [forty_two] Node B" in result.stdout
    assert f"   ID: {ids['b']}" in result.stdout
    assert f"3. [max] Node C" in result.stdout
    assert f"   ID: {ids['c']}" in result.stdout

    store = Store(str(db_path))
    assert store.count_units() == 4
    assert store.count_units(source_project="forty_two") == 2
    assert len(store.get_all_edges()) == 2
    store.close()


def test_path_reports_disconnected_nodes(monkeypatch: pytest.MonkeyPatch, graph_db: tuple[Path, dict[str, str]]) -> None:
    db_path, ids = graph_db
    monkeypatch.setattr("graph.cli.main._get_store", lambda: Store(str(db_path)))

    result = runner.invoke(app, ["path", ids["a"], ids["d"]])

    assert result.exit_code == 1
    assert f"Error: no path found between {ids['a']} and {ids['d']}." in result.stdout


def test_path_reports_missing_node(monkeypatch: pytest.MonkeyPatch, graph_db: tuple[Path, dict[str, str]]) -> None:
    db_path, ids = graph_db
    monkeypatch.setattr("graph.cli.main._get_store", lambda: Store(str(db_path)))

    missing_id = "missing-node-id"
    result = runner.invoke(app, ["path", ids["a"], missing_id])

    assert result.exit_code == 1
    assert f"Error: unit not found: {missing_id}" in result.stdout
