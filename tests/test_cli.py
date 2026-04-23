"""CLI tests for graph commands."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from graph.cli.main import app
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


runner = CliRunner()


class StoreProxy:
    def __init__(self, store: Store) -> None:
        self._store = store

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    def close(self) -> None:
        return None


def _make_store() -> Store:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = Store(path)
    store._test_db_path = path  # type: ignore[attr-defined]
    return store


def _populate_graph(store: Store) -> tuple[str, str, str, str]:
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
    return a.id, b.id, c.id, d.id


def _cleanup_db(path: str) -> None:
    db_path = Path(path)
    for candidate in (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def test_shortest_path_command_prints_readable_path(monkeypatch):
    store = _make_store()
    a_id, _, c_id, _ = _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["shortest-path", a_id, c_id])

        assert result.exit_code == 0
        assert "Shortest path (3 nodes):" in result.output
        assert "[forty_two] Node A" in result.output
        assert "[forty_two] Node B" in result.output
        assert "[max] Node C" in result.output
        assert "ID:" in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_shortest_path_command_reports_missing_units(monkeypatch):
    store = _make_store()
    _, _, c_id, _ = _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["shortest-path", "missing", c_id])

        assert result.exit_code == 0
        assert "Error: source unit not found: missing." in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_shortest_path_command_reports_no_path(monkeypatch):
    store = _make_store()
    a_id, _, _, d_id = _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["shortest-path", a_id, d_id])

        assert result.exit_code == 0
        assert "No path found between the selected units." in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]
