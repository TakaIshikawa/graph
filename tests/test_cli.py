"""CLI tests for graph commands."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
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


class MockEmbeddingProvider:
    """Mock provider that returns deterministic embeddings based on content."""

    def embed(self, text: str) -> list[float]:
        words = text.lower().split()
        vec = [0.0] * 8
        for w in words:
            h = hash(w) % 8
            vec[h] += 0.1
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


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


def _populate_search_graph(store: Store) -> None:
    from graph.rag.search import RAGService

    provider = MockEmbeddingProvider()
    rag_service = RAGService(store, provider)

    units = [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="approved",
            source_entity_type="insight",
            title="Solar approved insight",
            content="Solar energy storage market growth",
            content_type=ContentType.INSIGHT,
            metadata={"review_state": "approved"},
            tags=["energy", "solar"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="rejected",
            source_entity_type="insight",
            title="Solar rejected insight",
            content="Solar energy storage market risk",
            content_type=ContentType.INSIGHT,
            metadata={"review_state": "rejected"},
            tags=["energy", "solar"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="wrong-tag",
            source_entity_type="insight",
            title="Solar research insight",
            content="Solar energy storage market research",
            content_type=ContentType.INSIGHT,
            metadata={"review_state": "approved"},
            tags=["research", "solar"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="wrong-type",
            source_entity_type="design_brief",
            title="Solar approved brief",
            content="Solar energy storage market brief",
            content_type=ContentType.DESIGN_BRIEF,
            metadata={"review_state": "approved"},
            tags=["energy", "solar"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="other-project",
            source_entity_type="knowledge_node",
            title="Solar forty two note",
            content="Solar energy storage market note",
            content_type=ContentType.FINDING,
            tags=["energy", "solar"],
        ),
    ]

    ids = []
    for unit in units:
        inserted = store.insert_unit(unit)
        store.fts_index_unit(inserted)
        ids.append(inserted.id)

    rag_service.embed_batch_and_store(ids)


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


def test_search_command_preserves_default_format(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.rag.embeddings.get_embedding_provider", lambda *args, **kwargs: MockEmbeddingProvider())

    try:
        result = runner.invoke(app, ["search", "solar", "--mode", "fulltext", "--limit", "1"])

        assert result.exit_code == 0
        assert "[max] Solar approved insight" in result.output
        assert "ID:" in result.output
        assert "Type:" in result.output
        assert "Tags:" in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_bridges_command_prints_readable_units_and_rebuilds(monkeypatch):
    from graph.graph.service import GraphService

    store = _make_store()
    _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    original_rebuild = GraphService.rebuild
    calls = []

    def tracking_rebuild(self):
        calls.append(True)
        return original_rebuild(self)

    monkeypatch.setattr(GraphService, "rebuild", tracking_rebuild)

    try:
        result = runner.invoke(app, ["bridges", "--limit", "2"])

        assert result.exit_code == 0
        assert calls == [True]
        assert "[forty_two] Node B" in result.output
        assert "betweenness:" in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_cross_project_command_summarizes_project_pairs(monkeypatch):
    from graph.graph.service import GraphService

    store = _make_store()
    _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    original_rebuild = GraphService.rebuild
    calls = []

    def tracking_rebuild(self):
        calls.append(True)
        return original_rebuild(self)

    monkeypatch.setattr(GraphService, "rebuild", tracking_rebuild)

    try:
        result = runner.invoke(app, ["cross-project"])

        assert result.exit_code == 0
        assert calls == [True]
        assert "Cross-project connections:" in result.output
        assert "forty_two <-> max: 1 edges" in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode", ["fulltext", "semantic", "hybrid"])
def test_search_command_applies_filters_in_all_modes(monkeypatch, mode):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.rag.embeddings.get_embedding_provider", lambda *args, **kwargs: MockEmbeddingProvider())

    try:
        result = runner.invoke(
            app,
            [
                "search",
                "solar",
                "--mode",
                mode,
                "--limit",
                "10",
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--tag",
                "energy",
                "--review-state",
                "approved",
            ],
        )

        assert result.exit_code == 0
        assert "[max] Solar approved insight" in result.output
        assert "Solar rejected insight" not in result.output
        assert "Solar research insight" not in result.output
        assert "Solar approved brief" not in result.output
        assert "Solar forty two note" not in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]
