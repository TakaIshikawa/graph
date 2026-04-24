"""CLI tests for graph commands."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graph.cli.main import app
from graph.rag.embeddings import serialize_embedding
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


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
            h = sum(ord(c) for c in w) % 8
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
            utility_score=0.92,
            created_at=datetime.fromisoformat("2026-04-22T00:00:00+00:00"),
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
            utility_score=0.4,
            created_at=datetime.fromisoformat("2026-04-23T00:00:00+00:00"),
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
            utility_score=0.65,
            created_at=datetime.fromisoformat("2026-04-20T00:00:00+00:00"),
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
            utility_score=0.8,
            created_at=datetime.fromisoformat("2026-04-24T00:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="other-project",
            source_entity_type="knowledge_node",
            title="Solar forty two note",
            content="Solar energy storage market note",
            content_type=ContentType.FINDING,
            tags=["energy", "solar"],
            utility_score=0.95,
            created_at=datetime.fromisoformat("2026-04-21T00:00:00+00:00"),
        ),
    ]

    ids = []
    for unit in units:
        inserted = store.insert_unit(unit)
        store.fts_index_unit(inserted)
        ids.append(inserted.id)

    rag_service.embed_batch_and_store(ids)


def _populate_design_briefs(store: Store) -> None:
    briefs = [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="brief-1",
            source_entity_type="design_brief",
            title="Ops workflow brief",
            content="Brief content one",
            content_type=ContentType.DESIGN_BRIEF,
            metadata={
                "domain": "devtools",
                "theme": "workflow",
                "readiness_score": 82,
                "lead_idea_id": "idea-lead-1",
                "source_idea_ids": ["idea-a", "idea-b"],
                "design_status": "draft",
                "validation_plan": "Interview 5 ops teams and validate workflow fit.",
                "first_milestones": ["Draft interview guide", "Run first pilot"],
            },
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="brief-2",
            source_entity_type="design_brief",
            title="Platform expansion brief",
            content="Brief content two",
            content_type=ContentType.DESIGN_BRIEF,
            metadata={
                "domain": "platform",
                "theme": "scale",
                "readiness_score": 64,
                "lead_idea_id": "idea-lead-2",
                "source_idea_ids": ["idea-c"],
                "design_status": "review",
                "validation_plan": "Validate platform constraints with infrastructure stakeholders.",
                "first_milestones": ["Map constraints", "Propose rollout"],
            },
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="brief-3",
            source_entity_type="design_brief",
            title="Ignored external brief",
            content="Should not appear",
            content_type=ContentType.DESIGN_BRIEF,
            metadata={
                "domain": "devtools",
                "theme": "workflow",
                "readiness_score": 99,
                "lead_idea_id": "idea-x",
                "source_idea_ids": ["idea-y"],
                "design_status": "draft",
            },
        ),
    ]

    for brief in briefs:
        store.insert_unit(brief)


def _populate_tags_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="solar-storage",
            source_entity_type="insight",
            title="Solar storage",
            content="Solar storage insight",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar", "storage"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="solar-grid",
            source_entity_type="knowledge_node",
            title="Solar grid",
            content="Solar grid finding",
            content_type=ContentType.FINDING,
            tags=["energy", "solar", "grid"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="battery-storage",
            source_entity_type="insight",
            title="Battery storage",
            content="Battery storage insight",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage", "battery"],
        ),
    ]:
        store.insert_unit(unit)


def _populate_duplicates_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="same-title-a",
            source_entity_type="insight",
            title="Repeated Import",
            content="First independent note about sales workflow cleanup.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="same-title-b",
            source_entity_type="insight",
            title=" repeated   import ",
            content="Second independent note about onboarding research.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="same-content-a",
            source_entity_type="insight",
            title="Storage market signal",
            content="Solar storage adoption is accelerating across midmarket operations teams this quarter.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="same-content-b",
            source_entity_type="insight",
            title="Battery adoption memo",
            content="Solar storage adoption is accelerating across midmarket operations teams this quarter rapidly.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="filtered-title",
            source_entity_type="knowledge_node",
            title="Repeated Import",
            content="Outside project duplicate title.",
            content_type=ContentType.FINDING,
        ),
    ]:
        store.insert_unit(unit)


def _populate_links_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="markdown-link",
            source_entity_type="insight",
            title="Markdown citation",
            content="Read [docs](https://Example.com/docs).",
            content_type=ContentType.INSIGHT,
            metadata={"source_url": "https://meta.test/report)."},
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="duplicate-link",
            source_entity_type="knowledge_node",
            title="Repeated citation",
            content="Repeated https://example.com/docs",
            content_type=ContentType.FINDING,
        ),
    ]:
        store.insert_unit(unit)


def _cleanup_db(path: str) -> None:
    db_path = Path(path)
    for candidate in (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def test_export_obsidian_uses_configured_vault_default(tmp_path, monkeypatch):
    store = _make_store()
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node",
        )
    )
    proxy = StoreProxy(store)
    vault_path = tmp_path / "configured-vault"

    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.obsidian_vault_path", str(vault_path))

    result = runner.invoke(app, ["export-obsidian"])

    assert result.exit_code == 0
    assert (vault_path / "Graph" / "forty_two" / "Node A.md").exists()
    assert f"to {vault_path / 'Graph'}" in result.output
    _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_json_export_and_import_commands_round_trip(tmp_path, monkeypatch):
    source = _make_store()
    a_id, _, _, _ = _populate_graph(source)
    export_path = tmp_path / "graph-backup.json"

    source_proxy = StoreProxy(source)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: source_proxy)
    export_result = runner.invoke(app, ["export-json", str(export_path)])

    assert export_result.exit_code == 0
    assert export_path.exists()
    exported = json.loads(export_path.read_text())
    assert exported["schema_version"] == 1
    assert exported["exported_at"]
    assert len(exported["units"]) == 4
    assert len(exported["edges"]) == 2
    assert "Exported 4 units and 2 edges" in export_result.output

    target = _make_store()
    target_proxy = StoreProxy(target)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: target_proxy)
    import_result = runner.invoke(app, ["import-json", str(export_path)])

    try:
        assert import_result.exit_code == 0
        assert "Imported 4 units, updated 0 units, inserted 2 edges" in import_result.output
        assert target.count_units() == 4
        assert len(target.get_all_edges()) == 2
        assert target.get_unit(a_id) is not None
        assert target.fts_search("First")

        second_result = runner.invoke(app, ["import-json", str(export_path)])

        assert second_result.exit_code == 0
        assert "Imported 0 units, updated 4 units, inserted 0 edges" in second_result.output
        assert target.count_units() == 4
        assert len(target.get_all_edges()) == 2
    finally:
        source.close()
        target.close()
        _cleanup_db(source._test_db_path)  # type: ignore[attr-defined]
        _cleanup_db(target._test_db_path)  # type: ignore[attr-defined]


def test_export_graphml_command_writes_valid_graphml(tmp_path, monkeypatch):
    store = _make_store()
    a_id, b_id, _, _ = _populate_graph(store)
    export_path = tmp_path / "graph.graphml"

    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    result = runner.invoke(app, ["export-graphml", str(export_path)])

    try:
        assert result.exit_code == 0
        assert export_path.exists()
        assert "Exported 4 nodes and 2 edges" in result.output

        exported = nx.read_graphml(export_path)
        assert exported.nodes[a_id]["title"] == "Node A"
        assert exported.nodes[a_id]["source_project"] == "forty_two"
        assert exported.nodes[a_id]["source_entity_type"] == "knowledge_node"
        assert exported.nodes[a_id]["created_at"]
        assert exported.get_edge_data(a_id, b_id)["relation"] == "builds_on"
        assert exported.get_edge_data(a_id, b_id)["source"] == "inferred"
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_turtle_command_writes_turtle_with_counts(tmp_path, monkeypatch):
    store = _make_store()
    a_id, b_id, _, _ = _populate_graph(store)
    export_path = tmp_path / "graph.ttl"

    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    result = runner.invoke(
        app,
        [
            "export-turtle",
            str(export_path),
            "--base-uri",
            "https://example.test/unit/",
        ],
    )

    try:
        assert result.exit_code == 0
        assert export_path.exists()
        assert "Exported 4 nodes and 2 edges" in result.output

        text = export_path.read_text()
        assert "@prefix graph: <https://graph.local/schema#> ." in text
        assert f"<https://example.test/unit/{a_id}>" in text
        assert 'graph:title "Node A"' in text
        assert 'graph:tag "energy"' not in text
        assert f"graph:builds_on <https://example.test/unit/{b_id}>" in text
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_neighborhood_command_writes_local_json_and_caps_depth(
    tmp_path, monkeypatch
):
    store = _make_store()
    a_id, _, _, d_id = _populate_graph(store)
    export_path = tmp_path / "neighborhood.json"

    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    result = runner.invoke(
        app,
        ["export-neighborhood", a_id, str(export_path), "--depth", "9"],
    )

    try:
        assert result.exit_code == 0
        assert "Exported 3 units and 2 edges" in result.output
        assert "(depth 3)" in result.output
        payload = json.loads(export_path.read_text())
        assert payload["center"]["id"] == a_id
        assert payload["depth"] == 3
        assert len(payload["units"]) == 3
        assert len(payload["edges"]) == 2
        assert d_id not in {unit["id"] for unit in payload["units"]}
        assert {edge["relation"] for edge in payload["edges"]} == {
            "builds_on",
            "inspires",
        }
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_neighborhood_command_reports_missing_unit(tmp_path, monkeypatch):
    store = _make_store()
    export_path = tmp_path / "missing.json"
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            ["export-neighborhood", "missing", str(export_path)],
        )
        assert result.exit_code == 1
        assert "Unit not found: missing" in result.output
        assert not export_path.exists()

        json_result = runner.invoke(
            app,
            ["export-neighborhood", "missing", str(export_path), "--json"],
        )
        payload = json.loads(json_result.output)
        assert json_result.exit_code == 1
        assert payload["error"] == "unit_not_found"
        assert payload["unit_id"] == "missing"
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_report_command_writes_markdown_and_json_counts(tmp_path, monkeypatch):
    store = _make_store()
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="report-a",
            source_entity_type="knowledge_node",
            title="Report Node A",
            content="First report node",
            tags=["energy", "solar"],
            utility_score=0.9,
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="report-b",
            source_entity_type="insight",
            title="Report Node B",
            content="Second report node",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage"],
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
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    report_path = tmp_path / "graph-report.md"

    try:
        result = runner.invoke(
            app,
            ["export-report", str(report_path), "--limit", "2", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["path"] == str(report_path)
        assert payload["section_counts"]["central"] == 2
        assert payload["section_counts"]["tags"] == 2
        assert payload["section_counts"]["cross_project"] == 1

        report = report_path.read_text()
        for heading in [
            "## Stats",
            "## Top Central Nodes",
            "## Bridge Nodes",
            "## Largest Clusters",
            "## Gap Candidates",
            "## Top Tags",
            "## Cross-Project Connections",
        ]:
            assert heading in report
        assert "Report Node A" in report
        assert "Report Node B" in report
        assert "energy: 2 units" in report
        assert "forty_two <-> max: 1 edges" in report
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_report_command_handles_empty_graph(tmp_path, monkeypatch):
    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    report_path = tmp_path / "empty-report.md"

    try:
        result = runner.invoke(app, ["export-report", str(report_path), "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["section_counts"] == {
            "stats": 1,
            "central": 0,
            "bridges": 0,
            "clusters": 0,
            "gaps": 0,
            "tags": 0,
            "cross_project": 0,
        }
        report = report_path.read_text()
        assert "Nodes: 0" in report
        assert "Edges: 0" in report
        assert "## Top Tags" in report
        assert "_None._" in report
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_ingest_markdown_command_uses_configured_root(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "Alpha.md").write_text("Alpha links to [[Beta]] #cli.\n", encoding="utf-8")
    (notes / "Beta.md").write_text("---\ntitle: Beta\n---\nBeta body.\n", encoding="utf-8")

    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.markdown_root", str(notes))

    try:
        result = runner.invoke(app, ["ingest", "markdown"])

        assert result.exit_code == 0
        assert "Ingesting from markdown" in result.output
        assert "markdown: 2 new" in result.output

        alpha = store.get_unit_by_source("me", "Alpha.md", "markdown_note")
        beta = store.get_unit_by_source("me", "Beta.md", "markdown_note")
        assert alpha is not None
        assert beta is not None
        assert alpha.tags == ["cli"]
        assert beta.title == "Beta"
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].from_unit_id == alpha.id
        assert edges[0].to_unit_id == beta.id
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_ingest_markdown_incremental_links_to_existing_note(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    alpha_path = notes / "Alpha.md"
    beta_path = notes / "Beta.md"
    alpha_path.write_text("Alpha links to [[Beta]].\n", encoding="utf-8")
    beta_path.write_text("Beta body.\n", encoding="utf-8")
    os.utime(beta_path, (1_700_000_000, 1_700_000_000))
    os.utime(alpha_path, (1_700_100_000, 1_700_100_000))

    store = _make_store()
    beta = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="Beta.md",
            source_entity_type="markdown_note",
            title="Beta",
            content="Beta body.",
        )
    )
    store.upsert_sync_state(
        SyncState(
            source_project="markdown",
            source_entity_type="markdown_note",
            last_sync_at=datetime.fromtimestamp(1_700_050_000, tz=timezone.utc),
        )
    )
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.markdown_root", str(notes))

    try:
        result = runner.invoke(app, ["ingest", "markdown"])

        assert result.exit_code == 0
        alpha = store.get_unit_by_source("me", "Alpha.md", "markdown_note")
        assert alpha is not None
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].from_unit_id == alpha.id
        assert edges[0].to_unit_id == beta.id
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


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


def test_shortest_path_command_emits_json_with_edge_relations(monkeypatch):
    store = _make_store()
    a_id, _, c_id, _ = _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["shortest-path", a_id, c_id, "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert [unit["title"] for unit in payload["path"]] == ["Node A", "Node B", "Node C"]
        assert [edge["relation"] for edge in payload["edges"]] == ["builds_on", "inspires"]
        assert payload["edges"][0]["from_unit_id"] == a_id
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


def test_search_command_emits_semantic_json_with_scores(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.rag.embeddings.get_embedding_provider", lambda *args, **kwargs: MockEmbeddingProvider())

    try:
        result = runner.invoke(
            app,
            ["search", "solar", "--mode", "semantic", "--limit", "1", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["mode"] == "semantic"
        assert payload["results"][0]["title"].startswith("Solar")
        assert payload["results"][0]["source_project"] == "max"
        assert isinstance(payload["results"][0]["score"], float)
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_infer_edges_command_dry_run_and_normal_mode(monkeypatch):
    store = _make_store()
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="edge-a",
            source_entity_type="insight",
            title="Edge A",
            content="Edge A content",
            content_type=ContentType.INSIGHT,
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="edge-b",
            source_entity_type="insight",
            title="Edge B",
            content="Edge B content",
            content_type=ContentType.INSIGHT,
        )
    )
    store.update_embedding(a.id, serialize_embedding([1.0, 0.0]))
    store.update_embedding(b.id, serialize_embedding([0.9, 0.1]))
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        dry_result = runner.invoke(
            app,
            ["infer-edges", "--min-similarity", "0.8", "--dry-run", "--json"],
        )

        assert dry_result.exit_code == 0
        dry_payload = json.loads(dry_result.output)
        assert dry_payload["inserted"] == 0
        assert dry_payload["candidates"][0]["status"] == "would_insert"
        assert len(store.get_all_edges()) == 0

        result = runner.invoke(app, ["infer-edges", "--min-similarity", "0.8", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["inserted"] == 1
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].relation == EdgeRelation.RELATES_TO
        assert edges[0].source == EdgeSource.INFERRED
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_design_briefs_command_prints_readable_fields(monkeypatch):
    store = _make_store()
    _populate_design_briefs(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["design-briefs"])

        assert result.exit_code == 0
        assert "[draft] Ops workflow brief" in result.output
        assert "Domain: devtools | Theme: workflow" in result.output
        assert "Readiness: 82" in result.output
        assert "Lead idea: idea-lead-1 | Source ideas: idea-a, idea-b" in result.output
        assert "Validation plan: Interview 5 ops teams" in result.output
        assert "First milestones:" in result.output
        assert "Ignored external brief" not in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_design_briefs_command_emits_json_metadata(monkeypatch):
    store = _make_store()
    _populate_design_briefs(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["design-briefs", "--domain", "devtools", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert len(payload["results"]) == 1
        brief = payload["results"][0]
        assert brief["title"] == "Ops workflow brief"
        assert brief["content_type"] == "design_brief"
        assert brief["metadata"]["readiness_score"] == 82
        assert brief["metadata"]["source_idea_ids"] == ["idea-a", "idea-b"]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("args", "expected_title", "unexpected_title"),
    [
        (["--domain", "devtools"], "Ops workflow brief", "Platform expansion brief"),
        (["--status", "review"], "Platform expansion brief", "Ops workflow brief"),
    ],
)
def test_design_briefs_command_applies_filters(monkeypatch, args, expected_title, unexpected_title):
    store = _make_store()
    _populate_design_briefs(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["design-briefs", *args, "--limit", "1"])

        assert result.exit_code == 0
        assert expected_title in result.output
        assert unexpected_title not in result.output
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


def test_neighbors_command_emits_json_with_edge_relations(monkeypatch):
    store = _make_store()
    _, b_id, _, _ = _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["neighbors", b_id, "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["center"]["title"] == "Node B"
        assert {neighbor["title"] for neighbor in payload["neighbors"]} == {"Node A", "Node C"}
        assert {edge["relation"] for edge in payload["edges"]} == {"builds_on", "inspires"}
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_backlinks_command_emits_expanded_json_and_applies_filters(monkeypatch):
    store = _make_store()
    _, b_id, _, _ = _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["backlinks", b_id, "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["center"]["title"] == "Node B"
        assert {
            (link["direction"], link["relation"], link["unit"]["title"])
            for link in payload["links"]
        } == {
            ("incoming", "builds_on", "Node A"),
            ("outgoing", "inspires", "Node C"),
        }
        assert "edge" in payload["links"][0]

        filtered = runner.invoke(
            app,
            [
                "backlinks",
                b_id,
                "--direction",
                "incoming",
                "--relation",
                "builds_on",
                "--json",
            ],
        )
        filtered_payload = json.loads(filtered.output)
        assert filtered.exit_code == 0
        assert len(filtered_payload["links"]) == 1
        assert filtered_payload["links"][0]["unit"]["title"] == "Node A"
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_backlinks_command_reports_missing_unit(monkeypatch):
    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["backlinks", "missing"])
        assert result.exit_code == 1
        assert "Unit not found: missing" in result.output

        json_result = runner.invoke(app, ["backlinks", "missing", "--json"])
        assert json_result.exit_code == 0
        assert json.loads(json_result.output)["error"] == "unit_not_found"
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


def test_source_coverage_command_emits_json_matching_service(monkeypatch):
    from graph.graph.service import GraphService

    store = _make_store()
    _populate_graph(store)
    store.upsert_sync_state(
        SyncState(
            source_project="sota",
            source_entity_type="paper",
            last_sync_at="2026-04-24T01:00:00+00:00",
            last_source_id="paper-1",
            items_synced=5,
        )
    )
    expected = GraphService(store).analyze_source_coverage()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["source-coverage", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload == expected
        by_source = {
            (item["source_project"], item["source_entity_type"]): item
            for item in payload["sources"]
        }
        assert by_source[("presence", "knowledge_item")]["orphan_count"] == 1
        assert by_source[("sota", "paper")]["unit_count"] == 0
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_source_coverage_command_prints_readable_summary(monkeypatch):
    store = _make_store()
    _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["source-coverage"])

        assert result.exit_code == 0
        assert "Source coverage:" in result.output
        assert "forty_two/knowledge_node" in result.output
        assert "Units: 2 | Edges: 2 | Orphans: 0" in result.output
        assert "presence/knowledge_item" in result.output
        assert "Units: 1 | Edges: 0 | Orphans: 1" in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_analytics_commands_emit_parseable_json(monkeypatch):
    store = _make_store()
    _populate_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        stats_result = runner.invoke(app, ["stats", "--json"])
        bridges_result = runner.invoke(app, ["bridges", "--limit", "1", "--json"])
        cross_project_result = runner.invoke(app, ["cross-project", "--json"])

        assert stats_result.exit_code == 0
        assert json.loads(stats_result.output)["nodes"] == 4

        assert bridges_result.exit_code == 0
        bridge_payload = json.loads(bridges_result.output)
        assert bridge_payload["results"][0]["unit"]["title"] == "Node B"
        assert "score" in bridge_payload["results"][0]

        assert cross_project_result.exit_code == 0
        cross_project_payload = json.loads(cross_project_result.output)
        assert cross_project_payload["results"] == [
            {"edge_count": 1, "projects": ["forty_two", "max"]}
        ]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_tags_command_lists_top_tags_with_breakdowns(monkeypatch):
    store = _make_store()
    _populate_tags_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["tags", "--limit", "2", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert [item["tag"] for item in payload["tags"]] == ["energy", "solar"]
        assert payload["tags"][0]["source_projects"] == {"max": 2, "forty_two": 1}
        assert payload["tags"][0]["content_types"] == {"insight": 2, "finding": 1}
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_tags_command_detail_applies_filters_and_co_occurrences(monkeypatch):
    store = _make_store()
    _populate_tags_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "tags",
                "--tag",
                "energy",
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["count"] == 2
        assert {unit["title"] for unit in payload["units"]} == {
            "Solar storage",
            "Battery storage",
        }
        assert [item["tag"] for item in payload["co_occurring_tags"]] == [
            "storage",
            "battery",
            "solar",
        ]
        assert [item["count"] for item in payload["co_occurring_tags"]] == [2, 1, 1]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_links_command_emits_json_and_filters_by_domain(monkeypatch):
    store = _make_store()
    _populate_links_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            ["links", "--domain", "example.com", "--limit", "5", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["filters"] == {"domain": "example.com"}
        assert payload["domains"][0]["domain"] == "example.com"
        assert payload["domains"][0]["count"] == 2
        assert payload["domains"][0]["url_count"] == 1
        assert payload["domains"][0]["urls"] == [
            {"url": "https://example.com/docs", "count": 2}
        ]
        assert {
            unit["source_id"] for unit in payload["domains"][0]["representative_units"]
        } == {"markdown-link", "duplicate-link"}
        assert payload["links"][0]["url"] == "https://example.com/docs"
        assert payload["links"][0]["count"] == 2
        assert {item["field"] for item in payload["links"][0]["occurrences"]} == {
            "content"
        }
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_links_command_human_output_shows_domains_and_units(monkeypatch):
    store = _make_store()
    _populate_links_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["links", "--limit", "2"])

        assert result.exit_code == 0
        assert "Top external link domains:" in result.output
        assert "example.com: 2 occurrences across 1 URLs" in result.output
        assert "[max] Markdown citation" in result.output
        assert "https://example.com/docs (2)" in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_duplicates_command_emits_reasons_units_and_applies_filters(monkeypatch):
    store = _make_store()
    _populate_duplicates_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "duplicates",
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        by_reason = {item["reason"]: item for item in payload["results"]}
        assert set(by_reason) == {"same_title", "similar_content"}
        assert {unit["source_id"] for unit in by_reason["same_title"]["units"]} == {
            "same-title-a",
            "same-title-b",
        }
        assert {
            unit["source_id"] for unit in by_reason["similar_content"]["units"]
        } == {
            "same-content-a",
            "same-content-b",
        }
        assert all(
            unit["source_project"] == "max"
            for item in payload["results"]
            for unit in item["units"]
        )
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
                "--created-after",
                "2026-04-21T00:00:00+00:00",
                "--created-before",
                "2026-04-23T00:00:00+00:00",
                "--min-utility",
                "0.9",
                "--max-utility",
                "0.93",
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


def test_search_command_emits_active_date_and_utility_filters_in_json(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "search",
                "solar",
                "--mode",
                "fulltext",
                "--limit",
                "10",
                "--created-after",
                "2026-04-21",
                "--created-before",
                "2026-04-23T00:00:00+00:00",
                "--min-utility",
                "0.9",
                "--max-utility",
                "0.93",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["filters"] == {
            "created_after": "2026-04-21",
            "created_before": "2026-04-23T00:00:00+00:00",
            "max_utility": 0.93,
            "min_utility": 0.9,
        }
        assert [result["title"] for result in payload["results"]] == [
            "Solar approved insight"
        ]
        assert payload["results"][0]["created_at"] == "2026-04-22T00:00:00+00:00"
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_queries_cli_save_list_run_and_delete(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.rag.embeddings.get_embedding_provider", lambda *args, **kwargs: MockEmbeddingProvider())

    try:
        save_result = runner.invoke(
            app,
            [
                "queries",
                "save",
                "approved-solar",
                "solar",
                "--mode",
                "fulltext",
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
                "--created-after",
                "2026-04-21T00:00:00+00:00",
                "--created-before",
                "2026-04-23T00:00:00+00:00",
                "--min-utility",
                "0.9",
                "--max-utility",
                "0.93",
                "--json",
            ],
        )

        assert save_result.exit_code == 0
        saved = json.loads(save_result.output)
        assert saved["name"] == "approved-solar"
        assert saved["filters"] == {
            "content_type": "insight",
            "created_after": "2026-04-21T00:00:00+00:00",
            "created_before": "2026-04-23T00:00:00+00:00",
            "max_utility": 0.93,
            "min_utility": 0.9,
            "review_state": "approved",
            "source_project": "max",
            "tag": "energy",
        }

        list_result = runner.invoke(app, ["queries", "list", "--json"])

        assert list_result.exit_code == 0
        assert json.loads(list_result.output)["queries"][0]["name"] == "approved-solar"

        run_result = runner.invoke(app, ["queries", "run", "approved-solar", "--json"])

        assert run_result.exit_code == 0
        payload = json.loads(run_result.output)
        assert payload["query"] == "solar"
        assert payload["mode"] == "fulltext"
        assert payload["filters"] == saved["filters"]
        assert [result["title"] for result in payload["results"]] == [
            "Solar approved insight"
        ]

        delete_result = runner.invoke(app, ["queries", "delete", "approved-solar", "--json"])

        assert delete_result.exit_code == 0
        assert json.loads(delete_result.output) == {
            "deleted": True,
            "name": "approved-solar",
        }

        missing_delete = runner.invoke(app, ["queries", "delete", "approved-solar", "--json"])

        assert missing_delete.exit_code == 0
        assert json.loads(missing_delete.output) == {
            "deleted": False,
            "error": "Saved query not found: approved-solar",
            "name": "approved-solar",
        }
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]
