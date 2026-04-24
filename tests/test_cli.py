"""CLI tests for graph commands."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
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


class RecordingEmbeddingProvider(MockEmbeddingProvider):
    def __init__(self) -> None:
        self.batch_texts: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_texts.append(texts)
        return super().embed_batch(texts)


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


def _populate_timeline_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="jan-solar",
            source_entity_type="insight",
            title="January solar",
            content="Solar storage",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar", "storage"],
            created_at=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="feb-grid",
            source_entity_type="knowledge_node",
            title="February grid",
            content="Grid finding",
            content_type=ContentType.FINDING,
            tags=["energy", "grid"],
            created_at=datetime.fromisoformat("2026-02-05T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="feb-battery",
            source_entity_type="insight",
            title="February battery",
            content="Battery storage",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage"],
            created_at=datetime.fromisoformat("2026-02-20T10:00:00+00:00"),
        ),
    ]:
        store.insert_unit(unit)


def _populate_tag_synonym_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="agent-hyphen",
            source_entity_type="insight",
            title="Agent hyphen",
            content="Agent note",
            content_type=ContentType.INSIGHT,
            tags=["ai-agent"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="agent-underscore",
            source_entity_type="insight",
            title="Agent underscore",
            content="Agent note",
            content_type=ContentType.INSIGHT,
            tags=["ai_agent"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="agent-plural",
            source_entity_type="knowledge_node",
            title="Agent plural",
            content="Agent note",
            content_type=ContentType.FINDING,
            tags=["AI Agents"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="unrelated",
            source_entity_type="knowledge_item",
            title="Unrelated",
            content="Unrelated note",
            content_type=ContentType.ARTIFACT,
            tags=["storage"],
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


def _populate_edge_suggestion_graph(store: Store) -> tuple[str, str]:
    solar = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="solar-storage",
            source_entity_type="insight",
            title="Solar storage roadmap",
            content="Battery grid planning references https://example.com/docs.",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar", "storage"],
        )
    )
    battery = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="battery-storage",
            source_entity_type="insight",
            title="Battery storage plan",
            content="Grid battery storage notes cite https://example.com/docs.",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage", "battery"],
        )
    )
    grid = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="solar-grid",
            source_entity_type="knowledge_node",
            title="Solar grid plan",
            content="Solar storage grid reference https://example.com/docs.",
            content_type=ContentType.FINDING,
            tags=["energy", "solar", "grid"],
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=grid.id,
            to_unit_id=solar.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    return battery.id, solar.id


def _populate_review_queue_graph(store: Store) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    old_isolated = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="old-isolated",
            source_entity_type="insight",
            title="Old isolated",
            content="Old unreviewed insight",
            content_type=ContentType.INSIGHT,
            utility_score=0.6,
            created_at=now - timedelta(days=420),
        )
    )
    recent_reviewed = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="recent-reviewed",
            source_entity_type="insight",
            title="Recent reviewed",
            content="Recent reviewed insight",
            content_type=ContentType.INSIGHT,
            metadata={"last_reviewed_at": now.isoformat()},
            utility_score=1.0,
            created_at=now - timedelta(days=2),
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="neighbor",
            source_entity_type="knowledge_node",
            title="Neighbor",
            content="Connected node",
            content_type=ContentType.FINDING,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=recent_reviewed.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="old-finding",
            source_entity_type="knowledge_node",
            title="Old finding",
            content="Old finding",
            content_type=ContentType.FINDING,
            created_at=now - timedelta(days=365),
        )
    )
    return old_isolated.id, recent_reviewed.id


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
    assert exported["schema_version"] == 2
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


def test_export_mermaid_command_writes_markdown_and_supports_neighborhood(tmp_path, monkeypatch):
    store = _make_store()
    a_id, b_id, c_id, d_id = _populate_graph(store)
    export_path = tmp_path / "graph.md"

    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    result = runner.invoke(
        app,
        [
            "export-mermaid",
            str(export_path),
            "--unit-id",
            a_id,
            "--depth",
            "1",
            "--limit",
            "10",
        ],
    )

    try:
        assert result.exit_code == 0
        assert "Exported 2 nodes and 1 edges" in result.output
        text = export_path.read_text()
        assert text.startswith("```mermaid\ngraph TD\n")
        assert 'n0["Node A"]' in text
        assert 'n1["Node B"]' in text
        assert "n0 -->|builds_on| n1" in text
        assert c_id not in text
        assert d_id not in text
        assert b_id not in text
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_neighborhood_command_writes_local_json_and_caps_depth(tmp_path, monkeypatch):
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


def test_export_anki_command_writes_sanitized_tsv(tmp_path, monkeypatch):
    store = _make_store()
    inserted = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="anki\t1",
            source_entity_type="insight",
            title="Solar\tPrompt\nLine",
            content="First\tfact\nSecond fact\r\nThird fact",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar"],
        )
    )
    export_path = tmp_path / "anki.tsv"
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["export-anki", str(export_path)])

        assert result.exit_code == 0
        assert "Exported 1 Anki rows" in result.output
        lines = export_path.read_text().splitlines()
        assert len(lines) == 1
        row = lines[0].split("\t")
        assert row == [
            "Solar Prompt Line",
            (
                "First fact Second fact Third fact "
                f"Source: max/insight/anki 1 (graph_id: {inserted.id})"
            ),
            "",
            "anki 1",
        ]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_anki_command_honors_filters_tags_limit_and_json(tmp_path, monkeypatch):
    store = _make_store()
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="keep-newer",
            source_entity_type="insight",
            title="Keep Newer",
            content="Keep newer content",
            content_type=ContentType.INSIGHT,
            tags=["energy", "AI Agents"],
            created_at=datetime.fromisoformat("2026-04-24T00:00:00+00:00"),
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="keep-older",
            source_entity_type="insight",
            title="Keep Older",
            content="Keep older content",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar"],
            created_at=datetime.fromisoformat("2026-04-23T00:00:00+00:00"),
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="wrong-tag",
            source_entity_type="insight",
            title="Wrong Tag",
            content="Wrong tag content",
            content_type=ContentType.INSIGHT,
            tags=["solar"],
            created_at=datetime.fromisoformat("2026-04-25T00:00:00+00:00"),
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="wrong-project",
            source_entity_type="knowledge_node",
            title="Wrong Project",
            content="Wrong project content",
            content_type=ContentType.INSIGHT,
            tags=["energy"],
            created_at=datetime.fromisoformat("2026-04-26T00:00:00+00:00"),
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="wrong-type",
            source_entity_type="design_brief",
            title="Wrong Type",
            content="Wrong type content",
            content_type=ContentType.DESIGN_BRIEF,
            tags=["energy"],
            created_at=datetime.fromisoformat("2026-04-27T00:00:00+00:00"),
        )
    )
    export_path = tmp_path / "filtered-anki.tsv"
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "export-anki",
                str(export_path),
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--tag",
                "energy",
                "--limit",
                "1",
                "--include-tags",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["rows_exported"] == 1
        assert payload["include_tags"] is True
        assert payload["filters"] == {
            "source_project": "max",
            "content_type": "insight",
            "tag": "energy",
            "limit": 1,
        }
        assert payload["path"] == str(export_path)

        lines = export_path.read_text().splitlines()
        assert len(lines) == 1
        row = lines[0].split("\t")
        assert row[0] == "Keep Newer"
        assert row[2] == "energy AI_Agents"
        assert row[3] == "keep-newer"
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


def test_ingest_feed_command_uses_configured_sources(tmp_path, monkeypatch):
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>CLI Feed</title>
            <item>
              <guid>cli-feed-1</guid>
              <title>Local feed item</title>
              <link>https://example.com/local</link>
              <description>Read from a local XML fixture.</description>
              <category>local</category>
              <pubDate>Thu, 24 Apr 2025 09:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.feed_sources", str(feed))

    try:
        result = runner.invoke(app, ["ingest", "feed"])

        assert result.exit_code == 0
        assert "Ingesting from feed" in result.output
        assert "feed: 1 new" in result.output
        unit = store.conn.execute(
            """SELECT * FROM knowledge_units
               WHERE source_project = 'me' AND source_entity_type = 'feed_item'"""
        ).fetchone()
        assert unit is not None
        assert unit["title"] == "Local feed item"
        assert json.loads(unit["tags"]) == ["local"]
        metadata = json.loads(unit["metadata"])
        assert metadata["link"] == "https://example.com/local"
        assert metadata["id"] == "cli-feed-1"
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_ingest_bookmarks_command_uses_configured_path_and_updates_by_url(tmp_path, monkeypatch):
    bookmarks = tmp_path / "bookmarks.html"
    bookmarks.write_text(
        """<!DOCTYPE NETSCAPE-Bookmark-file-1>
        <DL><p>
          <DT><H3>Bookmarks Bar</H3>
          <DL><p>
            <DT><H3>Research</H3>
            <DL><p>
              <DT><A HREF="https://example.com/bookmark"
                     ADD_DATE="1713952800"
                     LAST_MODIFIED="1713956400">Useful Bookmark</A>
            </DL><p>
          </DL><p>
        </DL><p>
        """,
        encoding="utf-8",
    )

    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.bookmarks_path", str(bookmarks))

    try:
        result = runner.invoke(app, ["ingest", "bookmarks"])

        assert result.exit_code == 0
        assert "Ingesting from bookmarks" in result.output
        assert "bookmarks: 1 new" in result.output
        unit = store.get_unit_by_source("bookmarks", "https://example.com/bookmark", "bookmark")
        assert unit is not None
        assert unit.title == "Useful Bookmark"
        assert unit.tags == ["Bookmarks Bar", "Bookmarks Bar/Research"]
        assert unit.metadata["url"] == "https://example.com/bookmark"
        assert unit.metadata["folder_path"] == "Bookmarks Bar/Research"

        bookmarks.write_text(
            bookmarks.read_text(encoding="utf-8").replace("Useful Bookmark", "Updated Bookmark"),
            encoding="utf-8",
        )
        second = runner.invoke(app, ["ingest", "bookmarks", "--full"])

        assert second.exit_code == 0
        assert "bookmarks: 0 new, 1 updated" in second.output
        rows = store.conn.execute(
            """SELECT title FROM knowledge_units
               WHERE source_project = 'bookmarks'
                 AND source_id = 'https://example.com/bookmark'
                 AND source_entity_type = 'bookmark'"""
        ).fetchall()
        assert [row["title"] for row in rows] == ["Updated Bookmark"]

        search = runner.invoke(
            app,
            [
                "search",
                "https://example.com/bookmark",
                "--mode",
                "fulltext",
                "--limit",
                "1",
            ],
        )

        assert search.exit_code == 0
        assert "Updated Bookmark" in search.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_ingest_csv_command_uses_configured_path_and_indexes_fulltext(tmp_path, monkeypatch):
    csv_path = tmp_path / "knowledge.csv"
    csv_path.write_text(
        "source_id,title,content,content_type,tags,utility_score,confidence,created_at,metadata_json\n"
        'csv-1,CSV Import,"Spreadsheet search phrase.",artifact,"spreadsheet, import",9.1,0.8,2025-04-24T12:00:00Z,"{""tool"": ""sheet""}"\n',
        encoding="utf-8",
    )

    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.csv_path", str(csv_path))

    try:
        result = runner.invoke(app, ["ingest", "csv"])

        assert result.exit_code == 0
        assert "Ingesting from csv" in result.output
        assert "csv: 1 new" in result.output
        unit = store.get_unit_by_source("csv", "csv-1", "csv_row")
        assert unit is not None
        assert unit.title == "CSV Import"
        assert unit.content_type == "artifact"
        assert unit.tags == ["spreadsheet", "import"]
        assert unit.utility_score == 9.1
        assert unit.confidence == 0.8
        assert unit.metadata == {"tool": "sheet"}

        search = runner.invoke(
            app,
            [
                "search",
                "Spreadsheet",
                "--mode",
                "fulltext",
                "--limit",
                "1",
            ],
        )

        assert search.exit_code == 0
        assert "CSV Import" in search.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_ingest_jsonl_command_uses_configured_path_and_indexes_fulltext(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "knowledge.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "source_id": "jsonl-1",
                "title": "JSONL Import",
                "content": "Transcript search phrase.",
                "content_type": "artifact",
                "tags": ["transcript", "import"],
                "utility_score": 9.2,
                "confidence": 0.82,
                "created_at": "2025-04-24T12:00:00Z",
                "updated_at": "2025-04-25T12:00:00Z",
                "metadata": {"tool": "transcript"},
            }
        )
        + "\n{not json\n",
        encoding="utf-8",
    )

    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.jsonl_path", str(jsonl_path))

    try:
        result = runner.invoke(app, ["ingest", "jsonl"])

        assert result.exit_code == 0
        assert "Ingesting from jsonl" in result.output
        assert "jsonl: 1 new" in result.output
        unit = store.get_unit_by_source("jsonl", "jsonl-1", "jsonl_record")
        assert unit is not None
        assert unit.title == "JSONL Import"
        assert unit.content_type == "artifact"
        assert unit.tags == ["transcript", "import"]
        assert unit.utility_score == 9.2
        assert unit.confidence == 0.82
        assert unit.metadata == {"tool": "transcript"}
        assert unit.updated_at.isoformat() == "2025-04-25T12:00:00+00:00"

        search = runner.invoke(
            app,
            [
                "search",
                "Transcript",
                "--mode",
                "fulltext",
                "--limit",
                "1",
            ],
        )

        assert search.exit_code == 0
        assert "JSONL Import" in search.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_ingest_opml_command_uses_configured_path_and_inserts_edges(tmp_path, monkeypatch):
    opml_path = tmp_path / "subscriptions.opml"
    opml_path.write_text(
        """<opml version="2.0">
          <body>
            <outline text="Research">
              <outline text="AI Feed" xmlUrl="https://example.com/ai.xml"
                htmlUrl="https://example.com/ai" />
            </outline>
          </body>
        </opml>
        """,
        encoding="utf-8",
    )

    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.opml_path", str(opml_path))

    try:
        result = runner.invoke(app, ["ingest", "opml"])

        assert result.exit_code == 0
        assert "Ingesting from opml" in result.output
        assert "opml: 2 new, 0 updated, 1 edges" in result.output
        unit = store.conn.execute(
            """SELECT * FROM knowledge_units
               WHERE source_project = 'opml' AND title = 'AI Feed'"""
        ).fetchone()
        assert unit is not None
        assert "https://example.com/ai.xml" in unit["content"]
        assert json.loads(unit["metadata"])["xmlUrl"] == "https://example.com/ai.xml"
        assert len(store.get_all_edges()) == 1

        search = runner.invoke(
            app,
            ["search", "https://example.com/ai.xml", "--mode", "fulltext", "--limit", "1"],
        )
        assert search.exit_code == 0
        assert "AI Feed" in search.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_ingest_text_command_uses_configured_root_and_indexes_fulltext(tmp_path, monkeypatch):
    root = tmp_path / "text"
    nested = root / "nested"
    nested.mkdir(parents=True)
    note = nested / "transcript.txt"
    note.write_text("Transcript Heading\nPlain text search phrase.\n", encoding="utf-8")

    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr("graph.cli.main.settings.text_root", str(root))

    try:
        result = runner.invoke(app, ["ingest", "text"])

        assert result.exit_code == 0
        assert "Ingesting from text" in result.output
        assert "text: 1 new" in result.output
        unit = store.get_unit_by_source("me", "nested/transcript.txt", "text_document")
        assert unit is not None
        assert unit.title == "Transcript Heading"
        assert unit.content == "Transcript Heading\nPlain text search phrase.\n"
        assert unit.metadata == {
            "path": "nested/transcript.txt",
            "file_size": note.stat().st_size,
        }
        sync_state = store.get_sync_state("text", "text_document")
        assert sync_state is not None
        assert sync_state.items_synced == 1

        search = runner.invoke(
            app,
            ["search", "Plain text", "--mode", "fulltext", "--limit", "1"],
        )

        assert search.exit_code == 0
        assert "Transcript Heading" in search.output
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
    monkeypatch.setattr(
        "graph.rag.embeddings.get_embedding_provider",
        lambda *args, **kwargs: MockEmbeddingProvider(),
    )

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
    monkeypatch.setattr(
        "graph.rag.embeddings.get_embedding_provider",
        lambda *args, **kwargs: MockEmbeddingProvider(),
    )

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
        assert payload["results"][0]["snippet"]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_similar_command_emits_json_with_seed_scores_and_reason(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    seed = store.get_unit_by_source("max", "approved", "insight")
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            ["similar", seed.id, "--limit", "2", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["seed_id"] == seed.id
        assert payload["source_mode"] == "embedding"
        assert all(item["id"] != seed.id for item in payload["results"])
        assert all(isinstance(item["score"], float) for item in payload["results"])
        assert all(item["reason"] == "embedding_similarity" for item in payload["results"])
        assert all(item["source_mode"] == "embedding" for item in payload["results"])
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_similar_command_fallback_json_respects_filters(monkeypatch):
    store = _make_store()
    seed = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="seed",
            source_entity_type="insight",
            title="Solar storage seed",
            content="Solar storage market adoption",
            content_type=ContentType.INSIGHT,
            tags=["solar"],
        )
    )
    keep = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="keep",
            source_entity_type="insight",
            title="Solar storage match",
            content="Solar storage market demand",
            content_type=ContentType.INSIGHT,
            tags=["energy"],
        )
    )
    skip = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="skip",
            source_entity_type="knowledge_node",
            title="Solar storage finding",
            content="Solar storage market demand",
            content_type=ContentType.FINDING,
            tags=["energy"],
        )
    )
    for unit in [seed, keep, skip]:
        store.fts_index_unit(unit)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "similar",
                seed.id,
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--tag",
                "energy",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["source_mode"] == "local_search"
        assert [item["id"] for item in payload["results"]] == [keep.id]
        assert payload["results"][0]["reason"] == "seed_text_fulltext"
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_context_command_emits_ranked_units_neighbors_and_budget(monkeypatch):
    store = _make_store()
    center = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="context-center",
            source_entity_type="insight",
            title="Solar context center",
            content="Solar context content " * 10,
            content_type=ContentType.INSIGHT,
            tags=["solar"],
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="context-neighbor",
            source_entity_type="knowledge_node",
            title="Panel neighbor",
            content="Neighbor content " * 10,
            content_type=ContentType.FINDING,
        )
    )
    store.fts_index_unit(center)
    store.fts_index_unit(neighbor)
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=neighbor.id,
            to_unit_id=center.id,
            relation=EdgeRelation.BUILDS_ON,
            source=EdgeSource.MANUAL,
        )
    )
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "context",
                "solar",
                "--mode",
                "fulltext",
                "--limit",
                "1",
                "--neighbor-depth",
                "9",
                "--char-budget",
                "40",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["query"] == "solar"
        assert payload["ranked_units"][0]["id"] == center.id
        assert payload["neighbors"][0]["id"] == neighbor.id
        assert payload["selected_edges"][0]["relation"] == "builds_on"
        assert payload["metadata"]["neighbor_depth"] == 2
        assert (
            sum(
                len(unit["content_excerpt"])
                for unit in [*payload["ranked_units"], *payload["neighbors"]]
            )
            <= 40
        )
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_embeddings_status_command_emits_filtered_json(monkeypatch):
    store = _make_store()
    fresh = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="fresh",
            source_entity_type="insight",
            title="Fresh",
            content="Fresh content",
            content_type=ContentType.INSIGHT,
        )
    )
    stale = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="stale",
            source_entity_type="insight",
            title="Stale",
            content="Stale content",
            content_type=ContentType.INSIGHT,
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="missing",
            source_entity_type="knowledge_node",
            title="Missing",
            content="Missing content",
            content_type=ContentType.FINDING,
        )
    )
    store.update_embedding(fresh.id, serialize_embedding([1.0, 0.0]))
    store.update_embedding(stale.id, serialize_embedding([1.0, 0.0]))
    store.update_unit_fields(stale.id, content="Updated stale content")

    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "embeddings-status",
                "--project",
                "max",
                "--content-type",
                "insight",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload == {
            "content_type": "insight",
            "fresh": 1,
            "missing": 0,
            "source_project": "max",
            "stale": 1,
            "total": 2,
        }
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_embed_command_honors_limit_and_stale_only(monkeypatch):
    store = _make_store()
    fresh = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="fresh",
            source_entity_type="insight",
            title="Fresh",
            content="Fresh content",
        )
    )
    stale = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="stale",
            source_entity_type="insight",
            title="Stale",
            content="Stale content",
        )
    )
    missing = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="missing",
            source_entity_type="insight",
            title="Missing",
            content="Missing content",
        )
    )
    store.update_embedding(fresh.id, serialize_embedding([1.0, 0.0]))
    store.update_embedding(stale.id, serialize_embedding([1.0, 0.0]))
    store.update_unit_fields(stale.id, content="Updated stale content")

    provider = RecordingEmbeddingProvider()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr(
        "graph.rag.embeddings.get_embedding_provider", lambda *args, **kwargs: provider
    )

    try:
        result = runner.invoke(
            app,
            ["embed", "--stale-only", "--limit", "1", "--batch-size", "1", "--delay", "0"],
        )

        assert result.exit_code == 0
        assert "Embedding 1 units" in result.output
        assert sum(len(batch) for batch in provider.batch_texts) == 1
        embedded_text = provider.batch_texts[0][0]
        assert ("Stale" in embedded_text) or ("Missing" in embedded_text)
        assert "Fresh" not in embedded_text
        status = store.get_embedding_status(source_project="max")
        assert status["fresh"] == 2
        assert status["stale"] + status["missing"] == 1
        assert missing.id or stale.id
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_embed_command_force_refreshes_existing_embeddings(monkeypatch):
    store = _make_store()
    for index in range(2):
        unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=f"force-{index}",
                source_entity_type="insight",
                title=f"Force {index}",
                content=f"Force content {index}",
            )
        )
        store.update_embedding(unit.id, serialize_embedding([1.0, 0.0]))

    provider = RecordingEmbeddingProvider()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr(
        "graph.rag.embeddings.get_embedding_provider", lambda *args, **kwargs: provider
    )

    try:
        result = runner.invoke(
            app,
            ["embed", "--force", "--limit", "1", "--batch-size", "1", "--delay", "0"],
        )

        assert result.exit_code == 0
        assert "Embedding 1 units" in result.output
        assert sum(len(batch) for batch in provider.batch_texts) == 1
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


def test_freshness_command_emits_json_and_days_controls_window(monkeypatch):
    store = _make_store()
    now = datetime.now(timezone.utc)
    recent = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="fresh-recent",
            source_entity_type="insight",
            title="Recent freshness unit",
            content="Freshness content",
        )
    )
    old = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="fresh-old",
            source_entity_type="insight",
            title="Old freshness unit",
            content="Old freshness content",
        )
    )
    store.conn.execute(
        "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
        ((now - timedelta(days=3)).isoformat(), recent.id),
    )
    store.conn.execute(
        "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
        ((now - timedelta(days=8)).isoformat(), old.id),
    )
    store.conn.commit()
    store.upsert_sync_state(
        SyncState(
            source_project="max",
            source_entity_type="insight",
            last_sync_at=now - timedelta(days=8),
            items_synced=2,
        )
    )
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr(
        "graph.cli.main._supported_sync_targets",
        lambda: [("max", "insight"), ("presence", "knowledge_item")],
    )

    try:
        seven = runner.invoke(app, ["freshness", "--days", "7", "--json"])
        ten = runner.invoke(app, ["freshness", "--days", "10", "--json"])

        assert seven.exit_code == 0
        seven_payload = json.loads(seven.output)
        by_target = {
            (item["source_project"], item["source_entity_type"]): item
            for item in seven_payload["results"]
        }
        assert seven_payload["days"] == 7
        assert by_target[("max", "insight")]["recent_unit_count"] == 1
        assert by_target[("max", "insight")]["total_unit_count"] == 2
        assert by_target[("max", "insight")]["stale"] is True
        assert by_target[("presence", "knowledge_item")]["last_sync_at"] is None
        assert by_target[("presence", "knowledge_item")]["stale"] is True

        assert ten.exit_code == 0
        ten_payload = json.loads(ten.output)
        ten_max = ten_payload["results"][0]
        assert ten_payload["days"] == 10
        assert ten_max["recent_unit_count"] == 2
        assert ten_max["stale"] is False
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


def test_timeline_command_returns_sorted_json_buckets_and_filters(monkeypatch):
    store = _make_store()
    _populate_timeline_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "timeline",
                "--bucket",
                "month",
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--tag",
                "storage",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["total"] == 2
        assert [item["bucket"] for item in payload["buckets"]] == [
            "2026-01",
            "2026-02",
        ]
        assert payload["buckets"][0]["source_projects"] == {"max": 1}
        assert payload["buckets"][1]["top_tags"][0] == {"tag": "energy", "count": 1}
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_timeline_command_empty_graph_returns_empty_buckets(monkeypatch):
    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["timeline", "--bucket", "month", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["total"] == 0
        assert payload["buckets"] == []
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


def test_tag_graph_command_emits_json_and_readable_top_pairs(monkeypatch):
    store = _make_store()
    _populate_tags_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["tag-graph", "--min-count", "2", "--limit", "2", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert [node["tag"] for node in payload["nodes"]] == [
            "energy",
            "solar",
            "storage",
        ]
        assert [
            (edge["source"], edge["target"], edge["co_occurrence_count"])
            for edge in payload["edges"]
        ] == [
            ("energy", "solar", 2),
            ("energy", "storage", 2),
        ]
        assert all(edge["representative_unit_ids"] for edge in payload["edges"])

        readable = runner.invoke(app, ["tag-graph", "--min-count", "2", "--limit", "1"])
        assert readable.exit_code == 0
        assert "Top tag pairs:" in readable.output
        assert "energy <-> solar: 2 units" in readable.output
        assert "Representative units:" in readable.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_tag_synonyms_command_emits_json_and_readable_output(monkeypatch):
    store = _make_store()
    _populate_tag_synonym_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        json_result = runner.invoke(
            app,
            ["tag-synonyms", "--limit", "5", "--min-similarity", "0.8", "--json"],
        )

        assert json_result.exit_code == 0
        payload = json.loads(json_result.output)
        assert len(payload["suggestions"]) == 1
        suggestion = payload["suggestions"][0]
        assert suggestion["canonical_candidate"] == "ai-agent"
        assert {variant["tag"] for variant in suggestion["variants"]} == {
            "ai-agent",
            "ai_agent",
            "AI Agents",
        }
        assert "storage" not in {variant["tag"] for variant in suggestion["variants"]}

        readable_result = runner.invoke(app, ["tag-synonyms"])

        assert readable_result.exit_code == 0
        assert "Tag synonym suggestions:" in readable_result.output
        assert "ai-agent (3 uses, 3 variants, similarity 1.000)" in readable_result.output
        assert "AI Agents: 1" in readable_result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_rename_tag_command_dry_run_json_and_execute(monkeypatch):
    store = _make_store()
    _populate_tag_synonym_graph(store)
    for unit in store.get_all_units(limit=100):
        store.fts_index_unit(unit)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        dry_run = runner.invoke(
            app,
            [
                "rename-tag",
                "ai_agent",
                "ai-agent",
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--dry-run",
                "--json",
            ],
        )

        assert dry_run.exit_code == 0
        dry_payload = json.loads(dry_run.output)
        assert dry_payload["dry_run"] is True
        assert dry_payload["changed_count"] == 1
        assert dry_payload["sample_units"][0]["title"] == "Agent underscore"
        unit = store.get_unit_by_source("max", "agent-underscore", "insight")
        assert unit is not None
        assert unit.tags == ["ai_agent"]

        executed = runner.invoke(app, ["rename-tag", "ai_agent", "ai-agent"])

        assert executed.exit_code == 0
        assert "Updated 1 units: ai_agent -> ai-agent" in executed.output
        unit = store.get_unit_by_source("max", "agent-underscore", "insight")
        assert unit is not None
        assert unit.tags == ["ai-agent"]
        assert store.fts_search("ai_agent") == []
        assert {row["unit_id"] for row in store.fts_search("ai-agent")}
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_tags_apply_search_dry_run_execute_filters_and_reindexes(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        dry_run = runner.invoke(
            app,
            [
                "tags",
                "apply-search",
                "solar",
                "--add",
                "curated",
                "--add",
                "curated",
                "--remove",
                "energy",
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
                "--min-utility",
                "0.9",
                "--dry-run",
                "--json",
            ],
        )

        assert dry_run.exit_code == 0
        dry_payload = json.loads(dry_run.output)
        assert dry_payload["dry_run"] is True
        assert dry_payload["matched_count"] == 1
        assert dry_payload["changed_count"] == 1
        assert dry_payload["add_tags"] == ["curated"]
        assert dry_payload["remove_tags"] == ["energy"]
        assert dry_payload["changed_units"][0]["old_tags"] == ["energy", "solar"]
        assert dry_payload["changed_units"][0]["new_tags"] == ["solar", "curated"]
        approved = store.get_unit_by_source("max", "approved", "insight")
        assert approved is not None
        assert approved.tags == ["energy", "solar"]

        executed = runner.invoke(
            app,
            [
                "tags",
                "apply-search",
                "solar",
                "--add",
                "curated",
                "--remove",
                "energy",
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--tag",
                "energy",
                "--review-state",
                "approved",
                "--min-utility",
                "0.9",
                "--json",
            ],
        )

        assert executed.exit_code == 0
        payload = json.loads(executed.output)
        assert payload["dry_run"] is False
        assert payload["affected_count"] == 1
        approved = store.get_unit_by_source("max", "approved", "insight")
        rejected = store.get_unit_by_source("max", "rejected", "insight")
        assert approved is not None
        assert rejected is not None
        assert approved.tags == ["solar", "curated"]
        assert rejected.tags == ["energy", "solar"]
        assert store.fts_search("curated")[0]["unit_id"] == approved.id
        search_result = runner.invoke(
            app,
            ["search", "solar", "--tag", "curated", "--json"],
        )
        assert search_result.exit_code == 0
        search_payload = json.loads(search_result.output)
        assert [unit["id"] for unit in search_payload["results"]] == [approved.id]
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
        assert payload["domains"][0]["urls"] == [{"url": "https://example.com/docs", "count": 2}]
        assert {unit["source_id"] for unit in payload["domains"][0]["representative_units"]} == {
            "markdown-link",
            "duplicate-link",
        }
        assert payload["links"][0]["url"] == "https://example.com/docs"
        assert payload["links"][0]["count"] == 2
        assert {item["field"] for item in payload["links"][0]["occurrences"]} == {"content"}
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


def test_suggest_edges_command_emits_json_and_readable_output(monkeypatch):
    store = _make_store()
    from_id, to_id = _populate_edge_suggestion_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        json_result = runner.invoke(
            app,
            [
                "suggest-edges",
                "--source-project",
                "max",
                "--min-score",
                "0.4",
                "--limit",
                "5",
                "--json",
            ],
        )

        assert json_result.exit_code == 0
        payload = json.loads(json_result.output)
        assert payload["filters"] == {"source_project": "max"}
        assert len(payload["candidates"]) == 1
        candidate = payload["candidates"][0]
        assert candidate["from_id"] == from_id
        assert candidate["to_id"] == to_id
        assert candidate["score"] >= 0.8
        assert any(reason.startswith("shared tags:") for reason in candidate["reasons"])
        assert any(reason.startswith("shared links:") for reason in candidate["reasons"])

        readable_result = runner.invoke(
            app,
            ["suggest-edges", "--source-project", "max"],
        )

        assert readable_result.exit_code == 0
        assert "Edge suggestions:" in readable_result.output
        assert "Battery storage plan -> Solar storage roadmap" in readable_result.output
        assert "shared links: https://example.com/docs" in readable_result.output
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
        assert {unit["source_id"] for unit in by_reason["similar_content"]["units"]} == {
            "same-content-a",
            "same-content-b",
        }
        assert all(
            unit["source_project"] == "max" for item in payload["results"] for unit in item["units"]
        )
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_review_queue_command_emits_ranked_json_and_applies_filters(monkeypatch):
    store = _make_store()
    old_id, recent_id = _populate_review_queue_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "review-queue",
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["filters"] == {
            "source_project": "max",
            "content_type": "insight",
        }
        assert [item["unit"]["id"] for item in payload["queue"]] == [
            old_id,
            recent_id,
        ]
        assert payload["queue"][0]["score"] > payload["queue"][1]["score"]
        assert payload["queue"][0]["degree"] == 0
        assert {"age", "isolated", "utility_score", "unreviewed"} <= {
            reason["code"] for reason in payload["queue"][0]["reasons"]
        }
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode", ["fulltext", "semantic", "hybrid"])
def test_search_command_applies_filters_in_all_modes(monkeypatch, mode):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr(
        "graph.rag.embeddings.get_embedding_provider",
        lambda *args, **kwargs: MockEmbeddingProvider(),
    )

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
        assert [result["title"] for result in payload["results"]] == ["Solar approved insight"]
        assert payload["results"][0]["created_at"] == "2026-04-22T00:00:00+00:00"
        assert payload["results"][0]["snippet"]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_search_command_sorts_fulltext_by_created_at_desc(monkeypatch):
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
                "3",
                "--sort",
                "created_at_desc",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["sort"] == "created_at_desc"
        assert payload["metadata"]["sort"] == "created_at_desc"
        assert [item["title"] for item in payload["results"]] == [
            "Solar approved brief",
            "Solar rejected insight",
            "Solar approved insight",
        ]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_search_command_utility_sort_places_missing_values_last(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    missing = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="missing-utility",
            source_entity_type="insight",
            title="Solar missing utility",
            content="Solar energy storage market missing utility",
            content_type=ContentType.INSIGHT,
            tags=["solar"],
            created_at=datetime.fromisoformat("2026-04-25T00:00:00+00:00"),
        )
    )
    store.fts_index_unit(missing)
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
                "6",
                "--sort",
                "utility_desc",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert [item["title"] for item in payload["results"]][-1] == "Solar missing utility"
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_search_command_invalid_sort_emits_clear_json_error(monkeypatch):
    store = _make_store()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            ["search", "solar", "--sort", "newest", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "Unknown sort: newest" in payload["error"]
        assert "created_at_desc" in payload["valid_sorts"]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_search_facets_command_emits_fulltext_json(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            ["search-facets", "solar", "--mode", "fulltext", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["query"] == "solar"
        assert payload["mode"] == "fulltext"
        assert payload["total_matches"] == 5
        assert payload["facets"]["source_project"] == {"max": 4, "forty_two": 1}
        assert payload["facets"]["source_entity_type"] == {
            "insight": 3,
            "design_brief": 1,
            "knowledge_node": 1,
        }
        assert payload["facets"]["content_type"] == {
            "insight": 3,
            "design_brief": 1,
            "finding": 1,
        }
        assert payload["facets"]["tag"]["solar"] == 5
        assert payload["facets"]["tag"]["energy"] == 4
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode", ["fulltext", "semantic", "hybrid"])
def test_search_facets_command_respects_filters_in_all_modes(monkeypatch, mode):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr(
        "graph.rag.embeddings.get_embedding_provider",
        lambda *args, **kwargs: MockEmbeddingProvider(),
    )

    try:
        result = runner.invoke(
            app,
            [
                "search-facets",
                "solar",
                "--mode",
                mode,
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

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["mode"] == mode
        assert payload["filters"] == {
            "content_type": "insight",
            "created_after": "2026-04-21T00:00:00+00:00",
            "created_before": "2026-04-23T00:00:00+00:00",
            "max_utility": 0.93,
            "min_utility": 0.9,
            "review_state": "approved",
            "source_project": "max",
            "tag": "energy",
        }
        assert payload["total_matches"] == 1
        assert payload["facets"]["source_project"] == {"max": 1}
        assert payload["facets"]["source_entity_type"] == {"insight": 1}
        assert payload["facets"]["content_type"] == {"insight": 1}
        assert payload["facets"]["tag"] == {"energy": 1, "solar": 1}
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_queries_cli_save_list_run_and_delete(monkeypatch):
    store = _make_store()
    _populate_search_graph(store)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)
    monkeypatch.setattr(
        "graph.rag.embeddings.get_embedding_provider",
        lambda *args, **kwargs: MockEmbeddingProvider(),
    )

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
                "--sort",
                "utility_desc",
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
            "sort": "utility_desc",
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
        assert payload["sort"] == "utility_desc"
        assert payload["filters"] == saved["filters"]
        assert [result["title"] for result in payload["results"]] == ["Solar approved insight"]

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


def test_update_and_delete_unit_cli_json_reindexes_and_cleans_edges(monkeypatch):
    store = _make_store()
    manual = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-1",
            source_entity_type="manual",
            title="Original solar note",
            content="Original solar content",
            metadata={"owner": "me", "review_state": "draft"},
            tags=["solar"],
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-2",
            source_entity_type="manual",
            title="Neighbor",
            content="Neighbor content",
        )
    )
    store.fts_index_unit(manual)
    store.fts_index_unit(neighbor)
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=manual.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        update_result = runner.invoke(
            app,
            [
                "update-unit",
                manual.id,
                "--title",
                "Updated battery note",
                "--content",
                "Updated battery content",
                "--content-type",
                "finding",
                "--tag",
                "battery",
                "--metadata-json",
                '{"review_state":"approved","priority":"high"}',
                "--json",
            ],
        )

        assert update_result.exit_code == 0
        payload = json.loads(update_result.output)
        assert payload["updated"] is True
        assert payload["unit"]["title"] == "Updated battery note"
        assert payload["unit"]["content_type"] == "finding"
        assert payload["unit"]["tags"] == ["solar", "battery"]
        assert payload["unit"]["metadata"] == {
            "owner": "me",
            "priority": "high",
            "review_state": "approved",
        }

        assert store.fts_search("battery")[0]["unit_id"] == manual.id
        assert store.fts_search("Original") == []

        delete_result = runner.invoke(app, ["delete-unit", manual.id, "--yes", "--json"])

        assert delete_result.exit_code == 0
        assert json.loads(delete_result.output) == {
            "unit_id": manual.id,
            "deleted": True,
            "edges_deleted": 1,
        }
        assert store.get_unit(manual.id) is None
        assert store.get_all_edges() == []
        assert store.fts_search("battery") == []
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_pin_and_unpin_unit_cli_json_preserves_metadata(monkeypatch):
    store = _make_store()
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-pin",
            source_entity_type="manual",
            title="Pinned solar reference",
            content="Evergreen solar content",
            metadata={"owner": "me", "review_state": "approved"},
            tags=["solar"],
        )
    )
    store.fts_index_unit(unit)
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        pin_result = runner.invoke(
            app,
            ["pin-unit", unit.id, "--reason", "evergreen", "--json"],
        )

        assert pin_result.exit_code == 0
        pin_payload = json.loads(pin_result.output)
        assert pin_payload["updated"] is True
        assert pin_payload["unit"]["tags"] == ["solar"]
        assert pin_payload["unit"]["content"] == "Evergreen solar content"
        assert pin_payload["unit"]["metadata"]["owner"] == "me"
        assert pin_payload["unit"]["metadata"]["pinned"] is True
        assert pin_payload["unit"]["metadata"]["pin_reason"] == "evergreen"
        assert pin_payload["unit"]["metadata"]["pinned_at"]

        unpin_result = runner.invoke(app, ["unpin-unit", unit.id, "--json"])

        assert unpin_result.exit_code == 0
        unpin_payload = json.loads(unpin_result.output)
        assert unpin_payload["updated"] is True
        assert unpin_payload["unit"]["metadata"] == {
            "owner": "me",
            "review_state": "approved",
        }

        missing_result = runner.invoke(app, ["pin-unit", "missing", "--json"])

        assert missing_result.exit_code == 0
        assert json.loads(missing_result.output) == {
            "unit_id": "missing",
            "updated": False,
            "error": "unit_not_found",
            "message": "Unit not found: missing",
        }
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_pinned_cli_json_filters_orders_and_controls_content(monkeypatch):
    store = _make_store()
    older = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="older-pin",
            source_entity_type="insight",
            title="Older pinned",
            content="Older pinned content",
            content_type=ContentType.INSIGHT,
            tags=["solar", "workspace"],
            metadata={
                "pinned": True,
                "pinned_at": "2026-01-01T00:00:00+00:00",
                "pin_reason": "older",
            },
        )
    )
    newer = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="newer-pin",
            source_entity_type="insight",
            title="Newer pinned",
            content="Newer pinned content",
            content_type=ContentType.INSIGHT,
            tags=["solar", "workspace"],
            metadata={
                "pinned": True,
                "pinned_at": "2026-01-02T00:00:00+00:00",
                "pin_reason": "newer",
            },
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="wrong-tag",
            source_entity_type="insight",
            title="Wrong tag pinned",
            content="Wrong tag pinned content",
            content_type=ContentType.INSIGHT,
            tags=["archive"],
            metadata={
                "pinned": True,
                "pinned_at": "2026-01-03T00:00:00+00:00",
            },
        )
    )
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(
            app,
            [
                "pinned",
                "--source-project",
                "max",
                "--content-type",
                "insight",
                "--tag",
                "workspace",
                "--limit",
                "2",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert [unit["id"] for unit in payload["units"]] == [newer.id, older.id]
        assert payload["units"][0]["pin_reason"] == "newer"
        assert payload["units"][0]["pinned_at"] == "2026-01-02T00:00:00+00:00"
        assert "content" not in payload["units"][0]
        assert payload["filters"] == {
            "source_project": "max",
            "content_type": "insight",
            "tag": "workspace",
            "limit": 2,
        }

        with_content = runner.invoke(
            app,
            ["pinned", "--tag", "workspace", "--include-content", "--json"],
        )

        assert with_content.exit_code == 0
        content_payload = json.loads(with_content.output)
        assert content_payload["units"][0]["content"] == "Newer pinned content"
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_integrity_cli_json_reports_and_repairs_fts(monkeypatch):
    store = _make_store()
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="audit",
            source_entity_type="insight",
            title="Audit target",
            content="Search drift",
        )
    )
    store.conn.execute(
        "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
        ("deleted-unit", "Deleted", "stale content", "stale"),
    )
    store.conn.commit()
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        result = runner.invoke(app, ["integrity", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["categories"]["units_missing_fts_rows"]["count"] == 1
        assert payload["categories"]["stale_fts_rows"]["count"] == 1

        repaired = runner.invoke(app, ["integrity", "--json", "--repair-fts"])

        assert repaired.exit_code == 0
        repaired_payload = json.loads(repaired.output)
        assert repaired_payload["repair"]["requested"] is True
        assert repaired_payload["repair"]["fts_rows_inserted"] == 1
        assert repaired_payload["repair"]["fts_rows_deleted"] == 1
        assert repaired_payload["categories"]["units_missing_fts_rows"]["count"] == 0
        assert repaired_payload["categories"]["stale_fts_rows"]["count"] == 0
        assert store.get_unit(unit.id) is not None
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_edge_management_cli_json_lists_updates_and_deletes(monkeypatch):
    store = _make_store()
    center = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-1",
            source_entity_type="manual",
            title="Center",
            content="Center content",
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-2",
            source_entity_type="manual",
            title="Neighbor",
            content="Neighbor content",
        )
    )
    edge = store.insert_edge(
        KnowledgeEdge(
            from_unit_id=center.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.INFERRED,
            metadata={"old": "value"},
        )
    )
    proxy = StoreProxy(store)
    monkeypatch.setattr("graph.cli.main._get_store", lambda: proxy)

    try:
        list_result = runner.invoke(app, ["edges", "list", center.id, "--json"])
        assert list_result.exit_code == 0
        list_payload = json.loads(list_result.output)
        assert list_payload["center"]["title"] == "Center"
        assert list_payload["edges"][0]["id"] == edge.id
        assert list_payload["edges"][0]["direction"] == "outgoing"
        assert list_payload["edges"][0]["relation"] == "relates_to"
        assert list_payload["edges"][0]["source"] == "inferred"
        assert list_payload["edges"][0]["metadata"] == {"old": "value"}
        assert list_payload["edges"][0]["neighbor"]["title"] == "Neighbor"

        update_result = runner.invoke(
            app,
            [
                "update-edge",
                edge.id,
                "--relation",
                "inspires",
                "--weight",
                "0.5",
                "--source",
                "manual",
                "--metadata-json",
                '{"new":"value"}',
                "--json",
            ],
        )
        assert update_result.exit_code == 0
        update_payload = json.loads(update_result.output)
        assert update_payload["updated"] is True
        assert update_payload["edge"]["relation"] == "inspires"
        assert update_payload["edge"]["weight"] == 0.5
        assert update_payload["edge"]["source"] == "manual"
        assert update_payload["edge"]["metadata"] == {"old": "value", "new": "value"}

        backlinks = store.get_backlinks(neighbor.id)
        assert backlinks["links"][0]["relation"] == "inspires"

        delete_result = runner.invoke(app, ["delete-edge", edge.id, "--yes", "--json"])
        assert delete_result.exit_code == 0
        assert json.loads(delete_result.output) == {
            "edge_id": edge.id,
            "deleted": True,
        }
        assert store.get_unit(center.id) is not None
        assert store.get_unit(neighbor.id) is not None
        assert store.get_all_edges() == []

        missing_result = runner.invoke(app, ["delete-edge", "missing", "--yes", "--json"])
        assert missing_result.exit_code == 0
        assert json.loads(missing_result.output)["error"] == "edge_not_found"

        invalid_result = runner.invoke(
            app,
            ["update-edge", edge.id, "--relation", "invalid", "--json"],
        )
        assert invalid_result.exit_code == 0
        assert "invalid" in json.loads(invalid_result.output)["error"]
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]
