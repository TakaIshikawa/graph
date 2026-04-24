"""CLI tests for graph commands."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graph.cli.main import app
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
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
