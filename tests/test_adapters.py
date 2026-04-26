"""Tests for source adapters."""

from __future__ import annotations

import os
import json
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest
import yaml

from graph.adapters.bookmarks import BookmarksAdapter
from graph.adapters.csv_adapter import CsvAdapter
from graph.adapters.feed import FeedAdapter
from graph.adapters.forty_two import FortyTwoAdapter
from graph.adapters.html import HtmlAdapter
from graph.adapters.ical import ICalAdapter
from graph.adapters.jsonl_adapter import JsonlAdapter
from graph.adapters.markdown import MarkdownAdapter
from graph.adapters.max_adapter import MaxAdapter
from graph.adapters.me import MeAdapter
from graph.adapters.opml import OpmlAdapter
from graph.adapters.presence import PresenceAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.text import TextAdapter
from graph.store.db import Store
from graph.types.models import SyncState


@pytest.fixture
def forty_two_db():
    """Create a minimal forty-two database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE experiments (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            title TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            code TEXT NOT NULL,
            language TEXT DEFAULT 'python',
            utility_dimensions TEXT DEFAULT '[]',
            parent_experiment_id TEXT,
            benchmark_id TEXT,
            status TEXT DEFAULT 'completed',
            created_at TEXT NOT NULL
        );
        CREATE TABLE knowledge_nodes (
            id TEXT PRIMARY KEY,
            experiment_id TEXT REFERENCES experiments(id),
            summary TEXT NOT NULL,
            utility_contribution REAL DEFAULT 0.0,
            tags TEXT DEFAULT '[]',
            findings TEXT,
            is_negative INTEGER DEFAULT 0,
            novelty_score REAL DEFAULT 0.0,
            is_pinned INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE knowledge_edges (
            id TEXT PRIMARY KEY,
            from_node_id TEXT REFERENCES knowledge_nodes(id),
            to_node_id TEXT REFERENCES knowledge_nodes(id),
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            created_by_agent_id TEXT,
            auto_generated INTEGER DEFAULT 0
        );
    """)
    # Insert test data
    conn.execute(
        """INSERT INTO experiments (id, agent_id, title, hypothesis, code, created_at)
           VALUES ('exp-1', 'agent-1', 'Solar Test', 'Solar panels are efficient', 'print(1)', '2025-06-01T00:00:00+00:00')"""
    )
    conn.execute(
        """INSERT INTO experiments (id, agent_id, title, hypothesis, code, created_at)
           VALUES ('exp-2', 'agent-1', 'Wind Test', 'Wind power scales', 'print(2)', '2025-06-02T00:00:00+00:00')"""
    )
    conn.execute(
        """INSERT INTO knowledge_nodes (id, experiment_id, summary, utility_contribution, tags, findings, created_at)
           VALUES ('kn-1', 'exp-1', 'Solar panels achieve 22% efficiency', 0.85, '["energy","solar"]', '{"key": "value"}', '2025-06-01T00:00:00+00:00')"""
    )
    conn.execute(
        """INSERT INTO knowledge_nodes (id, experiment_id, summary, utility_contribution, tags, created_at)
           VALUES ('kn-2', 'exp-2', 'Wind turbines scale linearly', 0.72, '["energy","wind"]', '2025-06-02T00:00:00+00:00')"""
    )
    conn.execute(
        """INSERT INTO knowledge_edges (id, from_node_id, to_node_id, relation, weight, created_by_agent_id)
           VALUES ('ke-1', 'kn-1', 'kn-2', 'builds_on', 1.0, 'agent-1')"""
    )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def max_db():
    """Create a minimal max database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE insights (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            evidence TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.5,
            domains TEXT DEFAULT '[]',
            implications TEXT DEFAULT '[]',
            time_horizon TEXT DEFAULT 'near_term',
            created_at TEXT NOT NULL
        );
        CREATE TABLE buildable_units (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            one_liner TEXT NOT NULL,
            category TEXT NOT NULL,
            domain TEXT DEFAULT '',
            ideation_mode TEXT DEFAULT 'direct',
            problem TEXT NOT NULL,
            solution TEXT NOT NULL,
            target_users TEXT DEFAULT 'both',
            specific_user TEXT DEFAULT '',
            buyer TEXT DEFAULT '',
            workflow_context TEXT DEFAULT '',
            current_workaround TEXT DEFAULT '',
            why_now TEXT DEFAULT '',
            validation_plan TEXT DEFAULT '',
            first_10_customers TEXT DEFAULT '',
            domain_risks TEXT DEFAULT '[]',
            evidence_rationale TEXT DEFAULT '',
            novelty_score REAL DEFAULT 0.0,
            usefulness_score REAL DEFAULT 0.0,
            quality_score REAL DEFAULT 0.0,
            rejection_tags TEXT DEFAULT '[]',
            value_proposition TEXT NOT NULL,
            inspiring_insights TEXT DEFAULT '[]',
            evidence_signals TEXT DEFAULT '[]',
            tech_approach TEXT DEFAULT '',
            suggested_stack TEXT DEFAULT '{}',
            composability_notes TEXT DEFAULT '',
            status TEXT DEFAULT 'draft',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE evaluations (
            buildable_unit_id TEXT PRIMARY KEY,
            overall_score REAL NOT NULL DEFAULT 0.0,
            recommendation TEXT NOT NULL DEFAULT 'maybe'
        );
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buildable_unit_id TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            dimension_values TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            approval_score INTEGER DEFAULT NULL
        );
        CREATE TABLE idea_critiques (
            id TEXT PRIMARY KEY,
            buildable_unit_id TEXT NOT NULL,
            dimensions TEXT NOT NULL DEFAULT '{}',
            reasoning TEXT NOT NULL DEFAULT '',
            rejection_tags TEXT NOT NULL DEFAULT '[]',
            evidence_pack TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE design_briefs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            domain TEXT NOT NULL,
            theme TEXT NOT NULL,
            readiness_score REAL NOT NULL DEFAULT 0.0,
            lead_idea_id TEXT NOT NULL,
            buyer TEXT NOT NULL,
            specific_user TEXT NOT NULL,
            workflow_context TEXT NOT NULL,
            why_this_now TEXT NOT NULL,
            merged_product_concept TEXT NOT NULL,
            synthesis_rationale TEXT NOT NULL,
            mvp_scope TEXT NOT NULL DEFAULT '[]',
            first_milestones TEXT NOT NULL DEFAULT '[]',
            validation_plan TEXT NOT NULL,
            risks TEXT NOT NULL DEFAULT '[]',
            source_idea_ids TEXT NOT NULL DEFAULT '[]',
            design_status TEXT NOT NULL DEFAULT 'draft',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE design_brief_sources (
            brief_id TEXT NOT NULL,
            idea_id TEXT NOT NULL,
            role TEXT NOT NULL,
            rank INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.execute(
        """INSERT INTO insights (id, category, title, summary, confidence, domains, created_at)
           VALUES ('ins-1', 'pain_point', 'API monitoring gap', 'Most teams lack proper API monitoring', 0.8, '["devtools","monitoring"]', '2025-05-01T00:00:00+00:00')"""
    )
    conn.execute(
        """INSERT INTO buildable_units (
               id, title, one_liner, category, domain, problem, solution,
               specific_user, buyer, workflow_context, validation_plan,
               value_proposition, inspiring_insights, status,
               novelty_score, usefulness_score, quality_score,
               created_at, updated_at
           )
           VALUES (
               'bu-1', 'API Monitor', 'Lightweight API health checker',
               'cli_tool', 'devtools', 'No simple API monitoring',
               'CLI tool for API health', 'platform engineer', 'VP Engineering',
               'post-deploy API verification', 'Interview 10 platform teams',
               'Save debugging time', '["ins-1"]', 'evaluated',
               0.7, 0.8, 7.5,
               '2025-05-02T00:00:00+00:00', '2025-05-02T00:00:00+00:00'
           )"""
    )
    conn.execute(
        """INSERT INTO evaluations (buildable_unit_id, overall_score, recommendation)
           VALUES ('bu-1', 82.0, 'yes')"""
    )
    conn.execute(
        """INSERT INTO feedback (buildable_unit_id, outcome, reason, created_at, approval_score)
           VALUES ('bu-1', 'approved', 'strong buyer clarity', '2025-05-03T00:00:00+00:00', 4)"""
    )
    conn.execute(
        """INSERT INTO idea_critiques (id, buildable_unit_id, dimensions, reasoning, rejection_tags, evidence_pack, created_at)
           VALUES (
               'crit-1', 'bu-1',
               '{"buyer_clarity": 0.9, "specificity": 0.8}',
               'Specific workflow and buyer',
               '[]',
               '{"domain": "devtools", "validated_gaps": ["ins-1"]}',
               '2025-05-02T12:00:00+00:00'
           )"""
    )
    conn.execute(
        """INSERT INTO design_briefs (
               id, title, domain, theme, readiness_score, lead_idea_id,
               buyer, specific_user, workflow_context, why_this_now,
               merged_product_concept, synthesis_rationale, mvp_scope,
               first_milestones, validation_plan, risks, source_idea_ids,
               design_status, created_at, updated_at
           )
           VALUES (
               'dbf-1', 'API Monitor', 'devtools',
               'agent security evaluation', 86.0, 'bu-1',
               'VP Engineering', 'platform engineer',
               'post-deploy API verification',
               'API reliability budgets are under pressure',
               'A focused API verification suite for release workflows',
               'Combines reviewed ideas around monitoring and validation',
               '["CLI smoke runner", "Failure summary"]',
               '["Interview platform teams", "Ship prototype"]',
               'Test with 10 platform teams in 2 weeks',
               '["Crowded monitoring market"]',
               '["bu-1"]',
               'draft',
               '2025-05-04T00:00:00+00:00',
               '2025-05-04T00:00:00+00:00'
           )"""
    )
    conn.execute(
        """INSERT INTO design_brief_sources (brief_id, idea_id, role, rank, created_at)
           VALUES ('dbf-1', 'bu-1', 'lead', 0, '2025-05-04T00:00:00+00:00')"""
    )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def presence_db():
    """Create a minimal presence database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id TEXT,
            source_url TEXT,
            author TEXT,
            content TEXT NOT NULL,
            insight TEXT,
            embedding BLOB,
            attribution_required INTEGER DEFAULT 1,
            approved INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_type, source_id)
        );
        CREATE TABLE generated_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_type TEXT NOT NULL,
            source_commits TEXT,
            source_messages TEXT,
            content TEXT NOT NULL,
            eval_score REAL,
            eval_feedback TEXT,
            published INTEGER DEFAULT 0,
            published_url TEXT,
            tweet_id TEXT,
            published_at TEXT,
            retry_count INTEGER DEFAULT 0,
            last_retry_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute(
        """INSERT INTO knowledge (source_type, source_id, content, insight, approved, created_at)
           VALUES ('own_post', 'tweet-1', 'Full tweet content here', 'Key insight about coding', 1, '2025-04-01T00:00:00')"""
    )
    conn.execute(
        """INSERT INTO knowledge (source_type, source_id, content, approved, created_at)
           VALUES ('curated_x', 'tweet-2', 'Unapproved content', 0, '2025-04-02T00:00:00')"""
    )
    conn.execute(
        """INSERT INTO generated_content (content_type, content, eval_score, published, published_url, created_at)
           VALUES ('x_post', 'Great post about async patterns', 8.5, 1, 'https://x.com/post/1', '2025-04-03T00:00:00')"""
    )
    conn.execute(
        """INSERT INTO generated_content (content_type, content, eval_score, published, created_at)
           VALUES ('x_post', 'Low quality post', 4.0, 1, '2025-04-04T00:00:00')"""
    )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def me_config():
    """Create a minimal me YAML config."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    data = {
        "version": "1.0",
        "projects": [
            {
                "id": "tact",
                "name": "TACT",
                "repoPath": "~/Project/experiments/tact",
                "description": "AI agent orchestration system",
                "enabled": True,
                "metadata": {
                    "url": "https://github.com/TakaIshikawa/tact",
                    "tags": ["ai-agents", "orchestration"],
                },
                "updateRules": {"lastUpdated": "2026-03-07T00:00:00Z"},
            },
            {
                "id": "disabled-project",
                "name": "Disabled",
                "repoPath": "~/somewhere",
                "description": "Should be skipped",
                "enabled": False,
            },
        ],
    }
    with open(path, "w") as f:
        yaml.dump(data, f)
    yield path
    os.unlink(path)


class TestFortyTwoAdapter:
    def test_ingest_nodes(self, forty_two_db: str):
        adapter = FortyTwoAdapter(db_path=forty_two_db)
        result = adapter.ingest()
        assert len(result.units) == 2
        assert result.units[0].source_project == "forty_two"
        assert result.units[0].content_type == "finding"
        assert result.units[0].tags == ["energy", "solar"]
        assert result.units[0].utility_score == 0.85

    def test_ingest_edges(self, forty_two_db: str):
        adapter = FortyTwoAdapter(db_path=forty_two_db)
        result = adapter.ingest()
        assert len(result.edges) == 1
        assert result.edges[0].relation == "builds_on"
        assert result.edges[0].source == "source"

    def test_ingest_with_entity_filter(self, forty_two_db: str):
        adapter = FortyTwoAdapter(db_path=forty_two_db)
        result = adapter.ingest(entity_types=["knowledge_node"])
        assert len(result.units) == 2
        assert len(result.edges) == 0

    def test_nonexistent_db(self):
        adapter = FortyTwoAdapter(db_path="/nonexistent/path.db")
        result = adapter.ingest()
        assert len(result.units) == 0


class TestMaxAdapter:
    def test_ingest_insights(self, max_db: str):
        adapter = MaxAdapter(db_path=max_db)
        result = adapter.ingest(entity_types=["insight"])
        assert len(result.units) == 1
        assert result.units[0].content_type == "insight"
        assert result.units[0].title == "API monitoring gap"
        assert result.units[0].confidence == 0.8

    def test_ingest_buildable_units_with_edges(self, max_db: str):
        adapter = MaxAdapter(db_path=max_db)
        result = adapter.ingest(entity_types=["buildable_unit"])
        ideas = [u for u in result.units if u.content_type == "idea"]
        assert len(ideas) == 1
        assert ideas[0].metadata["review_state"] == "approved"
        assert ideas[0].metadata["feedback_outcome"] == "approved"
        assert ideas[0].metadata["feedback_reason"] == "strong buyer clarity"
        assert ideas[0].metadata["is_approved"] is True
        assert ideas[0].metadata["buyer"] == "VP Engineering"
        assert ideas[0].metadata["critique_dimensions"]["buyer_clarity"] == 0.9
        assert ideas[0].metadata["evidence_pack"]["domain"] == "devtools"
        assert ideas[0].utility_score == 0.82
        assert "review-approved" in ideas[0].tags
        assert "approved" in ideas[0].tags
        assert len(result.edges) == 1
        assert result.edges[0].relation == "inspires"
        assert result.edges[0].from_unit_id == "ins-1"
        assert result.edges[0].to_unit_id == "bu-1"

    def test_ingest_design_briefs_with_source_edges(self, max_db: str):
        adapter = MaxAdapter(db_path=max_db)
        result = adapter.ingest(entity_types=["design_brief"])
        assert len(result.units) == 1
        brief = result.units[0]
        assert brief.content_type == "design_brief"
        assert brief.source_entity_type == "design_brief"
        assert brief.title == "API Monitor - Design Brief"
        assert brief.metadata["brief_title"] == "API Monitor"
        assert brief.metadata["lead_idea_id"] == "bu-1"
        assert brief.metadata["source_idea_ids"] == ["bu-1"]
        assert brief.metadata["mvp_scope"] == ["CLI smoke runner", "Failure summary"]
        assert brief.utility_score == 0.86
        assert "design-brief" in brief.tags
        assert "theme-agent-security-evaluation" in brief.tags

        assert len(result.edges) == 1
        edge = result.edges[0]
        assert edge.from_unit_id == "dbf-1"
        assert edge.to_unit_id == "bu-1"
        assert edge.relation == "derives_from"
        assert edge.metadata["from_entity_type"] == "design_brief"
        assert edge.metadata["to_entity_type"] == "buildable_unit"


class TestPresenceAdapter:
    def test_ingest_approved_knowledge(self, presence_db: str):
        adapter = PresenceAdapter(db_path=presence_db)
        result = adapter.ingest(entity_types=["knowledge_item"])
        assert len(result.units) == 1  # Only approved
        assert result.units[0].content == "Key insight about coding"

    def test_ingest_high_scoring_content(self, presence_db: str):
        adapter = PresenceAdapter(db_path=presence_db, min_score=7.0)
        result = adapter.ingest(entity_types=["generated_content"])
        assert len(result.units) == 1  # Only score >= 7.0
        assert result.units[0].content_type == "artifact"
        assert result.units[0].utility_score == 0.85


class TestMeAdapter:
    def test_ingest_projects(self, me_config: str):
        adapter = MeAdapter(config_path=me_config)
        result = adapter.ingest()
        assert len(result.units) == 1  # Disabled project skipped
        assert result.units[0].title == "TACT"
        assert result.units[0].content_type == "metadata"
        assert "ai-agents" in result.units[0].tags


class TestMarkdownAdapter:
    def test_ingest_markdown_notes_with_front_matter_tags_and_wikilinks(self, tmp_path):
        (tmp_path / "Some Note.md").write_text(
            "---\n"
            "title: Custom Title\n"
            "tags:\n"
            "  - front\n"
            "  - research\n"
            "---\n"
            "Body with #inline, #nested/tag. Links to [[Other Note]] and [[Missing]].\n",
            encoding="utf-8",
        )
        (tmp_path / "Other Note.md").write_text(
            "Other body with #other! and [[Custom Title|alias]].\n",
            encoding="utf-8",
        )

        result = MarkdownAdapter(root_path=str(tmp_path)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "Other Note.md",
            "Some Note.md",
        ]
        by_source = {unit.source_id: unit for unit in result.units}
        assert by_source["Some Note.md"].source_project == "me"
        assert by_source["Some Note.md"].source_entity_type == "markdown_note"
        assert by_source["Some Note.md"].title == "Custom Title"
        assert by_source["Some Note.md"].content.startswith("Body with")
        assert by_source["Some Note.md"].tags == ["front", "research", "inline", "nested/tag"]
        assert by_source["Other Note.md"].title == "Other Note"
        assert by_source["Other Note.md"].tags == ["other"]

        assert {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges} == {
            ("Some Note.md", "Other Note.md"),
            ("Other Note.md", "Some Note.md"),
        }
        assert all(edge.relation == "relates_to" for edge in result.edges)
        assert all(edge.metadata["to_entity_type"] == "markdown_note" for edge in result.edges)

    def test_missing_markdown_root_returns_empty_result(self, tmp_path):
        result = MarkdownAdapter(root_path=str(tmp_path / "missing")).ingest()

        assert result.units == []
        assert result.edges == []


class TestTextAdapter:
    def test_ingest_text_documents_recursively_with_titles_and_metadata(self, tmp_path):
        nested = tmp_path / "notes" / "nested"
        nested.mkdir(parents=True)
        first = tmp_path / "notes" / "draft.txt"
        second = nested / "transcript.txt"
        first.write_text("\n  Draft Title  \nBody search phrase.\n", encoding="utf-8")
        second.write_text("Transcript Title\nSecond body.\n", encoding="utf-8")
        (nested / "skip.md").write_text("Not plain text.\n", encoding="utf-8")

        result = TextAdapter(root_path=str(tmp_path / "notes")).ingest()

        assert [unit.source_id for unit in result.units] == [
            "draft.txt",
            "nested/transcript.txt",
        ]
        by_source = {unit.source_id: unit for unit in result.units}
        draft = by_source["draft.txt"]
        assert draft.source_project == "me"
        assert draft.source_entity_type == "text_document"
        assert draft.title == "Draft Title"
        assert draft.content == "\n  Draft Title  \nBody search phrase.\n"
        assert draft.metadata == {
            "path": "draft.txt",
            "file_size": first.stat().st_size,
        }
        assert draft.created_at.tzinfo is not None
        assert by_source["nested/transcript.txt"].title == "Transcript Title"
        assert result.edges == []

    def test_empty_missing_and_non_directory_roots_return_empty_result(self, tmp_path):
        empty = TextAdapter(root_path=str(tmp_path)).ingest()
        missing = TextAdapter(root_path=str(tmp_path / "missing")).ingest()
        file_root = tmp_path / "file.txt"
        file_root.write_text("Root file.\n", encoding="utf-8")
        non_directory = TextAdapter(root_path=str(file_root)).ingest()

        assert empty.units == []
        assert missing.units == []
        assert non_directory.units == []

    def test_title_falls_back_to_file_stem_and_sync_skips_old_files(self, tmp_path):
        old_path = tmp_path / "old.txt"
        new_path = tmp_path / "untitled.txt"
        old_path.write_text("Old Title\n", encoding="utf-8")
        new_path.write_text("\n\n   \n", encoding="utf-8")
        os.utime(old_path, (1_700_000_000, 1_700_000_000))
        os.utime(new_path, (1_700_100_000, 1_700_100_000))

        result = TextAdapter(root_path=str(tmp_path)).ingest(
            since=SyncState(
                source_project="text",
                source_entity_type="text_document",
                last_sync_at=datetime.fromtimestamp(1_700_050_000, tz=timezone.utc),
            )
        )

        assert [unit.source_id for unit in result.units] == ["untitled.txt"]
        assert result.units[0].title == "untitled"

    def test_entity_type_filter_skips_text_documents(self, tmp_path):
        (tmp_path / "note.txt").write_text("Note\n", encoding="utf-8")

        result = TextAdapter(root_path=str(tmp_path)).ingest(entity_types=["markdown_note"])

        assert result.units == []
        assert result.edges == []


class TestHtmlAdapter:
    def test_ingest_html_documents_recursively_with_metadata_and_tags(self, tmp_path):
        root = tmp_path / "html"
        nested = root / "nested"
        nested.mkdir(parents=True)
        page = nested / "page.html"
        page.write_text(
            """<!doctype html>
            <html>
              <head>
                <title>HTML Export Title</title>
                <meta name="description" content="Readable page summary">
                <meta name="keywords" content="docs, html, docs">
                <link rel="canonical" href="https://example.com/docs/page">
                <style>.hidden { color: red; }</style>
                <script>window.secret = "ignore me";</script>
              </head>
              <body>
                <h1>Fallback Heading</h1>
                <p>Readable text &amp; content.</p>
                <div>Nested search phrase.</div>
              </body>
            </html>
            """,
            encoding="utf-8",
        )
        (root / "brief.HTM").write_text("<h1>Brief Heading</h1><p>Brief body.</p>", encoding="utf-8")
        (root / "skip.txt").write_text("<h1>Not HTML</h1>", encoding="utf-8")

        result = HtmlAdapter(root_path=str(root)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "brief.HTM",
            "nested/page.html",
        ]
        by_source = {unit.source_id: unit for unit in result.units}
        unit = by_source["nested/page.html"]
        assert unit.source_project == "me"
        assert unit.source_entity_type == "html_document"
        assert unit.title == "HTML Export Title"
        assert "Readable text & content." in unit.content
        assert "Nested search phrase." in unit.content
        assert "ignore me" not in unit.content
        assert "hidden" not in unit.content
        assert unit.tags == ["docs", "html"]
        assert unit.metadata == {
            "path": "nested/page.html",
            "file_size": page.stat().st_size,
            "description": "Readable page summary",
            "canonical_url": "https://example.com/docs/page",
        }
        assert by_source["brief.HTM"].title == "Brief Heading"
        assert result.edges == []

    def test_malformed_html_does_not_abort_entire_ingest(self, tmp_path):
        (tmp_path / "good.html").write_text("<title>Good</title><p>Good body.</p>", encoding="utf-8")
        (tmp_path / "broken.html").write_text(
            "<html><head><title>Broken</title><body><h1>Broken Heading<p>Open tags",
            encoding="utf-8",
        )

        result = HtmlAdapter(root_path=str(tmp_path)).ingest()

        assert {unit.source_id for unit in result.units} == {"broken.html", "good.html"}
        assert {unit.title for unit in result.units} == {"Broken", "Good"}

    def test_missing_root_sync_and_entity_filter(self, tmp_path):
        old_path = tmp_path / "old.html"
        new_path = tmp_path / "new.html"
        old_path.write_text("<title>Old</title>", encoding="utf-8")
        new_path.write_text("<h1>New Heading</h1>", encoding="utf-8")
        os.utime(old_path, (1_700_000_000, 1_700_000_000))
        os.utime(new_path, (1_700_100_000, 1_700_100_000))

        result = HtmlAdapter(root_path=str(tmp_path)).ingest(
            since=SyncState(
                source_project="html",
                source_entity_type="html_document",
                last_sync_at=datetime.fromtimestamp(1_700_050_000, tz=timezone.utc),
            )
        )
        filtered = HtmlAdapter(root_path=str(tmp_path)).ingest(entity_types=["text_document"])
        missing = HtmlAdapter(root_path=str(tmp_path / "missing")).ingest()

        assert [unit.source_id for unit in result.units] == ["new.html"]
        assert result.units[0].title == "New Heading"
        assert filtered.units == []
        assert missing.units == []


class TestICalAdapter:
    def test_ingest_single_ics_event_with_metadata_tags_and_content(self, tmp_path):
        calendar = tmp_path / "calendar.ics"
        calendar.write_text(
            """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:event-1@example.com
SUMMARY:Planning Meeting
DESCRIPTION:Discuss launch plan\\nand follow-ups.
LOCATION:Conference Room
DTSTART:20260424T090000Z
DTEND:20260424T100000Z
CREATED:20260420T120000Z
LAST-MODIFIED:20260423T180000Z
ORGANIZER;CN=Alice:mailto:alice@example.com
ATTENDEE;CN=Bob:mailto:bob@example.com
CATEGORIES:work,planning
END:VEVENT
END:VCALENDAR
""",
            encoding="utf-8",
        )

        result = ICalAdapter(path=str(calendar)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.source_project == "me"
        assert unit.source_id == "calendar.ics#event-1@example.com"
        assert unit.source_entity_type == "calendar_event"
        assert unit.title == "Planning Meeting"
        assert unit.content_type == "artifact"
        assert "Discuss launch plan\nand follow-ups." in unit.content
        assert "Start: 2026-04-24T09:00:00+00:00" in unit.content
        assert "Location: Conference Room" in unit.content
        assert unit.tags == ["work", "planning"]
        assert unit.metadata == {
            "uid": "event-1@example.com",
            "start": "2026-04-24T09:00:00+00:00",
            "end": "2026-04-24T10:00:00+00:00",
            "location": "Conference Room",
            "organizer": "Alice <alice@example.com>",
            "attendees": ["Bob <bob@example.com>"],
            "source_path": "calendar.ics",
            "created": "2026-04-20T12:00:00+00:00",
            "updated": "2026-04-23T18:00:00+00:00",
        }
        assert unit.created_at.isoformat() == "2026-04-24T09:00:00+00:00"
        assert result.edges == []

    def test_folder_recurses_and_skips_malformed_events(self, tmp_path):
        nested = tmp_path / "calendars" / "nested"
        nested.mkdir(parents=True)
        (tmp_path / "calendars" / "root.ics").write_text(
            """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:root-event
SUMMARY:Root Event
DTSTART:20260425T120000Z
END:VEVENT
BEGIN:VEVENT
UID:bad-date
SUMMARY:Broken Event
DTSTART:not-a-date
END:VEVENT
END:VCALENDAR
""",
            encoding="utf-8",
        )
        (nested / "nested.ics").write_text(
            """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:nested-event
SUMMARY:Nested Event
DTSTART:20260426
END:VEVENT
END:VCALENDAR
""",
            encoding="utf-8",
        )
        (nested / "skip.txt").write_text("BEGIN:VEVENT\nUID:skip\nEND:VEVENT\n", encoding="utf-8")

        result = ICalAdapter(path=str(tmp_path / "calendars")).ingest()

        assert [unit.source_id for unit in result.units] == [
            "nested/nested.ics#nested-event",
            "root.ics#root-event",
        ]
        assert {unit.title for unit in result.units} == {"Nested Event", "Root Event"}

    def test_since_and_entity_type_filter_use_event_timestamps(self, tmp_path):
        calendar = tmp_path / "calendar.ics"
        calendar.write_text(
            """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:old-event
SUMMARY:Old Event
DTSTART:20260420T090000Z
CREATED:20260419T090000Z
LAST-MODIFIED:20260420T100000Z
END:VEVENT
BEGIN:VEVENT
UID:updated-event
SUMMARY:Updated Event
DTSTART:20260420T090000Z
LAST-MODIFIED:20260425T100000Z
END:VEVENT
BEGIN:VEVENT
UID:new-start-event
SUMMARY:New Start Event
DTSTART:20260426T090000Z
END:VEVENT
END:VCALENDAR
""",
            encoding="utf-8",
        )

        result = ICalAdapter(path=str(calendar)).ingest(
            since=SyncState(
                source_project="ical",
                source_entity_type="calendar_event",
                last_sync_at=datetime.fromisoformat("2026-04-24T00:00:00+00:00"),
            )
        )
        filtered = ICalAdapter(path=str(calendar)).ingest(entity_types=["feed_item"])

        assert [unit.source_id for unit in result.units] == [
            "calendar.ics#updated-event",
            "calendar.ics#new-start-event",
        ]
        assert filtered.units == []
        assert filtered.edges == []


class TestFeedAdapter:
    def test_ingest_rss_items_with_stable_metadata_and_tags(self, tmp_path):
        feed = tmp_path / "research.xml"
        feed.write_text(
            """<?xml version="1.0" encoding="utf-8"?>
            <rss version="2.0">
              <channel>
                <title>Research Feed</title>
                <item>
                  <guid isPermaLink="false">rss-1</guid>
                  <title>Solar storage update</title>
                  <link>https://example.com/solar</link>
                  <description><![CDATA[<p>Storage capacity doubled.</p>]]></description>
                  <author>alice@example.com</author>
                  <category>energy</category>
                  <category>solar</category>
                  <pubDate>Wed, 23 Apr 2025 10:30:00 GMT</pubDate>
                </item>
              </channel>
            </rss>
            """,
            encoding="utf-8",
        )

        first = FeedAdapter(sources=str(feed)).ingest()
        second = FeedAdapter(sources=str(feed)).ingest()

        assert len(first.units) == 1
        unit = first.units[0]
        assert unit.source_project == "me"
        assert unit.source_entity_type == "feed_item"
        assert unit.content_type == "artifact"
        assert unit.title == "Solar storage update"
        assert unit.content == "Storage capacity doubled."
        assert unit.source_id == second.units[0].source_id
        assert unit.tags == ["energy", "solar"]
        assert unit.metadata["feed_title"] == "Research Feed"
        assert unit.metadata["id"] == "rss-1"
        assert unit.metadata["link"] == "https://example.com/solar"
        assert unit.metadata["author"] == "alice@example.com"
        assert unit.created_at.isoformat() == "2025-04-23T10:30:00+00:00"

    def test_ingest_atom_entries_and_respects_entity_filter(self, tmp_path):
        feed = tmp_path / "atom.xml"
        feed.write_text(
            """<?xml version="1.0" encoding="utf-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
              <title>Atom Research</title>
              <entry>
                <id>tag:example.com,2025:atom-1</id>
                <title>Agent evaluation note</title>
                <link href="https://example.com/agent-eval"/>
                <updated>2025-04-24T12:00:00Z</updated>
                <author><name>Robin</name></author>
                <category term="agents"/>
                <category term="evaluation"/>
                <summary>Evaluation rubric changed.</summary>
              </entry>
            </feed>
            """,
            encoding="utf-8",
        )

        skipped = FeedAdapter(sources=str(feed)).ingest(entity_types=["markdown_note"])
        result = FeedAdapter(sources=str(feed)).ingest(entity_types=["feed_item"])

        assert skipped.units == []
        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.title == "Agent evaluation note"
        assert unit.content == "Evaluation rubric changed."
        assert unit.metadata["id"] == "tag:example.com,2025:atom-1"
        assert unit.metadata["link"] == "https://example.com/agent-eval"
        assert unit.metadata["author"] == "Robin"
        assert unit.tags == ["agents", "evaluation"]
        assert unit.created_at.isoformat() == "2025-04-24T12:00:00+00:00"


class TestBookmarksAdapter:
    def test_ingest_netscape_bookmarks_with_folder_metadata_and_url_source_id(
        self, tmp_path
    ):
        bookmarks = tmp_path / "bookmarks.html"
        bookmarks.write_text(
            """<!DOCTYPE NETSCAPE-Bookmark-file-1>
            <TITLE>Bookmarks</TITLE>
            <H1>Bookmarks</H1>
            <DL><p>
              <DT><H3 ADD_DATE="1713949200">Bookmarks Bar</H3>
              <DL><p>
                <DT><H3>Research</H3>
                <DL><p>
                  <DT><A HREF="https://example.com/agent-eval?ref=bookmarks"
                         ADD_DATE="1713952800"
                         LAST_MODIFIED="1713956400">Agent &amp; Evaluation</A>
                </DL><p>
              </DL><p>
            </DL><p>
            """,
            encoding="utf-8",
        )

        skipped = BookmarksAdapter(path=str(bookmarks)).ingest(
            entity_types=["feed_item"]
        )
        first = BookmarksAdapter(path=str(bookmarks)).ingest()
        second = BookmarksAdapter(path=str(bookmarks)).ingest()

        assert skipped.units == []
        assert len(first.units) == 1
        unit = first.units[0]
        assert unit.source_project == "bookmarks"
        assert unit.source_entity_type == "bookmark"
        assert unit.source_id == "https://example.com/agent-eval?ref=bookmarks"
        assert unit.source_id == second.units[0].source_id
        assert unit.title == "Agent & Evaluation"
        assert "https://example.com/agent-eval?ref=bookmarks" in unit.content
        assert unit.content_type == "artifact"
        assert unit.tags == ["Bookmarks Bar", "Bookmarks Bar/Research"]
        assert unit.metadata == {
            "url": "https://example.com/agent-eval?ref=bookmarks",
            "folder_path": "Bookmarks Bar/Research",
            "add_date": "1713952800",
            "last_modified": "1713956400",
        }
        assert unit.created_at == datetime.fromtimestamp(1713952800, tz=timezone.utc)
        assert unit.updated_at == datetime.fromtimestamp(1713956400, tz=timezone.utc)

    def test_missing_bookmarks_path_returns_empty_result(self, tmp_path):
        result = BookmarksAdapter(path=str(tmp_path / "missing.html")).ingest()

        assert result.units == []
        assert result.edges == []


class TestCsvAdapter:
    def test_ingest_csv_rows_with_optional_fields(self, tmp_path):
        csv_path = tmp_path / "notes.csv"
        csv_path.write_text(
            "source_id,title,content,content_type,tags,utility_score,confidence,created_at,metadata_json\n"
            'note-1,Solar note,"Storage doubled.",finding,"energy, solar",8.5,0.75,2025-04-24T12:00:00Z,"{""url"": ""https://example.com"", ""rank"": 3}"\n',
            encoding="utf-8",
        )

        result = CsvAdapter(path=str(csv_path)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.source_project == "csv"
        assert unit.source_entity_type == "csv_row"
        assert unit.source_id == "note-1"
        assert unit.title == "Solar note"
        assert unit.content == "Storage doubled."
        assert unit.content_type == "finding"
        assert unit.tags == ["energy", "solar"]
        assert unit.utility_score == 8.5
        assert unit.confidence == 0.75
        assert unit.created_at.isoformat() == "2025-04-24T12:00:00+00:00"
        assert unit.metadata == {"url": "https://example.com", "rank": 3}

    def test_missing_optional_columns_and_source_id_are_handled(self, tmp_path):
        csv_path = tmp_path / "minimal.csv"
        csv_path.write_text(
            "title,content\nMinimal row,Only required columns.\n",
            encoding="utf-8",
        )

        first = CsvAdapter(path=str(csv_path)).ingest()
        second = CsvAdapter(path=str(csv_path)).ingest()

        assert len(first.units) == 1
        unit = first.units[0]
        assert unit.source_id == second.units[0].source_id
        assert unit.source_id.startswith("row-2-minimal-row-")
        assert unit.content_type == "insight"
        assert unit.tags == []
        assert unit.metadata == {}
        assert unit.utility_score is None
        assert unit.confidence is None

    def test_malformed_metadata_falls_back_without_crashing(self, tmp_path):
        csv_path = tmp_path / "bad-metadata.csv"
        csv_path.write_text(
            "title,content,metadata_json\nBad metadata,Still imported,{not json\n",
            encoding="utf-8",
        )

        result = CsvAdapter(path=str(csv_path)).ingest()

        assert len(result.units) == 1
        assert result.units[0].metadata == {"metadata_json": "{not json"}

    def test_missing_path_and_missing_required_headers_return_empty_result(self, tmp_path):
        missing = CsvAdapter(path=str(tmp_path / "missing.csv")).ingest()
        assert missing.units == []
        assert missing.edges == []

        csv_path = tmp_path / "no-content.csv"
        csv_path.write_text("title,tags\nNo content,tag\n", encoding="utf-8")
        malformed = CsvAdapter(path=str(csv_path)).ingest()
        assert malformed.units == []
        assert malformed.edges == []


class TestJsonlAdapter:
    def test_ingest_jsonl_records_with_optional_fields(self, tmp_path):
        jsonl_path = tmp_path / "notes.jsonl"
        jsonl_path.write_text(
            json.dumps(
                {
                    "source_id": "jsonl-1",
                    "title": "JSONL note",
                    "content": "Structured export content.",
                    "content_type": "finding",
                    "tags": ["energy", "#solar", "energy"],
                    "utility_score": 8.7,
                    "confidence": "0.81",
                    "created_at": "2025-04-24T12:00:00Z",
                    "updated_at": "2025-04-25T09:30:00Z",
                    "metadata": {"url": "https://example.com", "rank": 3},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = JsonlAdapter(path=str(jsonl_path)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.source_project == "jsonl"
        assert unit.source_entity_type == "jsonl_record"
        assert unit.source_id == "jsonl-1"
        assert unit.title == "JSONL note"
        assert unit.content == "Structured export content."
        assert unit.content_type == "finding"
        assert unit.tags == ["energy", "solar"]
        assert unit.utility_score == 8.7
        assert unit.confidence == 0.81
        assert unit.created_at.isoformat() == "2025-04-24T12:00:00+00:00"
        assert unit.updated_at.isoformat() == "2025-04-25T09:30:00+00:00"
        assert unit.metadata == {"url": "https://example.com", "rank": 3}

    def test_malformed_json_lines_are_skipped_with_warning(self, tmp_path):
        jsonl_path = tmp_path / "mixed.jsonl"
        jsonl_path.write_text(
            '{"source_id": "ok", "title": "Valid", "content": "Imported."}\n'
            "{not json\n"
            '["not", "object"]\n',
            encoding="utf-8",
        )

        with pytest.warns(UserWarning, match="Skipped 2 malformed JSONL line"):
            result = JsonlAdapter(path=str(jsonl_path)).ingest()

        assert [unit.source_id for unit in result.units] == ["ok"]

    def test_missing_required_fields_and_entity_filter_return_empty_result(self, tmp_path):
        jsonl_path = tmp_path / "missing.jsonl"
        jsonl_path.write_text(
            '{"title": "No source", "content": "Skipped."}\n'
            '{"source_id": "no-title", "content": "Skipped."}\n'
            '{"source_id": "no-content", "title": "Skipped"}\n',
            encoding="utf-8",
        )

        filtered = JsonlAdapter(path=str(jsonl_path)).ingest(
            entity_types=["csv_row"]
        )
        missing = JsonlAdapter(path=str(jsonl_path)).ingest()

        assert filtered.units == []
        assert filtered.edges == []
        assert missing.units == []
        assert missing.edges == []


class TestOpmlAdapter:
    def test_ingest_nested_outlines_with_urls_and_edges(self, tmp_path):
        opml_path = tmp_path / "feeds.opml"
        opml_path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <opml version="2.0">
              <body>
                <outline text="Engineering">
                  <outline text="Python Weekly" type="rss"
                    xmlUrl="https://example.com/python.xml"
                    htmlUrl="https://example.com/python"
                    url="https://example.com/python-page" />
                  <outline title="Reading List">
                    <outline text="Graph Notes" url="https://example.com/graph" />
                  </outline>
                </outline>
              </body>
            </opml>
            """,
            encoding="utf-8",
        )

        result = OpmlAdapter(path=str(opml_path)).ingest()

        assert [unit.title for unit in result.units] == [
            "Engineering",
            "Python Weekly",
            "Reading List",
            "Graph Notes",
        ]
        feed = result.units[1]
        assert feed.source_project == "opml"
        assert feed.source_entity_type == "outline"
        assert feed.metadata["url"] == "https://example.com/python-page"
        assert feed.metadata["xmlUrl"] == "https://example.com/python.xml"
        assert feed.metadata["htmlUrl"] == "https://example.com/python"
        assert "https://example.com/python.xml" in feed.content
        assert feed.tags == ["Engineering", "Engineering/Python Weekly"]

        assert len(result.edges) == 3
        assert all(edge.relation == "contains" for edge in result.edges)
        assert all(edge.source == "source" for edge in result.edges)
        assert result.edges[0].metadata["from_entity_type"] == "outline"
        assert result.edges[0].from_unit_id == result.units[0].source_id
        assert result.edges[0].to_unit_id == result.units[1].source_id

    def test_missing_and_malformed_opml_return_empty_result_with_diagnostic(self, tmp_path):
        missing = tmp_path / "missing.opml"
        with pytest.warns(UserWarning, match="No OPML files found"):
            missing_result = OpmlAdapter(path=str(missing)).ingest()
        assert missing_result.units == []
        assert missing_result.edges == []

        malformed = tmp_path / "bad.opml"
        malformed.write_text("<opml><body><outline text='broken'></body>", encoding="utf-8")
        with pytest.warns(UserWarning, match="Skipping invalid OPML file"):
            malformed_result = OpmlAdapter(path=str(malformed)).ingest()
        assert malformed_result.units == []
        assert malformed_result.edges == []

    def test_opml_edges_are_inserted_by_store(self, tmp_path):
        opml_path = tmp_path / "outline.opml"
        opml_path.write_text(
            "<opml><body><outline text='Root'><outline text='Child' /></outline></body></opml>",
            encoding="utf-8",
        )
        result = OpmlAdapter(path=str(opml_path)).ingest()
        store = Store(str(tmp_path / "graph.db"))
        try:
            stats = store.ingest(result, "opml")
            edges = store.get_all_edges()
        finally:
            store.close()

        assert stats == {"units_inserted": 2, "units_skipped": 0, "edges_inserted": 1}
        assert len(edges) == 1
        assert edges[0].relation == "contains"


class TestRegistry:
    def test_list_adapters(self):
        adapters = list_adapters()
        assert set(adapters) == {
            "forty_two",
            "max",
            "presence",
            "me",
            "markdown",
            "kindle",
            "sota",
            "feed",
            "bookmarks",
            "csv",
            "jsonl",
            "opml",
            "text",
            "html",
            "ical",
        }

    def test_get_adapter(self):
        adapter = get_adapter("me", config_path="/tmp/test.yaml")
        assert adapter.name == "me"

        jsonl_adapter = get_adapter("jsonl", path="/tmp/test.jsonl")
        assert jsonl_adapter.name == "jsonl"

        opml_adapter = get_adapter("opml", path="/tmp/test.opml")
        assert opml_adapter.name == "opml"

        text_adapter = get_adapter("text", root_path="/tmp/text")
        assert text_adapter.name == "text"

        html_adapter = get_adapter("html", root_path="/tmp/html")
        assert html_adapter.name == "html"

        ical_adapter = get_adapter("ical", path="/tmp/calendar.ics")
        assert ical_adapter.name == "ical"

        feed_adapter = get_adapter("feed", sources="/tmp/feed.xml")
        assert feed_adapter.name == "feed"

    def test_unknown_adapter(self):
        with pytest.raises(KeyError):
            get_adapter("unknown")
