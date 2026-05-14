"""Tests for source adapters."""

from __future__ import annotations

import builtins
import json
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from graph.adapters.atom import AtomAdapter
from graph.adapters.bibdesk import BibDeskAdapter
from graph.adapters.bibtex import BibtexAdapter
from graph.adapters.bluesky_archive import BlueskyArchiveAdapter
from graph.adapters.bookmarks import BookmarksAdapter
from graph.adapters.bookmarks_html import BookmarksHtmlAdapter
from graph.adapters.browser_history_csv import BrowserHistoryCsvAdapter
from graph.adapters.chatgpt_json import ChatGptJsonAdapter
from graph.adapters.chrome_history import ChromeHistoryAdapter
from graph.adapters.csv_adapter import CsvAdapter
from graph.adapters.csv_rows import CsvRowsAdapter
from graph.adapters.crossref import CrossrefAdapter
from graph.adapters.csl_json import CslJsonAdapter
from graph.adapters.daily_journal import DailyJournalAdapter
from graph.adapters.discord_json import DiscordJsonAdapter
from graph.adapters.email import EmailAdapter
from graph.adapters.enex import EnexAdapter
from graph.adapters.feed import FeedAdapter
from graph.adapters.forty_two import FortyTwoAdapter
from graph.adapters.git_adapter import GitAdapter
from graph.adapters.google_keep import GoogleKeepAdapter
from graph.adapters.html import HtmlAdapter
from graph.adapters.hypothesis import HypothesisAdapter
from graph.adapters.ical import ICalAdapter
from graph.adapters.ipynb import IpynbAdapter
from graph.adapters.jsonl_adapter import JsonlAdapter
from graph.adapters.jsonl_notes import JsonlNotesAdapter
from graph.adapters.logseq import LogseqAdapter
from graph.adapters.markdown import MarkdownAdapter
from graph.adapters.markdown_callouts import MarkdownCalloutsAdapter
from graph.adapters.markdown_definitions import MarkdownDefinitionsAdapter
from graph.adapters.markdown_frontmatter import MarkdownFrontmatterAdapter
from graph.adapters.markdown_links import MarkdownLinksAdapter
from graph.adapters.markdown_notes import MarkdownNotesAdapter
from graph.adapters.markdown_tasks import MarkdownTasksAdapter
from graph.adapters.mastodon import MastodonAdapter
from graph.adapters.max_adapter import MaxAdapter
from graph.adapters.me import MeAdapter
from graph.adapters.mediawiki import MediaWikiAdapter
from graph.adapters.notion_markdown import NotionMarkdownAdapter
from graph.adapters.opml import OpmlAdapter
from graph.adapters.obsidian_canvas import ObsidianCanvasAdapter
from graph.adapters.org import OrgAdapter
from graph.adapters.pdf import PdfAdapter
from graph.adapters.pinboard import PinboardAdapter
from graph.adapters.plain_text import PlainTextAdapter
from graph.adapters.pocket_csv import PocketCsvAdapter
from graph.adapters.presence import PresenceAdapter
from graph.adapters.raindrop import RaindropAdapter
from graph.adapters.raindrop_csv import RaindropCsvAdapter
from graph.adapters.raindrop_json import RaindropJsonAdapter
from graph.adapters.readwise import ReadwiseAdapter
from graph.adapters.readwise_csv import ReadwiseCsvAdapter
from graph.adapters.reddit_saved_csv import RedditSavedCsvAdapter
from graph.adapters.registry import _ADAPTERS, get_adapter, get_all_adapters, list_adapters
from graph.adapters.ris import RisAdapter
from graph.adapters.roam import RoamAdapter
from graph.adapters.safari_bookmarks import SafariBookmarksAdapter
from graph.adapters.slack_json import SlackJsonAdapter
from graph.adapters.sqlite_query_log import SqliteQueryLogAdapter
from graph.adapters.spotify_takeout import SpotifyTakeoutAdapter
from graph.adapters.text import TextAdapter
from graph.adapters.text_outline import TextOutlineAdapter
from graph.adapters.tana_paste import TanaPasteAdapter
from graph.adapters.transcript import TranscriptAdapter
from graph.adapters.twitter_archive import TwitterArchiveAdapter
from graph.adapters.webvtt import WebVttAdapter
from graph.adapters.yaml_adapter import YamlAdapter
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
            "created_at: 2024-01-02T03:04:05-05:00\n"
            "updated_at: 2024-02-03\n"
            "content_type: finding\n"
            "confidence: 0.75\n"
            "utility_score: '0.8'\n"
            "status: evergreen\n"
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
        assert "---" not in by_source["Some Note.md"].content
        assert by_source["Some Note.md"].content_type == "finding"
        assert by_source["Some Note.md"].confidence == 0.75
        assert by_source["Some Note.md"].utility_score == 0.8
        assert by_source["Some Note.md"].created_at == datetime(
            2024, 1, 2, 8, 4, 5, tzinfo=timezone.utc
        )
        assert by_source["Some Note.md"].updated_at == datetime(
            2024, 2, 3, 0, 0, 0, tzinfo=timezone.utc
        )
        assert by_source["Some Note.md"].metadata["front_matter"] == {
            "status": "evergreen"
        }
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

    def test_ingest_markdown_note_without_front_matter_keeps_existing_behavior(self, tmp_path):
        note = tmp_path / "Plain.md"
        note.write_text("Plain body with #tag.\n", encoding="utf-8")

        result = MarkdownAdapter(root_path=str(tmp_path)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.source_id == "Plain.md"
        assert unit.title == "Plain"
        assert unit.content == "Plain body with #tag.\n"
        assert unit.content_type == "insight"
        assert unit.tags == ["tag"]
        assert unit.metadata == {"path": "Plain.md", "front_matter": {}}
        assert unit.created_at.tzinfo is not None
        assert unit.updated_at.tzinfo is not None

    def test_ingest_obsidian_folder_uses_vault_relative_ids_and_can_exclude_tags(
        self, tmp_path
    ):
        vault = tmp_path / "vault"
        folder = vault / "Projects"
        nested = folder / "Nested"
        nested.mkdir(parents=True)
        (folder / "Alpha.md").write_text(
            "---\n"
            "title: Alpha Title\n"
            "tags: [front]\n"
            "---\n"
            "Alpha links to [[Beta]] and [[Nested/Beta]]. #inline\n",
            encoding="utf-8",
        )
        (nested / "Beta.md").write_text("Beta body.\n", encoding="utf-8")
        (vault / "Outside.md").write_text("Outside body.\n", encoding="utf-8")

        result = MarkdownAdapter(
            root_path=str(folder),
            source_project="obsidian",
            source_id_root=str(vault),
            include_tags=False,
        ).ingest()

        assert [unit.source_id for unit in result.units] == [
            "Projects/Alpha.md",
            "Projects/Nested/Beta.md",
        ]
        alpha = next(unit for unit in result.units if unit.source_id == "Projects/Alpha.md")
        assert alpha.source_project == "obsidian"
        assert alpha.metadata["path"] == "Projects/Alpha.md"
        assert alpha.metadata["front_matter"] == {}
        assert alpha.tags == []
        assert {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges} == {
            ("Projects/Alpha.md", "Projects/Nested/Beta.md")
        }
        assert result.edges[0].metadata["source_project"] == "obsidian"


class TestOrgAdapter:
    def test_ingest_org_headings_with_tags_properties_and_links(self, tmp_path):
        (tmp_path / "Alpha.org").write_text(
            "* TODO Research Plan :research:python:\n"
            ":PROPERTIES:\n"
            ":CUSTOM_ID: plan\n"
            ":Owner: Taka\n"
            ":END:\n"
            "Body links to [[file:Beta.org::*Implementation][implementation]], "
            "[[#local]], and [[file:Missing.org]].\n"
            "** Local Details :local:\n"
            ":PROPERTIES:\n"
            ":CUSTOM_ID: local\n"
            ":END:\n"
            "Nested detail.\n",
            encoding="utf-8",
        )
        (tmp_path / "Beta.org").write_text(
            "* Implementation :build:\n"
            "Backlink to [[file:Alpha.org::#plan]].\n",
            encoding="utf-8",
        )

        result = OrgAdapter(root_path=str(tmp_path)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "Alpha.org#research-plan",
            "Alpha.org#local-details",
            "Beta.org#implementation",
        ]
        by_source = {unit.source_id: unit for unit in result.units}
        plan = by_source["Alpha.org#research-plan"]
        assert plan.source_project == "org"
        assert plan.source_entity_type == "org_heading"
        assert plan.title == "Research Plan"
        assert plan.tags == ["research", "python"]
        assert plan.metadata["path"] == "Alpha.org"
        assert plan.metadata["heading_level"] == 1
        assert plan.metadata["line"] == 1
        assert plan.metadata["properties"] == {
            "CUSTOM_ID": "plan",
            "Owner": "Taka",
        }
        assert "Body links to" in plan.content
        assert "Local Details" not in plan.content
        assert by_source["Alpha.org#local-details"].tags == ["local"]
        assert by_source["Alpha.org#local-details"].metadata["heading_level"] == 2

        assert {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges} == {
            ("Alpha.org#research-plan", "Beta.org#implementation"),
            ("Alpha.org#research-plan", "Alpha.org#local-details"),
            ("Beta.org#implementation", "Alpha.org#research-plan"),
        }
        assert all(edge.relation == "relates_to" for edge in result.edges)
        assert all(edge.metadata["relation_type"] == "org_link" for edge in result.edges)

    def test_org_adapter_keeps_unsupported_org_syntax_as_content(self, tmp_path):
        (tmp_path / "Agenda.org").write_text(
            "* Agenda\n"
            "SCHEDULED: <2026-05-01 Fri>\n"
            "- [ ] checkbox syntax is retained as plain content\n"
            "#+BEGIN_SRC python\n"
            "print('not parsed as a block')\n"
            "#+END_SRC\n",
            encoding="utf-8",
        )

        result = OrgAdapter(root_path=str(tmp_path)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.metadata["properties"] == {}
        assert "SCHEDULED: <2026-05-01 Fri>" in unit.content
        assert "#+BEGIN_SRC python" in unit.content

    def test_org_adapter_respects_entity_filter_and_missing_root(self, tmp_path):
        (tmp_path / "Note.org").write_text("* Note\nBody.\n", encoding="utf-8")

        filtered = OrgAdapter(root_path=str(tmp_path)).ingest(entity_types=["markdown_note"])
        missing = OrgAdapter(root_path=str(tmp_path / "missing")).ingest()

        assert filtered.units == []
        assert filtered.edges == []
        assert missing.units == []
        assert missing.edges == []


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


class TestTranscriptAdapter:
    def test_ingest_srt_and_vtt_recursively_with_metadata(self, tmp_path):
        root = tmp_path / "transcripts"
        nested = root / "meetings"
        nested.mkdir(parents=True)
        srt = root / "lecture.srt"
        vtt = nested / "standup.vtt"
        srt.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:03,500\n"
            "Welcome to the lecture.\n"
            "\n"
            "2\n"
            "00:00:05,250 --> 00:00:07,000\n"
            "Second cue search phrase.\n",
            encoding="utf-8",
        )
        vtt.write_text(
            "WEBVTT\n"
            "\n"
            "intro\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "Daily notes.\n"
            "\n"
            "00:00:04.500 --> 00:00:06.000\n"
            "Ship the transcript adapter.\n",
            encoding="utf-8",
        )
        (nested / "skip.txt").write_text("Not a transcript.\n", encoding="utf-8")

        result = TranscriptAdapter(root_path=str(root)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "lecture.srt",
            "meetings/standup.vtt",
        ]
        by_source = {unit.source_id: unit for unit in result.units}
        lecture = by_source["lecture.srt"]
        assert lecture.source_project == "transcript"
        assert lecture.source_entity_type == "transcript"
        assert lecture.title == "lecture"
        assert lecture.content == "Welcome to the lecture.\n\nSecond cue search phrase."
        assert lecture.content_type == "artifact"
        assert lecture.metadata == {
            "path": "lecture.srt",
            "source_path": "lecture.srt",
            "file_size": srt.stat().st_size,
            "transcript_format": "srt",
            "cue_count": 2,
            "first_timestamp": "00:00:01.000",
            "last_timestamp": "00:00:07.000",
            "duration_range": "00:00:01.000 --> 00:00:07.000",
        }
        assert lecture.created_at.tzinfo is not None
        assert by_source["meetings/standup.vtt"].metadata["transcript_format"] == "vtt"
        assert by_source["meetings/standup.vtt"].metadata["cue_count"] == 2
        assert "Ship the transcript adapter." in by_source["meetings/standup.vtt"].content
        assert result.edges == []

    def test_malformed_cues_are_skipped_while_valid_cues_ingest(self, tmp_path):
        transcript = tmp_path / "mixed.srt"
        transcript.write_text(
            "1\n"
            "not a timestamp\n"
            "This block is ignored.\n"
            "\n"
            "2\n"
            "00:00:10,000 --> 00:00:12,000\n"
            "Valid cue survives.\n"
            "\n"
            "3\n"
            "00:99:13,000 --> 00:00:14,000\n"
            "Invalid minutes are ignored.\n"
            "\n"
            "4\n"
            "00:00:15,000 --> 00:00:16,000\n"
            "\n",
            encoding="utf-8",
        )

        result = TranscriptAdapter(root_path=str(tmp_path)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.content == "Valid cue survives."
        assert unit.metadata["cue_count"] == 1
        assert unit.metadata["first_timestamp"] == "00:00:10.000"
        assert unit.metadata["last_timestamp"] == "00:00:12.000"

    def test_entity_filter_missing_root_and_no_valid_cues_return_empty_result(self, tmp_path):
        bad_root = tmp_path / "bad"
        bad_root.mkdir()
        (tmp_path / "note.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nValid.\n",
            encoding="utf-8",
        )
        (bad_root / "bad.vtt").write_text("WEBVTT\n\ninvalid cue\n", encoding="utf-8")

        filtered = TranscriptAdapter(root_path=str(tmp_path)).ingest(entity_types=["text_document"])
        missing = TranscriptAdapter(root_path=str(tmp_path / "missing")).ingest()
        no_valid = TranscriptAdapter(root_path=str(bad_root)).ingest()

        assert filtered.units == []
        assert filtered.edges == []
        assert missing.units == []
        assert missing.edges == []
        assert no_valid.units == []


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


class TestEmailAdapter:
    def test_ingest_plain_text_email_with_headers_and_metadata(self, tmp_path):
        root = tmp_path / "mail"
        nested = root / "archive"
        nested.mkdir(parents=True)
        message = nested / "hello.eml"
        message.write_text(
            "\n".join(
                [
                    "From: Alice <alice@example.com>",
                    "To: Bob <bob@example.com>",
                    "Cc: Carol <carol@example.com>",
                    "Date: Mon, 01 Apr 2024 10:30:00 +0000",
                    "Message-ID: <hello@example.com>",
                    "Subject: Project Hello",
                    "Content-Type: text/plain; charset=utf-8",
                    "",
                    "Plain email search phrase.",
                    "Second line.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "skip.txt").write_text("Not email.\n", encoding="utf-8")

        result = EmailAdapter(path=str(root)).ingest()

        assert [unit.source_id for unit in result.units] == ["archive/hello.eml"]
        unit = result.units[0]
        assert unit.source_project == "email"
        assert unit.source_entity_type == "email_message"
        assert unit.title == "Project Hello"
        assert unit.content == "Plain email search phrase.\nSecond line."
        assert "Subject:" not in unit.content
        assert unit.content_type == "artifact"
        assert unit.metadata == {
            "from": "Alice <alice@example.com>",
            "to": "Bob <bob@example.com>",
            "cc": "Carol <carol@example.com>",
            "date": "Mon, 01 Apr 2024 10:30:00 +0000",
            "message_id": "<hello@example.com>",
            "path": "archive/hello.eml",
            "file_size": message.stat().st_size,
            "attachment_count": 0,
        }
        assert unit.created_at.isoformat() == "2024-04-01T10:30:00+00:00"
        assert result.edges == []

    def test_multipart_prefers_plain_text_and_html_fallback(self, tmp_path):
        plain = tmp_path / "plain.eml"
        plain.write_text(
            """From: sender@example.com
To: reader@example.com
Subject: Multipart Plain
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="ALT"

--ALT
Content-Type: text/html; charset=utf-8

<html><body><p>HTML should not win.</p></body></html>
--ALT
Content-Type: text/plain; charset=utf-8

Preferred plain body.
--ALT--
""",
            encoding="utf-8",
        )
        html_only = tmp_path / "html-only.eml"
        html_only.write_text(
            """From: sender@example.com
To: reader@example.com
Subject: HTML Only
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="ALT"

--ALT
Content-Type: text/html; charset=utf-8

<html><body><h1>Rendered Heading</h1><script>ignore me</script><p>HTML fallback phrase &amp; detail.</p></body></html>
--ALT--
""",
            encoding="utf-8",
        )

        result = EmailAdapter(path=str(tmp_path)).ingest()
        by_source = {unit.source_id: unit for unit in result.units}

        assert by_source["plain.eml"].content == "Preferred plain body."
        assert "HTML should not win" not in by_source["plain.eml"].content
        assert "Rendered Heading" in by_source["html-only.eml"].content
        assert "HTML fallback phrase & detail." in by_source["html-only.eml"].content
        assert "ignore me" not in by_source["html-only.eml"].content

    def test_missing_subject_file_path_custom_source_and_filters(self, tmp_path):
        old_path = tmp_path / "old.eml"
        new_path = tmp_path / "untitled.eml"
        old_path.write_text("From: old@example.com\n\nOld body.\n", encoding="utf-8")
        new_path.write_text("From: new@example.com\n\nNew body.\n", encoding="utf-8")
        os.utime(old_path, (1_700_000_000, 1_700_000_000))
        os.utime(new_path, (1_700_100_000, 1_700_100_000))

        result = EmailAdapter(path=str(new_path), source_project="mail_archive").ingest()
        filtered = EmailAdapter(path=str(new_path)).ingest(entity_types=["text_document"])
        synced = EmailAdapter(path=str(tmp_path)).ingest(
            since=SyncState(
                source_project="email",
                source_entity_type="email_message",
                last_sync_at=datetime.fromtimestamp(1_700_050_000, tz=timezone.utc),
            )
        )
        missing = EmailAdapter(path=str(tmp_path / "missing")).ingest()

        assert len(result.units) == 1
        assert result.units[0].source_project == "mail_archive"
        assert result.units[0].source_id == "untitled.eml"
        assert result.units[0].title == "untitled"
        assert filtered.units == []
        assert [unit.source_id for unit in synced.units] == ["untitled.eml"]
        assert missing.units == []

    def test_multipart_with_attachments_extracts_metadata(self, tmp_path):
        email_path = tmp_path / "attachments.eml"
        email_path.write_text(
            """From: sender@example.com
To: receiver@example.com
Subject: Files Attached
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="MIXED"

--MIXED
Content-Type: text/plain; charset=utf-8

Email body with attachments.
--MIXED
Content-Type: application/pdf; name="document.pdf"
Content-Disposition: attachment; filename="document.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4K
ZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFsgXQovQ291bnQgMAo+PgplbmRv
YmoKeHJlZgowIDMKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDE1IDAwMDAwIG4gCjAwMDAw
MDAwNjQgMDAwMDAgbiAKdHJhaWxlcgo8PAovU2l6ZSAzCi9Sb290IDEgMCBSCj4+CnN0YXJ0eHJl
ZgoxMTMKJSVFT0YK
--MIXED
Content-Type: image/png
Content-Disposition: attachment; filename="chart.png"
Content-Transfer-Encoding: base64

iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9
awAAAABJRU5ErkJggg==
--MIXED
Content-Type: application/octet-stream
Content-Disposition: attachment
Content-Transfer-Encoding: base64

AQIDBA==
--MIXED--
""",
            encoding="utf-8",
        )

        result = EmailAdapter(path=str(email_path)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.content == "Email body with attachments."
        assert "JVBERi" not in unit.content
        assert "iVBORw0K" not in unit.content

        metadata = unit.metadata
        assert metadata["attachment_count"] == 3
        assert "attachments" in metadata
        attachments = metadata["attachments"]
        assert len(attachments) == 3

        # First attachment: document.pdf
        pdf_attachment = attachments[0]
        assert pdf_attachment["filename"] == "document.pdf"
        assert pdf_attachment["content_type"] == "application/pdf"
        assert pdf_attachment["content_disposition"] == "attachment"
        assert pdf_attachment["size_bytes"] == 240

        # Second attachment: chart.png
        png_attachment = attachments[1]
        assert png_attachment["filename"] == "chart.png"
        assert png_attachment["content_type"] == "image/png"
        assert png_attachment["content_disposition"] == "attachment"
        assert png_attachment["size_bytes"] == 70

        # Third attachment: no filename
        octet_attachment = attachments[2]
        assert "filename" not in octet_attachment
        assert octet_attachment["content_type"] == "application/octet-stream"
        assert octet_attachment["content_disposition"] == "attachment"
        assert octet_attachment["size_bytes"] == 4

    def test_email_without_attachments_has_zero_count(self, tmp_path):
        email_path = tmp_path / "no-attachments.eml"
        email_path.write_text(
            """From: sender@example.com
To: receiver@example.com
Subject: Plain Email
Content-Type: text/plain; charset=utf-8

Just a plain email with no attachments.
""",
            encoding="utf-8",
        )

        result = EmailAdapter(path=str(email_path)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.metadata["attachment_count"] == 0
        assert "attachments" not in unit.metadata


class TestEnexAdapter:
    def test_ingest_enex_notes_with_clean_content_and_metadata(self, tmp_path):
        enex = tmp_path / "evernote.enex"
        enex.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<en-export>
  <note>
    <title>Project Note</title>
    <content><![CDATA[<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">
<en-note><div>Hello <b>Evernote</b></div><div>Second line<br/>after break</div><en-media hash="abc" type="image/png"/></en-note>]]></content>
    <created>20240102T030405Z</created>
    <updated>20240203T040506Z</updated>
    <tag>research</tag>
    <tag>research</tag>
    <tag>archive</tag>
    <guid>note-guid-1</guid>
    <note-attributes>
      <author>Ada Lovelace</author>
      <source-url>https://example.com/source</source-url>
      <latitude>35.681236</latitude>
      <longitude>139.767125</longitude>
      <altitude>12.5</altitude>
    </note-attributes>
  </note>
</en-export>
""",
            encoding="utf-8",
        )

        result = EnexAdapter(path=str(enex)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.source_project == "enex"
        assert unit.source_id == "note-guid-1"
        assert unit.source_entity_type == "note"
        assert unit.title == "Project Note"
        assert unit.content == "Hello Evernote\nSecond line\nafter break"
        assert unit.tags == ["research", "archive"]
        assert unit.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        assert unit.updated_at == datetime(2024, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
        assert unit.metadata == {
            "source_path": "evernote.enex",
            "guid": "note-guid-1",
            "author": "Ada Lovelace",
            "source_url": "https://example.com/source",
            "latitude": "35.681236",
            "longitude": "139.767125",
            "altitude": "12.5",
        }
        assert result.edges == []

    def test_ingest_directory_recursively_and_filters_notes(self, tmp_path):
        root = tmp_path / "exports"
        nested = root / "nested"
        nested.mkdir(parents=True)
        (root / "old.enex").write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<en-export>
  <note>
    <title>Old</title>
    <content><![CDATA[<en-note><div>Old note</div></en-note>]]></content>
    <created>20230101T000000Z</created>
    <updated>20240101T000000Z</updated>
    <guid>old-guid</guid>
  </note>
</en-export>
""",
            encoding="utf-8",
        )
        (nested / "new.ENEX").write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<en-export>
  <note>
    <title>No Updated</title>
    <content><![CDATA[<en-note><p>Created timestamp is used.</p></en-note>]]></content>
    <created>20260425T120000Z</created>
  </note>
</en-export>
""",
            encoding="utf-8",
        )
        (nested / "ignore.txt").write_text("<en-export />\n", encoding="utf-8")

        result = EnexAdapter(path=str(root)).ingest(
            since=SyncState(
                source_project="enex",
                source_entity_type="note",
                last_sync_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
        )
        filtered = EnexAdapter(path=str(root)).ingest(entity_types=["text_document"])
        missing = EnexAdapter(path=str(tmp_path / "missing")).ingest()

        assert [unit.source_id for unit in result.units] == ["nested/new.ENEX:1"]
        assert result.units[0].metadata["source_path"] == "nested/new.ENEX"
        assert result.units[0].updated_at == datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
        assert filtered.units == []
        assert missing.units == []


class TestIpynbAdapter:
    def test_ingest_notebooks_with_content_metadata_and_tags(self, tmp_path):
        root = tmp_path / "notebooks"
        nested = root / "research"
        nested.mkdir(parents=True)
        notebook = nested / "analysis.ipynb"
        notebook.write_text(
            json.dumps(
                {
                    "metadata": {
                        "title": "Research Analysis",
                        "tags": ["research", "#notebook", "research"],
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {"name": "python", "version": "3.12.0"},
                    },
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "source": ["# Finding\n", "Searchable markdown insight."],
                        },
                        {
                            "cell_type": "code",
                            "source": ["import pandas as pd\n", "df.describe()\n"],
                        },
                        {"cell_type": "raw", "source": "ignored raw"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (root / "skip.txt").write_text("Not a notebook.\n", encoding="utf-8")

        result = IpynbAdapter(root_path=str(root)).ingest()

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.source_project == "me"
        assert unit.source_id == "research/analysis.ipynb"
        assert unit.source_entity_type == "notebook"
        assert unit.title == "Research Analysis"
        assert unit.content_type == "artifact"
        assert "# Finding" in unit.content
        assert "Searchable markdown insight." in unit.content
        assert "Code cell 2 (python, 2 lines):" in unit.content
        assert "import pandas as pd" in unit.content
        assert unit.tags == ["research", "notebook"]
        assert unit.metadata["path"] == "research/analysis.ipynb"
        assert unit.metadata["cell_count"] == 3
        assert unit.metadata["markdown_cell_count"] == 1
        assert unit.metadata["code_cell_count"] == 1
        assert unit.metadata["raw_cell_count"] == 1
        assert unit.metadata["kernelspec"]["name"] == "python3"
        assert unit.metadata["language_info"]["version"] == "3.12.0"
        assert unit.metadata["language"] == "python"
        assert unit.metadata["notebook_tags"] == ["research", "notebook"]
        assert unit.created_at.tzinfo is not None
        assert unit.updated_at.tzinfo is not None
        assert result.edges == []

    def test_sync_filter_and_invalid_notebooks_are_skipped(self, tmp_path):
        old_path = tmp_path / "old.ipynb"
        new_path = tmp_path / "new.ipynb"
        invalid_path = tmp_path / "invalid.ipynb"
        old_path.write_text(json.dumps({"cells": []}), encoding="utf-8")
        new_path.write_text(
            json.dumps(
                {
                    "metadata": {},
                    "cells": [{"cell_type": "markdown", "source": "New notebook body"}],
                }
            ),
            encoding="utf-8",
        )
        invalid_path.write_text("{not valid json", encoding="utf-8")
        os.utime(old_path, (1_700_000_000, 1_700_000_000))
        os.utime(new_path, (1_700_100_000, 1_700_100_000))
        os.utime(invalid_path, (1_700_100_000, 1_700_100_000))

        result = IpynbAdapter(root_path=str(tmp_path)).ingest(
            since=SyncState(
                source_project="ipynb",
                source_entity_type="notebook",
                last_sync_at=datetime.fromtimestamp(1_700_050_000, tz=timezone.utc),
            )
        )
        filtered = IpynbAdapter(root_path=str(tmp_path)).ingest(entity_types=["text_document"])
        missing = IpynbAdapter(root_path=str(tmp_path / "missing")).ingest()

        assert [unit.source_id for unit in result.units] == ["new.ipynb"]
        assert result.units[0].title == "new"
        assert filtered.units == []
        assert missing.units == []


class TestBibtexAdapter:
    def test_ingest_bibtex_file_with_multiple_entries_and_metadata(self, tmp_path):
        bib = tmp_path / "refs.bib"
        bib.write_text(
            r"""@article{smith2024graph,
  title = {Semantic Personal Graphs},
  author = {Smith, Ada and Doe, Grace},
  year = {2024},
  journal = {Journal of Knowledge Systems},
  abstract = {A study of personal semantic graphs.},
  note = {Includes longitudinal evaluation.},
  keywords = {graphs, knowledge, graphs},
  doi = {10.1000/example},
  url = {https://example.com/paper}
}

@inproceedings{lee2025agents,
  booktitle = {Proceedings of AgentConf},
  author = "Lee, Robin",
  year = 2025,
  keywords = {agents; evaluation}
}
""",
            encoding="utf-8",
        )

        result = BibtexAdapter(path=str(bib)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "refs.bib:smith2024graph",
            "refs.bib:lee2025agents",
        ]
        first = result.units[0]
        assert first.source_project == "bibtex"
        assert first.source_entity_type == "bibtex_entry"
        assert first.title == "Semantic Personal Graphs"
        assert first.tags == ["graphs", "knowledge"]
        assert "Title: Semantic Personal Graphs" in first.content
        assert "Authors: Smith, Ada; Doe, Grace" in first.content
        assert "Year: 2024" in first.content
        assert "Venue: Journal of Knowledge Systems" in first.content
        assert "Abstract: A study of personal semantic graphs." in first.content
        assert "Notes: Includes longitudinal evaluation." in first.content
        assert "DOI: 10.1000/example" in first.content
        assert "URL: https://example.com/paper" in first.content
        assert "Keywords: graphs, knowledge" in first.content
        assert first.metadata == {
            "citation_key": "smith2024graph",
            "entry_type": "article",
            "title": "Semantic Personal Graphs",
            "authors": ["Smith, Ada", "Doe, Grace"],
            "year": "2024",
            "doi": "10.1000/example",
            "url": "https://example.com/paper",
            "journal": "Journal of Knowledge Systems",
            "booktitle": "",
            "abstract": "A study of personal semantic graphs.",
            "keywords": ["graphs", "knowledge"],
            "source_file": "refs.bib",
        }
        assert result.units[1].title == "Proceedings of AgentConf"
        assert result.units[1].tags == ["agents", "evaluation"]
        assert result.edges == []

    def test_ingest_directory_recursively_includes_only_bib_files(self, tmp_path):
        root = tmp_path / "library"
        nested = root / "nested"
        nested.mkdir(parents=True)
        (root / "root.bib").write_text(
            "@book{book1, title = {Root Book}, year = {2023}}\n",
            encoding="utf-8",
        )
        (nested / "more.BIB").write_text(
            "@misc{misc1, title = {Nested Ref}, keywords = {nested}}\n",
            encoding="utf-8",
        )
        (nested / "ignore.txt").write_text(
            "@article{ignored, title = {Ignored}}\n",
            encoding="utf-8",
        )

        result = BibtexAdapter(path=str(root)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "nested/more.BIB:misc1",
            "root.bib:book1",
        ]
        assert result.units[0].metadata["source_file"] == "nested/more.BIB"

    def test_malformed_entries_are_skipped_with_warning(self, tmp_path):
        bib = tmp_path / "mixed.bib"
        bib.write_text(
            """@article{good, title = {Valid}, year = {2024}}
@article{bad, title = {Broken}
@inproceedings{also_good, title = {Also Valid}, booktitle = {Conference}}
""",
            encoding="utf-8",
        )

        with pytest.warns(UserWarning, match="Skipped 1 malformed BibTeX entr"):
            result = BibtexAdapter(path=str(bib)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "mixed.bib:good",
            "mixed.bib:also_good",
        ]

    def test_missing_sync_and_entity_filter_return_empty_result(self, tmp_path):
        old_path = tmp_path / "old.bib"
        new_path = tmp_path / "new.bib"
        old_path.write_text("@article{old, title = {Old}}\n", encoding="utf-8")
        new_path.write_text("@article{new, title = {New}}\n", encoding="utf-8")
        os.utime(old_path, (1_700_000_000, 1_700_000_000))
        os.utime(new_path, (1_700_100_000, 1_700_100_000))

        result = BibtexAdapter(path=str(tmp_path)).ingest(
            since=SyncState(
                source_project="bibtex",
                source_entity_type="bibtex_entry",
                last_sync_at=datetime.fromtimestamp(1_700_050_000, tz=timezone.utc),
            )
        )
        filtered = BibtexAdapter(path=str(new_path)).ingest(entity_types=["text_document"])
        missing = BibtexAdapter(path=str(tmp_path / "missing")).ingest()

        assert [unit.source_id for unit in result.units] == ["new.bib:new"]
        assert filtered.units == []
        assert missing.units == []


class TestCslJsonAdapter:
    def test_ingest_csl_json_array_with_metadata(self, tmp_path):
        refs = tmp_path / "refs.json"
        refs.write_text(
            json.dumps(
                [
                    {
                        "id": "smith-2024-graph",
                        "type": "article-journal",
                        "title": "Semantic Personal Graphs",
                        "author": [
                            {"family": "Smith", "given": "Ada"},
                            {"literal": "Knowledge Lab"},
                        ],
                        "issued": {"date-parts": [[2024, 5, 3]]},
                        "container-title": "Journal of Knowledge Systems",
                        "publisher": "Example Press",
                        "abstract": "A study of personal semantic graphs.",
                        "keyword": "graphs, knowledge, graphs",
                        "DOI": "https://doi.org/10.1000/example",
                        "URL": "https://example.com/paper",
                    },
                    {
                        "type": "paper-conference",
                        "title": "Agent Evaluation",
                        "author": [{"family": "Lee", "given": "Robin"}],
                        "issued": {"literal": "Spring 2025"},
                        "categories": ["agents", "evaluation"],
                    },
                ]
            ),
            encoding="utf-8",
        )

        result = CslJsonAdapter(path=str(refs)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "smith-2024-graph",
            "refs.json:1:2f61929a518a9ddf1a113131",
        ]
        first = result.units[0]
        assert first.source_project == "csl_json"
        assert first.source_entity_type == "csl_json_item"
        assert first.title == "Semantic Personal Graphs"
        assert first.tags == ["graphs", "knowledge", "article-journal"]
        assert "Authors: Smith, Ada; Knowledge Lab" in first.content
        assert "Issued: 2024-05-03" in first.content
        assert "Venue: Journal of Knowledge Systems" in first.content
        assert "Abstract: A study of personal semantic graphs." in first.content
        assert "DOI: 10.1000/example" in first.content
        assert first.metadata == {
            "csl_type": "article-journal",
            "doi": "10.1000/example",
            "url": "https://example.com/paper",
            "issued": "2024-05-03",
            "authors": ["Smith, Ada", "Knowledge Lab"],
            "publisher": "Example Press",
            "container_title": "Journal of Knowledge Systems",
            "source_file": "refs.json",
        }
        assert first.created_at == datetime(2024, 5, 3, tzinfo=timezone.utc)
        assert result.units[1].metadata["issued"] == "Spring 2025"
        assert result.edges == []

    def test_ingest_single_csl_json_object(self, tmp_path):
        ref = tmp_path / "single.json"
        ref.write_text(
            json.dumps(
                {
                    "id": "doe-2026",
                    "type": "book",
                    "title": "Single Bibliography Item",
                    "publisher": "Example Press",
                    "issued": {"date-parts": [[2026]]},
                }
            ),
            encoding="utf-8",
        )

        result = CslJsonAdapter(path=str(ref)).ingest()

        assert [unit.title for unit in result.units] == ["Single Bibliography Item"]
        assert result.units[0].source_id == "doe-2026"
        assert result.units[0].metadata["publisher"] == "Example Press"

    def test_ingest_directory_recursively_includes_only_json_files_deterministically(self, tmp_path):
        root = tmp_path / "library"
        nested = root / "nested"
        nested.mkdir(parents=True)
        (root / "root.json").write_text(
            json.dumps({"id": "root", "type": "book", "title": "Root Book"}),
            encoding="utf-8",
        )
        (nested / "more.JSON").write_text(
            json.dumps({"id": "nested", "type": "article", "title": "Nested Ref"}),
            encoding="utf-8",
        )
        (nested / "ignore.txt").write_text(
            json.dumps({"id": "ignored", "title": "Ignored"}),
            encoding="utf-8",
        )

        result = CslJsonAdapter(path=str(root)).ingest()

        assert [unit.metadata["source_file"] for unit in result.units] == [
            "nested/more.JSON",
            "root.json",
        ]
        assert [unit.source_id for unit in result.units] == ["nested", "root"]

    def test_malformed_json_missing_path_and_entity_filter_are_skipped(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        good = tmp_path / "good.json"
        good.write_text(
            json.dumps({"id": "good", "type": "article", "title": "Good"}),
            encoding="utf-8",
        )

        with pytest.warns(UserWarning, match="Skipped 1 malformed CSL-JSON input"):
            result = CslJsonAdapter(path=str(tmp_path)).ingest()
        filtered = CslJsonAdapter(path=str(good)).ingest(entity_types=["ris_record"])
        missing = CslJsonAdapter(path=str(tmp_path / "missing")).ingest()

        assert [unit.source_id for unit in result.units] == ["good"]
        assert filtered.units == []
        assert missing.units == []


class TestRisAdapter:
    def test_ingest_ris_file_with_multiple_records_and_metadata(self, tmp_path):
        ris = tmp_path / "refs.ris"
        ris.write_text(
            """TY  - JOUR
TI  - Semantic Personal Graphs
AU  - Smith, Ada
AU  - Doe, Grace
PY  - 2024
T2  - Journal of Knowledge Systems
AB  - A study of personal semantic graphs.
KW  - graphs
KW  - knowledge; graphs
DO  - 10.1000/example
UR  - https://example.com/paper
ER  -

TY  - CONF
T1  - Agent Evaluation
A1  - Lee, Robin
Y1  - 2025/05/03/
N2  - Evaluation methods for agents.
KW  - agents, evaluation
L2  - https://example.com/agent-eval
ER  -
""",
            encoding="utf-8",
        )

        result = RisAdapter(path=str(ris)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "doi:10.1000/example",
            "url:https://example.com/agent-eval",
        ]
        first = result.units[0]
        assert first.source_project == "ris"
        assert first.source_entity_type == "ris_record"
        assert first.title == "Semantic Personal Graphs"
        assert first.tags == ["graphs", "knowledge"]
        assert "Authors: Smith, Ada; Doe, Grace" in first.content
        assert "Year: 2024" in first.content
        assert "Venue: Journal of Knowledge Systems" in first.content
        assert "Abstract: A study of personal semantic graphs." in first.content
        assert first.metadata == {
            "ris_type": "JOUR",
            "authors": ["Smith, Ada", "Doe, Grace"],
            "year": "2024",
            "date": "2024",
            "doi": "10.1000/example",
            "url": "https://example.com/paper",
            "venue": "Journal of Knowledge Systems",
            "source_file": "refs.ris",
        }
        assert result.units[1].created_at == datetime(2025, 5, 3, tzinfo=timezone.utc)
        assert result.edges == []

    def test_ingest_directory_recursively_includes_only_ris_files(self, tmp_path):
        root = tmp_path / "library"
        nested = root / "nested"
        nested.mkdir(parents=True)
        (root / "root.ris").write_text(
            "TY  - BOOK\nTI  - Root Book\nPY  - 2023\nER  -\n",
            encoding="utf-8",
        )
        (nested / "more.RIS").write_text(
            "TY  - JOUR\nTI  - Nested Ref\nKW  - nested\nER  -\n",
            encoding="utf-8",
        )
        (nested / "ignore.txt").write_text(
            "TY  - JOUR\nTI  - Ignored\nER  -\n",
            encoding="utf-8",
        )

        result = RisAdapter(path=str(root)).ingest()

        assert [unit.metadata["source_file"] for unit in result.units] == [
            "nested/more.RIS",
            "root.ris",
        ]
        assert result.units[0].tags == ["nested"]

    def test_malformed_and_incomplete_records_are_skipped_with_warning(self, tmp_path):
        ris = tmp_path / "mixed.ris"
        ris.write_text(
            """TY  - JOUR
TI  - Valid
PY  - 2024
ER  -
TY  - JOUR
PY  - 2025
ER  -
TI  - Stray title
TY  - JOUR
TI  - Unterminated
""",
            encoding="utf-8",
        )

        with pytest.warns(UserWarning, match="Skipped 3 malformed RIS record"):
            result = RisAdapter(path=str(ris)).ingest()

        assert [unit.title for unit in result.units] == ["Valid"]

    def test_incremental_sync_uses_parseable_record_dates(self, tmp_path):
        ris = tmp_path / "sync.ris"
        ris.write_text(
            """TY  - JOUR
TI  - Old
PY  - 2024
ER  -
TY  - JOUR
TI  - New
DA  - 2026-04-25
ER  -
TY  - JOUR
TI  - Undated
ER  -
""",
            encoding="utf-8",
        )

        result = RisAdapter(path=str(ris)).ingest(
            since=SyncState(
                source_project="ris",
                source_entity_type="ris_record",
                last_sync_at=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
            )
        )
        filtered = RisAdapter(path=str(ris)).ingest(entity_types=["bibtex_entry"])
        missing = RisAdapter(path=str(tmp_path / "missing")).ingest()

        assert [unit.title for unit in result.units] == ["New", "Undated"]
        assert filtered.units == []
        assert missing.units == []
        assert result.units[0].updated_at == datetime(2026, 4, 25, tzinfo=timezone.utc)

    def test_hash_source_id_is_stable_when_doi_and_url_are_missing(self, tmp_path):
        ris = tmp_path / "hash.ris"
        ris.write_text(
            "TY  - JOUR\nTI  - Stable Hash\nAU  - Doe, Jane\nPY  - 2024\nER  -\n",
            encoding="utf-8",
        )

        first = RisAdapter(path=str(ris)).ingest()
        second = RisAdapter(path=str(ris)).ingest()

        assert first.units[0].source_id == second.units[0].source_id
        assert first.units[0].source_id.startswith("ris:")


class TestPdfAdapter:
    def test_ingest_pdf_documents_with_mocked_reader(self, tmp_path, monkeypatch):
        root = tmp_path / "pdfs"
        nested = root / "nested"
        nested.mkdir(parents=True)
        first = root / "paper.pdf"
        second = nested / "appendix.PDF"
        first.write_bytes(b"%PDF-1.4 first")
        second.write_bytes(b"%PDF-1.4 second")
        (root / "skip.txt").write_text("Not a PDF.\n", encoding="utf-8")

        class FakePage:
            def __init__(self, text=None, error: Exception | None = None):
                self.text = text
                self.error = error

            def extract_text(self):
                if self.error:
                    raise self.error
                return self.text

        class FakeReader:
            def __init__(self, path):
                name = Path(path).name
                if name == "paper.pdf":
                    self.pages = [FakePage("First page text."), FakePage(None)]
                else:
                    self.pages = [FakePage("Appendix text."), FakePage(error=ValueError("bad page"))]

        monkeypatch.setattr(PdfAdapter, "_load_pdf_reader", lambda self: FakeReader)

        result = PdfAdapter(path=str(root)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "nested/appendix.PDF",
            "paper.pdf",
        ]
        by_source = {unit.source_id: unit for unit in result.units}
        paper = by_source["paper.pdf"]
        assert paper.source_project == "pdf"
        assert paper.source_entity_type == "pdf_document"
        assert paper.title == "paper"
        assert paper.content == "First page text."
        assert paper.metadata == {
            "source_file": str(first),
            "page_count": 2,
            "extraction_warnings": [],
            "file_size": first.stat().st_size,
        }
        appendix = by_source["nested/appendix.PDF"]
        assert appendix.content == "Appendix text."
        assert appendix.metadata["page_count"] == 2
        assert appendix.metadata["extraction_warnings"] == ["page_2: bad page"]
        assert result.edges == []

    def test_ingest_single_pdf_file_sync_and_entity_filter(self, tmp_path, monkeypatch):
        old_path = tmp_path / "old.pdf"
        new_path = tmp_path / "new.pdf"
        old_path.write_bytes(b"%PDF old")
        new_path.write_bytes(b"%PDF new")
        os.utime(old_path, (1_700_000_000, 1_700_000_000))
        os.utime(new_path, (1_700_100_000, 1_700_100_000))

        class FakeReader:
            def __init__(self, path):
                self.pages = []

        monkeypatch.setattr(PdfAdapter, "_load_pdf_reader", lambda self: FakeReader)

        result = PdfAdapter(path=str(new_path)).ingest(
            since=SyncState(
                source_project="pdf",
                source_entity_type="pdf_document",
                last_sync_at=datetime.fromtimestamp(1_700_050_000, tz=timezone.utc),
            )
        )
        filtered = PdfAdapter(path=str(new_path)).ingest(entity_types=["text_document"])
        missing = PdfAdapter(path=str(tmp_path / "missing")).ingest()

        assert [unit.source_id for unit in result.units] == ["new.pdf"]
        assert result.units[0].source_project == "pdf"
        assert filtered.units == []
        assert missing.units == []

    def test_missing_pypdf_raises_actionable_import_error_when_used(self, tmp_path, monkeypatch):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError("No module named pypdf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="Install it with `uv sync --extra pdf`"):
            PdfAdapter(path=str(pdf)).ingest()


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
        assert unit.source_project == "calendar"
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
            "component": "VEVENT",
            "start": "2026-04-24T09:00:00+00:00",
            "end": "2026-04-24T10:00:00+00:00",
            "due": "",
            "completed": "",
            "location": "Conference Room",
            "organizer": "Alice <alice@example.com>",
            "attendees": ["Bob <bob@example.com>"],
            "categories": ["work", "planning"],
            "rrule": "",
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
        assert unit.source_project == "feed"
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
            "source_id,title,content,content_type,tags,utility_score,confidence,created_at,updated_at,metadata,source,priority\n"
            'note-1,Solar note,"Storage doubled.",finding,"energy, solar",8.5,0.75,2025-04-24T12:00:00Z,2025-04-25T12:00:00Z,"{""url"": ""https://example.com"", ""rank"": 3}",spreadsheet,high\n',
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
        assert unit.updated_at.isoformat() == "2025-04-25T12:00:00+00:00"
        assert unit.metadata == {
            "url": "https://example.com",
            "rank": 3,
            "fields": {"source": "spreadsheet", "priority": "high"},
        }

    def test_ingest_csv_rows_from_directory(self, tmp_path):
        root = tmp_path / "exports"
        nested = root / "nested"
        nested.mkdir(parents=True)
        (root / "notes.csv").write_text(
            "source_id,title,content,tags,metadata_json\n"
            'root-1,Root row,Root content,"alpha, beta","{""tool"": ""sheet""}"\n',
            encoding="utf-8",
        )
        (nested / "more.csv").write_text(
            "source_id,title,content,owner\nnested-1,Nested row,Nested content,Taka\n",
            encoding="utf-8",
        )
        (nested / "ignore.txt").write_text(
            "source_id,title,content\nignored,Ignored,Ignored\n",
            encoding="utf-8",
        )

        result = CsvAdapter(path=str(root)).ingest()

        assert [unit.source_id for unit in result.units] == ["nested-1", "root-1"]
        assert result.units[0].metadata == {"fields": {"owner": "Taka"}}
        assert result.units[1].tags == ["alpha", "beta"]
        assert result.units[1].metadata == {"tool": "sheet"}

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

    def test_malformed_values_fall_back_without_crashing(self, tmp_path):
        csv_path = tmp_path / "bad-metadata.csv"
        csv_path.write_text(
            "title,content,content_type,created_at,updated_at,metadata,source\n"
            "Bad metadata,Still imported,unknown-date,bad-date,also-bad,{not json,sheet\n",
            encoding="utf-8",
        )

        result = CsvAdapter(path=str(csv_path)).ingest()

        assert len(result.units) == 1
        assert result.units[0].content_type == "insight"
        assert result.units[0].metadata == {
            "metadata": "{not json",
            "fields": {"source": "sheet"},
        }

    def test_extra_fields_do_not_overwrite_explicit_metadata_fields(self, tmp_path):
        csv_path = tmp_path / "explicit-fields.csv"
        csv_path.write_text(
            "title,content,metadata,owner,priority\n"
            'Explicit fields,Imported,"{""fields"": {""owner"": ""metadata""}}",column,high\n',
            encoding="utf-8",
        )

        result = CsvAdapter(path=str(csv_path)).ingest()

        assert result.units[0].metadata == {
            "fields": {"owner": "metadata", "priority": "high"}
        }

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


class TestYamlAdapter:
    def test_ingest_yaml_documents_recursively_with_metadata_title_and_tags(self, tmp_path):
        root = tmp_path / "yaml"
        nested = root / "nested"
        nested.mkdir(parents=True)
        first = root / "note.yaml"
        second = nested / "record.yml"
        first.write_text(
            yaml.safe_dump(
                {
                    "title": "YAML Note",
                    "tags": ["knowledge", "#yaml", "knowledge"],
                    "summary": "Structured note content.",
                    "details": {"status": "active"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        second.write_text(
            yaml.safe_dump(
                {
                    "name": "Named Export",
                    "tags": ["export"],
                    "items": ["alpha", "beta"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (nested / "skip.txt").write_text("Not YAML.\n", encoding="utf-8")

        result = YamlAdapter(root_path=str(root)).ingest()

        assert [unit.source_id for unit in result.units] == [
            "nested/record.yml",
            "note.yaml",
        ]
        by_source = {unit.source_id: unit for unit in result.units}
        note = by_source["note.yaml"]
        assert note.source_project == "yaml"
        assert note.source_entity_type == "yaml_document"
        assert note.title == "YAML Note"
        assert note.tags == ["knowledge", "yaml"]
        assert "summary: Structured note content." in note.content
        assert "details:" in note.content
        assert "title:" not in note.content
        assert note.metadata == {
            "path": "note.yaml",
            "file_size": first.stat().st_size,
            "top_level_keys": ["title", "tags", "summary", "details"],
        }
        assert note.created_at.tzinfo is not None
        assert by_source["nested/record.yml"].title == "Named Export"
        assert by_source["nested/record.yml"].tags == ["export"]
        assert result.edges == []

    def test_malformed_yaml_file_is_skipped_without_aborting_tree(self, tmp_path):
        good = tmp_path / "good.yaml"
        bad = tmp_path / "bad.yml"
        good.write_text("title: Valid\nbody: Imported.\n", encoding="utf-8")
        bad.write_text("title: [broken\n", encoding="utf-8")

        with pytest.warns(UserWarning, match="Skipped 1 malformed YAML file"):
            result = YamlAdapter(root_path=str(tmp_path)).ingest()

        assert [unit.source_id for unit in result.units] == ["good.yaml"]
        assert result.units[0].title == "Valid"

    def test_empty_missing_non_directory_and_entity_filter_return_empty_result(self, tmp_path):
        file_root = tmp_path / "file.yaml"
        file_root.write_text("title: Root file\n", encoding="utf-8")

        empty = YamlAdapter(root_path=str(tmp_path / "empty")).ingest()
        missing = YamlAdapter(root_path=str(tmp_path / "missing")).ingest()
        non_directory = YamlAdapter(root_path=str(file_root)).ingest()
        filtered = YamlAdapter(root_path=str(tmp_path)).ingest(entity_types=["jsonl_record"])

        assert empty.units == []
        assert missing.units == []
        assert non_directory.units == []
        assert filtered.units == []
        assert filtered.edges == []

    def test_title_falls_back_to_file_stem_and_sync_skips_old_files(self, tmp_path):
        old_path = tmp_path / "old.yaml"
        new_path = tmp_path / "untitled.yml"
        old_path.write_text("title: Old\nbody: skipped\n", encoding="utf-8")
        new_path.write_text("body: imported\n", encoding="utf-8")
        os.utime(old_path, (1_700_000_000, 1_700_000_000))
        os.utime(new_path, (1_700_100_000, 1_700_100_000))

        result = YamlAdapter(root_path=str(tmp_path)).ingest(
            since=SyncState(
                source_project="yaml",
                source_entity_type="yaml_document",
                last_sync_at=datetime.fromtimestamp(1_700_050_000, tz=timezone.utc),
            )
        )

        assert [unit.source_id for unit in result.units] == ["untitled.yml"]
        assert result.units[0].title == "untitled"


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


class TestGitAdapter:
    def _run_git(
        self,
        repo: Path,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, **(env or {})},
            )
        except FileNotFoundError:
            pytest.skip("git executable is not available")

    def _init_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "knowledge-repo"
        repo.mkdir(parents=True)
        try:
            subprocess.run(
                ["git", "init", str(repo)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            pytest.skip("git executable is not available")
        self._run_git(repo, "config", "user.name", "Test User")
        self._run_git(repo, "config", "user.email", "test@example.com")
        return repo

    def _commit(
        self,
        repo: Path,
        filename: str,
        content: str,
        subject: str,
        body: str,
        timestamp: str,
    ) -> str:
        path = repo / filename
        path.write_text(content, encoding="utf-8")
        self._run_git(repo, "add", filename)
        env = {
            "GIT_AUTHOR_NAME": "Ada Lovelace",
            "GIT_AUTHOR_EMAIL": "ada@example.com",
            "GIT_AUTHOR_DATE": timestamp,
            "GIT_COMMITTER_NAME": "Grace Hopper",
            "GIT_COMMITTER_EMAIL": "grace@example.com",
            "GIT_COMMITTER_DATE": timestamp,
        }
        self._run_git(repo, "commit", "-m", subject, "-m", body, env=env)
        return self._run_git(repo, "rev-parse", "HEAD").stdout.strip()

    def test_ingests_commits_with_metadata_and_refs(self, tmp_path):
        repo = self._init_repo(tmp_path)
        first_sha = self._commit(
            repo,
            "notes.txt",
            "first\n",
            "Capture design decision",
            "Use the graph as commit memory.",
            "2025-01-01T10:00:00+00:00",
        )
        self._run_git(repo, "tag", "v1", first_sha)

        result = GitAdapter(repos=str(repo)).ingest()

        by_title = {unit.title: unit for unit in result.units}
        unit = by_title["Capture design decision"]
        assert unit.source_project == "git"
        assert unit.source_entity_type == "commit"
        assert unit.source_id == f"knowledge-repo:{first_sha}"
        assert unit.content == "Capture design decision\n\nUse the graph as commit memory."
        assert unit.content_type == "artifact"
        assert unit.metadata["sha"] == first_sha
        assert unit.metadata["author"] == "Ada Lovelace"
        assert unit.metadata["email"] == "ada@example.com"
        assert unit.metadata["repo_name"] == "knowledge-repo"
        assert unit.metadata["repo_path"] == str(repo.resolve())
        assert "tag: v1" in unit.metadata["refs"]
        assert unit.created_at == datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
        assert unit.updated_at == datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
        assert unit.tags == ["git", "knowledge-repo"]
        assert result.edges == []

    def test_incremental_sync_excludes_commits_at_or_before_last_sync(self, tmp_path):
        repo = self._init_repo(tmp_path)
        old_sha = self._commit(
            repo,
            "old.txt",
            "old\n",
            "Old commit",
            "Already synced.",
            "2025-01-01T00:00:00+00:00",
        )
        new_sha = self._commit(
            repo,
            "new.txt",
            "new\n",
            "New commit",
            "Needs ingest.",
            "2025-01-02T00:00:00+00:00",
        )

        result = GitAdapter(repos=str(repo)).ingest(
            since=SyncState(
                source_project="git",
                source_entity_type="commit",
                last_sync_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
        )

        assert [unit.source_id for unit in result.units] == [f"knowledge-repo:{new_sha}"]
        assert old_sha not in result.units[0].source_id

    def test_accepts_comma_and_newline_separated_repositories(self, tmp_path):
        first = self._init_repo(tmp_path / "first")
        second = self._init_repo(tmp_path / "second")
        first_sha = self._commit(
            first,
            "a.txt",
            "a\n",
            "First repo commit",
            "Imported from first.",
            "2025-01-01T00:00:00+00:00",
        )
        second_sha = self._commit(
            second,
            "b.txt",
            "b\n",
            "Second repo commit",
            "Imported from second.",
            "2025-01-02T00:00:00+00:00",
        )

        result = GitAdapter(repos=f"{first},\n{second}").ingest()

        assert {unit.source_id for unit in result.units} == {
            f"knowledge-repo:{first_sha}",
            f"knowledge-repo:{second_sha}",
        }

    def test_entity_filter_returns_empty_result(self, tmp_path):
        repo = self._init_repo(tmp_path)
        self._commit(
            repo,
            "notes.txt",
            "first\n",
            "Capture design decision",
            "Use the graph as commit memory.",
            "2025-01-01T10:00:00+00:00",
        )

        result = GitAdapter(repos=str(repo)).ingest(entity_types=["yaml_document"])

        assert result.units == []
        assert result.edges == []


class TestRegistry:
    def test_list_adapters(self):
        adapters = list_adapters()
        expected = set(_ADAPTERS)
        assert set(adapters) == expected
        # Verify sorted order
        assert adapters == sorted(expected)

    def test_get_adapter(self):
        adapter = get_adapter("me", config_path="/tmp/test.yaml")
        assert adapter.name == "me"

        atom_adapter = get_adapter("atom", path="/tmp/feed.xml")
        assert isinstance(atom_adapter, AtomAdapter)
        assert atom_adapter.name == "atom"

        jsonl_adapter = get_adapter("jsonl", path="/tmp/test.jsonl")
        assert jsonl_adapter.name == "jsonl"

        jsonl_notes_adapter = get_adapter("jsonl_notes", path="/tmp/notes.jsonl")
        assert isinstance(jsonl_notes_adapter, JsonlNotesAdapter)
        assert jsonl_notes_adapter.name == "jsonl_notes"

        daily_journal_adapter = get_adapter("daily_journal", path="/tmp/journal")
        assert isinstance(daily_journal_adapter, DailyJournalAdapter)
        assert daily_journal_adapter.name == "daily_journal"

        csv_rows_adapter = get_adapter("csv_rows", path="/tmp/rows.csv")
        assert isinstance(csv_rows_adapter, CsvRowsAdapter)
        assert csv_rows_adapter.name == "csv_rows"

        yaml_adapter = get_adapter("yaml", root_path="/tmp/yaml")
        assert yaml_adapter.name == "yaml"

        opml_adapter = get_adapter("opml", path="/tmp/test.opml")
        assert opml_adapter.name == "opml"

        obsidian_canvas_adapter = get_adapter("obsidian_canvas", path="/tmp/map.canvas")
        assert isinstance(obsidian_canvas_adapter, ObsidianCanvasAdapter)
        assert obsidian_canvas_adapter.name == "obsidian_canvas"

        notion_markdown_adapter = get_adapter("notion_markdown", path="/tmp/notion")
        assert isinstance(notion_markdown_adapter, NotionMarkdownAdapter)
        assert notion_markdown_adapter.name == "notion_markdown"

        org_adapter = get_adapter("org", root_path="/tmp/org")
        assert org_adapter.name == "org"

        pdf_adapter = get_adapter("pdf", path="/tmp/test.pdf")
        assert pdf_adapter.name == "pdf"

        plain_text_adapter = get_adapter("plain_text", path="/tmp/notes.txt")
        assert isinstance(plain_text_adapter, PlainTextAdapter)
        assert plain_text_adapter.name == "plain_text"

        email_adapter = get_adapter("email", path="/tmp/mail")
        assert email_adapter.name == "email"

        enex_adapter = get_adapter("enex", path="/tmp/evernote.enex")
        assert enex_adapter.name == "enex"

        text_adapter = get_adapter("text", root_path="/tmp/text")
        assert text_adapter.name == "text"

        text_outline_adapter = get_adapter("text_outline", path="/tmp/outline.txt")
        assert isinstance(text_outline_adapter, TextOutlineAdapter)
        assert text_outline_adapter.name == "text_outline"

        tana_paste_adapter = get_adapter("tana_paste", path="/tmp/tana.txt")
        assert isinstance(tana_paste_adapter, TanaPasteAdapter)
        assert tana_paste_adapter.name == "tana_paste"

        html_adapter = get_adapter("html", root_path="/tmp/html")
        assert html_adapter.name == "html"

        ical_adapter = get_adapter("ical", path="/tmp/calendar.ics")
        assert ical_adapter.name == "ical"

        ipynb_adapter = get_adapter("ipynb", root_path="/tmp/notebooks")
        assert ipynb_adapter.name == "ipynb"

        feed_adapter = get_adapter("feed", sources="/tmp/feed.xml")
        assert feed_adapter.name == "feed"

        bibtex_adapter = get_adapter("bibtex", path="/tmp/refs.bib")
        assert bibtex_adapter.name == "bibtex"

        bibdesk_adapter = get_adapter("bibdesk", path="/tmp/refs.plist")
        assert isinstance(bibdesk_adapter, BibDeskAdapter)
        assert bibdesk_adapter.name == "bibdesk"

        bluesky_adapter = get_adapter("bluesky_archive", path="/tmp/bluesky")
        assert isinstance(bluesky_adapter, BlueskyArchiveAdapter)
        assert bluesky_adapter.name == "bluesky_archive"

        csl_json_adapter = get_adapter("csl_json", path="/tmp/refs.json")
        assert csl_json_adapter.name == "csl_json"

        crossref_adapter = get_adapter("crossref", path="/tmp/crossref")
        assert isinstance(crossref_adapter, CrossrefAdapter)
        assert crossref_adapter.name == "crossref"

        ris_adapter = get_adapter("ris", path="/tmp/refs.ris")
        assert ris_adapter.name == "ris"

        jats_adapter = get_adapter("jats", path="/tmp/article.xml")
        assert jats_adapter.name == "jats"

        git_adapter = get_adapter("git", repos="/tmp/repo")
        assert git_adapter.name == "git"

        google_keep_adapter = get_adapter("google_keep", path="/tmp/keep")
        assert isinstance(google_keep_adapter, GoogleKeepAdapter)
        assert google_keep_adapter.name == "google_keep"

        transcript_adapter = get_adapter("transcript", root_path="/tmp/transcripts")
        assert transcript_adapter.name == "transcript"

        twitter_archive_adapter = get_adapter("twitter_archive", path="/tmp/tweets.js")
        assert isinstance(twitter_archive_adapter, TwitterArchiveAdapter)
        assert twitter_archive_adapter.name == "twitter_archive"

        webvtt_adapter = get_adapter("webvtt", path="/tmp/transcript.vtt")
        assert isinstance(webvtt_adapter, WebVttAdapter)
        assert webvtt_adapter.name == "webvtt"

        pocket_adapter = get_adapter("pocket", path="/tmp/pocket.csv")
        assert pocket_adapter.name == "pocket"

        pocket_csv_adapter = get_adapter("pocket_csv", path="/tmp/pocket.csv")
        assert isinstance(pocket_csv_adapter, PocketCsvAdapter)
        assert pocket_csv_adapter.name == "pocket_csv"

        browser_history_csv_adapter = get_adapter(
            "browser_history_csv", path="/tmp/history.csv"
        )
        assert isinstance(browser_history_csv_adapter, BrowserHistoryCsvAdapter)
        assert browser_history_csv_adapter.name == "browser_history_csv"

        chrome_history_adapter = get_adapter("chrome_history", path="/tmp/History")
        assert isinstance(chrome_history_adapter, ChromeHistoryAdapter)
        assert chrome_history_adapter.name == "chrome_history"

        bookmarks_html_adapter = get_adapter("bookmarks_html", path="/tmp/bookmarks.html")
        assert isinstance(bookmarks_html_adapter, BookmarksHtmlAdapter)
        assert bookmarks_html_adapter.name == "bookmarks_html"

        chatgpt_json_adapter = get_adapter("chatgpt_json", path="/tmp/conversations.json")
        assert isinstance(chatgpt_json_adapter, ChatGptJsonAdapter)
        assert chatgpt_json_adapter.name == "chatgpt_json"

        discord_json_adapter = get_adapter("discord_json", path="/tmp/discord")
        assert isinstance(discord_json_adapter, DiscordJsonAdapter)
        assert discord_json_adapter.name == "discord_json"

        pinboard_adapter = get_adapter("pinboard", path="/tmp/pinboard.json")
        assert isinstance(pinboard_adapter, PinboardAdapter)
        assert pinboard_adapter.name == "pinboard"

        raindrop_adapter = get_adapter("raindrop", path="/tmp/raindrop.json")
        assert isinstance(raindrop_adapter, RaindropAdapter)
        assert raindrop_adapter.name == "raindrop"

        raindrop_csv_adapter = get_adapter("raindrop_csv", path="/tmp/raindrop.csv")
        assert isinstance(raindrop_csv_adapter, RaindropCsvAdapter)
        assert raindrop_csv_adapter.name == "raindrop_csv"

        raindrop_json_adapter = get_adapter("raindrop_json", path="/tmp/raindrop.json")
        assert isinstance(raindrop_json_adapter, RaindropJsonAdapter)
        assert raindrop_json_adapter.name == "raindrop_json"

        safari_bookmarks_adapter = get_adapter(
            "safari_bookmarks", path="/tmp/Bookmarks.plist"
        )
        assert isinstance(safari_bookmarks_adapter, SafariBookmarksAdapter)
        assert safari_bookmarks_adapter.name == "safari_bookmarks"

        zotero_rdf_adapter = get_adapter("zotero_rdf", path="/tmp/library.rdf")
        assert zotero_rdf_adapter.name == "zotero_rdf"

        hypothesis_adapter = get_adapter("hypothesis", path="/tmp/hypothesis.json")
        assert isinstance(hypothesis_adapter, HypothesisAdapter)
        assert hypothesis_adapter.name == "hypothesis"

        readwise_adapter = get_adapter("readwise", path="/tmp/readwise.json")
        assert isinstance(readwise_adapter, ReadwiseAdapter)
        assert readwise_adapter.name == "readwise"

        readwise_csv_adapter = get_adapter("readwise_csv", path="/tmp/readwise.csv")
        assert isinstance(readwise_csv_adapter, ReadwiseCsvAdapter)
        assert readwise_csv_adapter.name == "readwise_csv"

        reddit_saved_csv_adapter = get_adapter("reddit_saved_csv", path="/tmp/reddit")
        assert isinstance(reddit_saved_csv_adapter, RedditSavedCsvAdapter)
        assert reddit_saved_csv_adapter.name == "reddit_saved_csv"

        roam_adapter = get_adapter("roam", file_path="/tmp/roam.json")
        assert isinstance(roam_adapter, RoamAdapter)
        assert roam_adapter.name == "roam"

        logseq_adapter = get_adapter("logseq", file_path="/tmp/logseq.edn")
        assert isinstance(logseq_adapter, LogseqAdapter)
        assert logseq_adapter.name == "logseq"

        sqlite_query_log_adapter = get_adapter(
            "sqlite_query_log", db_path="/tmp/queries.db"
        )
        assert isinstance(sqlite_query_log_adapter, SqliteQueryLogAdapter)
        assert sqlite_query_log_adapter.name == "sqlite_query_log"

        slack_json_adapter = get_adapter("slack_json", path="/tmp/slack/general")
        assert isinstance(slack_json_adapter, SlackJsonAdapter)
        assert slack_json_adapter.name == "slack_json"

        mediawiki_adapter = get_adapter("mediawiki", path="/tmp/wiki.xml")
        assert isinstance(mediawiki_adapter, MediaWikiAdapter)
        assert mediawiki_adapter.name == "mediawiki"

        markdown_links_adapter = get_adapter("markdown_links", path="/tmp/notes")
        assert isinstance(markdown_links_adapter, MarkdownLinksAdapter)
        assert markdown_links_adapter.name == "markdown_links"

        markdown_notes_adapter = get_adapter("markdown_notes", path="/tmp/notes")
        assert isinstance(markdown_notes_adapter, MarkdownNotesAdapter)
        assert markdown_notes_adapter.name == "markdown_notes"

        markdown_callouts_adapter = get_adapter("markdown_callouts", path="/tmp/notes")
        assert isinstance(markdown_callouts_adapter, MarkdownCalloutsAdapter)
        assert markdown_callouts_adapter.name == "markdown_callouts"

        markdown_definitions_adapter = get_adapter(
            "markdown_definitions", path="/tmp/notes"
        )
        assert isinstance(markdown_definitions_adapter, MarkdownDefinitionsAdapter)
        assert markdown_definitions_adapter.name == "markdown_definitions"

        markdown_frontmatter_adapter = get_adapter(
            "markdown_frontmatter", path="/tmp/notes"
        )
        assert isinstance(markdown_frontmatter_adapter, MarkdownFrontmatterAdapter)
        assert markdown_frontmatter_adapter.name == "markdown_frontmatter"

        markdown_tasks_adapter = get_adapter("markdown_tasks", path="/tmp/notes")
        assert isinstance(markdown_tasks_adapter, MarkdownTasksAdapter)
        assert markdown_tasks_adapter.name == "markdown_tasks"

        mastodon_adapter = get_adapter("mastodon", path="/tmp/outbox.json")
        assert isinstance(mastodon_adapter, MastodonAdapter)
        assert mastodon_adapter.name == "mastodon"

        spotify_takeout_adapter = get_adapter("spotify-takeout", path="/tmp/spotify")
        assert isinstance(spotify_takeout_adapter, SpotifyTakeoutAdapter)
        assert spotify_takeout_adapter.name == "spotify_takeout"

    def test_unknown_adapter(self):
        with pytest.raises(KeyError) as exc_info:
            get_adapter("unknown")
        # Verify error message includes normalized name and sorted available adapters
        error_msg = str(exc_info.value)
        assert "unknown" in error_msg
        assert "Available:" in error_msg

    def test_adapter_name_normalization_whitespace(self):
        # Leading/trailing whitespace should be stripped
        adapter1 = get_adapter(" jsonl ", path="/tmp/test.jsonl")
        assert isinstance(adapter1, JsonlAdapter)
        assert adapter1.name == "jsonl"

        adapter2 = get_adapter("  csv  ", path="/tmp/test.csv")
        assert isinstance(adapter2, CsvAdapter)
        assert adapter2.name == "csv"

    def test_adapter_name_normalization_case(self):
        # Case-insensitive lookup
        adapter1 = get_adapter("JSONL", path="/tmp/test.jsonl")
        assert isinstance(adapter1, JsonlAdapter)
        assert adapter1.name == "jsonl"

        adapter2 = get_adapter("Markdown", root_path="/tmp/notes")
        assert isinstance(adapter2, MarkdownAdapter)
        assert adapter2.name == "markdown"

        adapter3 = get_adapter("CSV_ROWS", path="/tmp/rows.csv")
        assert isinstance(adapter3, CsvRowsAdapter)
        assert adapter3.name == "csv_rows"

    def test_adapter_name_normalization_hyphens(self):
        # Hyphens should be treated as underscores
        adapter1 = get_adapter("jsonl-notes", path="/tmp/notes.jsonl")
        assert isinstance(adapter1, JsonlNotesAdapter)
        assert adapter1.name == "jsonl_notes"

        adapter2 = get_adapter("csv-rows", path="/tmp/rows.csv")
        assert isinstance(adapter2, CsvRowsAdapter)
        assert adapter2.name == "csv_rows"

        adapter3 = get_adapter("markdown-links", path="/tmp/notes")
        assert isinstance(adapter3, MarkdownLinksAdapter)
        assert adapter3.name == "markdown_links"

    def test_adapter_name_normalization_combined(self):
        # Combined: whitespace + case + hyphens
        adapter1 = get_adapter(" JSONL-NOTES ", path="/tmp/notes.jsonl")
        assert isinstance(adapter1, JsonlNotesAdapter)
        assert adapter1.name == "jsonl_notes"

        adapter2 = get_adapter("  Markdown-Callouts  ", path="/tmp/notes")
        assert isinstance(adapter2, MarkdownCalloutsAdapter)
        assert adapter2.name == "markdown_callouts"


# Registry-specific error handling tests


def test_registry_get_adapter_unknown_name():
    """Test that requesting an unknown adapter raises KeyError with helpful message."""
    import pytest

    with pytest.raises(KeyError, match="Unknown adapter: nonexistent_adapter"):
        get_adapter("nonexistent_adapter")


def test_registry_get_adapter_unknown_name_shows_available():
    """Test that error message includes list of available adapters."""
    import pytest

    with pytest.raises(KeyError) as exc_info:
        get_adapter("invalid_name")

    error_msg = str(exc_info.value)
    # Should list some known adapters
    assert "atom" in error_msg or "feed" in error_msg or "mbox" in error_msg
    assert "Available:" in error_msg


def test_registry_list_adapters_returns_sorted():
    """Test that list_adapters returns a sorted list."""
    adapters = list_adapters()

    assert isinstance(adapters, list)
    assert len(adapters) > 0
    # Verify it's sorted
    assert adapters == sorted(adapters)


def test_registry_list_adapters_contains_expected():
    """Test that list_adapters includes known adapters."""
    adapters = list_adapters()

    # Check for some well-known adapters
    assert "atom" in adapters
    assert "feed" in adapters
    assert "mbox" in adapters
    assert "markdown" in adapters
    assert "bibtex" in adapters


def test_registry_get_all_adapters_returns_instances():
    """Test that get_all_adapters returns adapter instances."""
    from graph.adapters.base import SourceAdapter

    adapters = get_all_adapters()

    assert isinstance(adapters, list)
    assert len(adapters) > 0
    # All should be SourceAdapter instances
    assert all(isinstance(a, SourceAdapter) for a in adapters)


def test_registry_get_all_adapters_unique_names():
    """Test that all adapters have unique names."""
    adapters = get_all_adapters()
    names = [a.name for a in adapters]

    # All names should be unique (no duplicate registrations)
    assert len(names) == len(set(names))


def test_registry_get_adapter_passes_kwargs():
    """Test that get_adapter passes keyword arguments to adapter constructor."""
    from graph.adapters.feed import FeedAdapter

    adapter = get_adapter("feed", sources="https://example.com/feed.xml")

    assert isinstance(adapter, FeedAdapter)
    assert adapter.sources == "https://example.com/feed.xml"


def test_registry_adapter_count_stability():
    """Test that adapter count is stable (catches accidental removals)."""
    adapters = list_adapters()

    # Should have a substantial number of adapters
    # This test will catch if adapters are accidentally removed from registry
    assert len(adapters) >= 50  # We have many adapters


def test_registry_no_duplicate_adapter_names():
    """Test that there are no duplicate adapter names in the registry."""
    from graph.adapters.registry import _ADAPTERS

    # Check that all keys in _ADAPTERS are unique (Python dict guarantees this)
    adapter_names = list(_ADAPTERS.keys())
    assert len(adapter_names) == len(set(adapter_names))


def test_registry_all_adapters_have_name_property():
    """Test that all registered adapters implement the name property."""
    adapters = get_all_adapters()

    for adapter in adapters:
        # Each adapter must have a name
        assert hasattr(adapter, "name")
        assert isinstance(adapter.name, str)
        assert len(adapter.name) > 0


def test_registry_all_adapters_have_entity_types():
    """Test that all registered adapters implement entity_types property."""
    adapters = get_all_adapters()

    for adapter in adapters:
        # Each adapter must have entity_types
        assert hasattr(adapter, "entity_types")
        assert isinstance(adapter.entity_types, list)


def test_registry_all_adapters_have_ingest_method():
    """Test that all registered adapters implement the ingest method."""
    adapters = get_all_adapters()

    for adapter in adapters:
        # Each adapter must have ingest method
        assert hasattr(adapter, "ingest")
        assert callable(adapter.ingest)


def test_registry_adapter_name_matches_registry_key():
    """Test that adapter name property matches its registry key."""
    from graph.adapters.registry import _ADAPTERS

    for key, adapter_class in _ADAPTERS.items():
        adapter = adapter_class()
        # Normalized adapter name should match registry key
        normalized_name = adapter.name.strip().lower().replace("-", "_")
        assert normalized_name == key, f"Adapter {adapter_class.__name__} has name '{adapter.name}' but is registered as '{key}'"


def test_registry_name_normalization_whitespace():
    """Test that adapter names with leading/trailing whitespace are normalized."""
    from graph.adapters.feed import FeedAdapter

    adapter = get_adapter("  feed  ")
    assert isinstance(adapter, FeedAdapter)


def test_registry_name_normalization_case_insensitive():
    """Test that adapter name lookup is case-insensitive."""
    from graph.adapters.mbox import MboxAdapter

    adapter1 = get_adapter("MBOX")
    adapter2 = get_adapter("mbox")
    adapter3 = get_adapter("Mbox")

    assert all(isinstance(a, MboxAdapter) for a in [adapter1, adapter2, adapter3])


def test_registry_get_adapter_with_empty_string():
    """Test that empty string adapter name raises KeyError."""
    import pytest

    with pytest.raises(KeyError):
        get_adapter("")


def test_registry_get_adapter_with_whitespace_only():
    """Test that whitespace-only adapter name raises KeyError."""
    import pytest

    with pytest.raises(KeyError):
        get_adapter("   ")


def test_registry_performance_with_many_lookups():
    """Test that multiple adapter lookups are efficient."""
    import time

    start = time.time()
    for _ in range(1000):
        get_adapter("feed", sources="test.xml")
    elapsed = time.time() - start

    # Should complete 1000 lookups in well under 1 second
    assert elapsed < 1.0


def test_registry_list_adapters_consistency():
    """Test that list_adapters returns the same list on multiple calls."""
    list1 = list_adapters()
    list2 = list_adapters()

    assert list1 == list2


def test_registry_get_all_adapters_creates_new_instances():
    """Test that get_all_adapters creates fresh instances each time."""
    adapters1 = get_all_adapters()
    adapters2 = get_all_adapters()

    # Should be different instances
    assert adapters1 is not adapters2
    # But same count
    assert len(adapters1) == len(adapters2)


def test_registry_adapter_classes_are_subclasses():
    """Test that all registered adapter classes are SourceAdapter subclasses."""
    from graph.adapters.base import SourceAdapter
    from graph.adapters.registry import _ADAPTERS

    for name, adapter_class in _ADAPTERS.items():
        assert issubclass(adapter_class, SourceAdapter), f"Adapter '{name}' is not a SourceAdapter subclass"


def test_registry_no_import_errors():
    """Test that all registered adapters can be imported without errors."""
    from graph.adapters.registry import _ADAPTERS

    # If we got here, all imports in registry.py succeeded
    assert len(_ADAPTERS) > 0


def test_registry_adapter_instantiation_without_args():
    """Test that adapters can be instantiated with default arguments."""
    from graph.adapters.registry import _ADAPTERS

    # Most adapters should be instantiable with no args or empty kwargs
    for name, adapter_class in _ADAPTERS.items():
        try:
            adapter = adapter_class()
            assert adapter.name == name
        except TypeError:
            # Some adapters may require arguments, which is acceptable
            pass
