"""Tests for source adapters."""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest
import yaml

from graph.adapters.forty_two import FortyTwoAdapter
from graph.adapters.max_adapter import MaxAdapter
from graph.adapters.me import MeAdapter
from graph.adapters.presence import PresenceAdapter
from graph.adapters.registry import get_adapter, list_adapters


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
        result = adapter.ingest()
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
        assert "review:approved" in ideas[0].tags
        assert "approved" in ideas[0].tags
        assert len(result.edges) == 1
        assert result.edges[0].relation == "inspires"
        assert result.edges[0].from_unit_id == "ins-1"
        assert result.edges[0].to_unit_id == "bu-1"


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


class TestRegistry:
    def test_list_adapters(self):
        adapters = list_adapters()
        assert set(adapters) == {"forty_two", "max", "presence", "me"}

    def test_get_adapter(self):
        adapter = get_adapter("me", config_path="/tmp/test.yaml")
        assert adapter.name == "me"

    def test_unknown_adapter(self):
        with pytest.raises(KeyError):
            get_adapter("unknown")
