"""Tests for environment-backed settings."""

from __future__ import annotations

from graph.config import Settings


def test_source_path_settings_honor_environment_overrides(monkeypatch):
    overrides = {
        "GRAPH_DATABASE_URL": "/tmp/graph.db",
        "GRAPH_FORTY_TWO_DB": "/tmp/forty_two.db",
        "GRAPH_MAX_DB": "/tmp/max.db",
        "GRAPH_PRESENCE_DB": "/tmp/presence.db",
        "GRAPH_KINDLE_DB": "/tmp/kindle.db",
        "GRAPH_SOTA_DB": "/tmp/sota.db",
        "GRAPH_ME_CONFIG": "/tmp/projects.yaml",
        "GRAPH_OBSIDIAN_VAULT_PATH": "/tmp/obsidian",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    loaded = Settings(_env_file=None)

    assert loaded.database_url == "/tmp/graph.db"
    assert loaded.forty_two_db == "/tmp/forty_two.db"
    assert loaded.max_db == "/tmp/max.db"
    assert loaded.presence_db == "/tmp/presence.db"
    assert loaded.kindle_db == "/tmp/kindle.db"
    assert loaded.sota_db == "/tmp/sota.db"
    assert loaded.me_config == "/tmp/projects.yaml"
    assert loaded.obsidian_vault_path == "/tmp/obsidian"
