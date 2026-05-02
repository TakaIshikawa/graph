from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from graph.cli.main import app
from graph.export import export_tag_glossary_markdown
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


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


def _cleanup_db(path: str) -> None:
    db_path = Path(path)
    for candidate in (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def _unit(
    unit_id: str,
    *,
    title: str,
    source_project: SourceProject = SourceProject.MAX,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def test_export_tag_glossary_command_writes_markdown_and_reports_counts(
    tmp_path, monkeypatch
):
    store = _make_store()
    store.insert_unit(_unit("unit-a", title="Solar note", tags=["solar", "storage"]))
    store.insert_unit(_unit("unit-b", title="Battery note", tags=["storage"]))
    output_path = tmp_path / "reports" / "tags.md"
    calls = []

    def recording_export(*args, **kwargs):
        calls.append((args, kwargs))
        return export_tag_glossary_markdown(*args, **kwargs)

    monkeypatch.setattr("graph.cli.main._get_store", lambda: StoreProxy(store))
    monkeypatch.setattr("graph.cli.main.export_tag_glossary_markdown", recording_export)

    try:
        result = runner.invoke(app, ["export-tag-glossary", str(output_path)])

        assert result.exit_code == 0
        assert "Exported 2 tags from 2 units" in result.output
        assert output_path.exists()
        text = output_path.read_text(encoding="utf-8")
        assert "# Tag Glossary" in text
        assert "## storage" in text
        assert "Solar note (`unit-a`)" in text
        assert len(calls) == 1
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_tag_glossary_command_honors_filters_and_json(tmp_path, monkeypatch):
    store = _make_store()
    store.insert_unit(
        _unit(
            "unit-max-a",
            title="Max Solar",
            source_project=SourceProject.MAX,
            tags=["solar", "storage"],
        )
    )
    store.insert_unit(
        _unit(
            "unit-max-b",
            title="Max Storage",
            source_project=SourceProject.MAX,
            tags=["storage"],
        )
    )
    store.insert_unit(
        _unit(
            "unit-presence",
            title="Presence Solar",
            source_project=SourceProject.PRESENCE,
            tags=["solar"],
        )
    )
    output_path = tmp_path / "tags.md"
    monkeypatch.setattr("graph.cli.main._get_store", lambda: StoreProxy(store))

    try:
        result = runner.invoke(
            app,
            [
                "export-tag-glossary",
                str(output_path),
                "--source-project",
                "max",
                "--min-count",
                "2",
                "--no-include-descriptions",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["units_scanned"] == 2
        assert payload["tags_scanned"] == 2
        assert payload["tags_exported"] == 1
        assert payload["source_project"] == "max"
        assert payload["include_descriptions"] is False

        text = output_path.read_text(encoding="utf-8")
        assert "## storage" in text
        assert "## solar" not in text
        assert "_No examples included._" in text
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_tag_glossary_command_rejects_invalid_filter(monkeypatch, tmp_path):
    store = _make_store()
    monkeypatch.setattr("graph.cli.main._get_store", lambda: StoreProxy(store))

    try:
        result = runner.invoke(
            app,
            ["export-tag-glossary", str(tmp_path / "tags.md"), "--source-project", "unknown"],
        )

        assert result.exit_code != 0
        assert "source_project must be one of" in result.output
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]


def test_export_tag_glossary_command_rejects_invalid_output_path(
    monkeypatch, tmp_path
):
    store = _make_store()
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    monkeypatch.setattr("graph.cli.main._get_store", lambda: StoreProxy(store))

    try:
        result = runner.invoke(
            app,
            [
                "export-tag-glossary",
                str(blocked_parent / "tags.md"),
                "--json",
            ],
        )

        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert payload["error"] == "export_failed"
        assert payload["path"] == str(blocked_parent / "tags.md")
    finally:
        store.close()
        _cleanup_db(store._test_db_path)  # type: ignore[attr-defined]

