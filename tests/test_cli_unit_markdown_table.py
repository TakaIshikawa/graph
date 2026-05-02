from __future__ import annotations

import json

from typer.testing import CliRunner

from graph.cli.main import app
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


def unit(
    source_id: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=source_project,
        source_id=source_id,
        source_entity_type="insight",
        title=f"Unit {source_id}",
        content=f"Content {source_id}",
        content_type=content_type,
        tags=tags or [],
    )


def test_export_unit_markdown_table_command_writes_filtered_table_and_json_stats(
    tmp_path, monkeypatch
):
    store = Store(str(tmp_path / "graph.db"))
    included = store.insert_unit(unit("included", tags=["energy", "solar"]))
    store.insert_unit(unit("wrong-tag", tags=["research"]))
    store.insert_unit(
        unit("wrong-content", content_type=ContentType.FINDING, tags=["energy"])
    )
    store.insert_unit(
        unit(
            "wrong-source",
            source_project=SourceProject.FORTY_TWO,
            tags=["energy"],
        )
    )
    export_path = tmp_path / "reports" / "units.md"

    monkeypatch.setattr("graph.cli.main._get_store", lambda: StoreProxy(store))
    result = runner.invoke(
        app,
        [
            "export-unit-markdown-table",
            str(export_path),
            "--source-project",
            "max",
            "--content-type",
            "insight",
            "--tag",
            "energy",
            "--json",
        ],
    )

    try:
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload == {
            "path": str(export_path),
            "units_exported": 1,
            "filters": {
                "tag": "energy",
                "source_project": "max",
                "content_type": "insight",
            },
        }

        text = export_path.read_text(encoding="utf-8")
        assert text.startswith(
            "| id | title | source_project | content_type | created_at | tags | content_excerpt |\n"
        )
        assert included.id in text
        assert "Unit included" in text
        assert "energy; solar" in text
        assert "wrong-tag" not in text
        assert "wrong-content" not in text
        assert "wrong-source" not in text
    finally:
        store.close()


def test_export_unit_markdown_table_command_reports_human_count(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "graph.db"))
    store.insert_unit(unit("a"))
    export_path = tmp_path / "units.md"

    monkeypatch.setattr("graph.cli.main._get_store", lambda: StoreProxy(store))
    result = runner.invoke(app, ["export-unit-markdown-table", str(export_path)])

    try:
        assert result.exit_code == 0
        assert result.output.strip() == f"Exported 1 units to {export_path}"
        assert export_path.exists()
    finally:
        store.close()
