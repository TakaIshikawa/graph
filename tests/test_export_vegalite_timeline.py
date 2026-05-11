from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_units_to_vegalite_timeline
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

CREATED_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
INGESTED_TIME = datetime(2026, 5, 2, 8, 30, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 3, 9, 45, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str | None = None,
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    source_project: SourceProject | str = SourceProject.CSV,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title or f"Title {unit_id}",
        content="A compact research note.",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags if tags is not None else ["timeline", "export"],
        created_at=CREATED_TIME,
        ingested_at=INGESTED_TIME,
        updated_at=UPDATED_TIME,
    )


def undated_unit(unit_id: str) -> KnowledgeUnit:
    return unit(unit_id, metadata={"date": "not a date"}).model_copy(
        update={"created_at": None, "updated_at": None, "ingested_at": None}
    )


def test_export_units_to_vegalite_timeline_returns_self_contained_spec():
    spec = json.loads(export_units_to_vegalite_timeline(unit("a", metadata={"date": "2026-01-01"})))

    assert spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"
    assert spec["mark"] == {"type": "tick", "tooltip": True}
    assert spec["encoding"]["x"]["field"] == "date"
    assert spec["data"]["values"] == [
        {
            "date": "2026-01-01",
            "id": "a",
            "source_project": "csv",
            "tags": "export, timeline",
            "title": "Title a",
        }
    ]


def test_export_units_to_vegalite_timeline_sorts_values_deterministically():
    text_a = export_units_to_vegalite_timeline(
        [
            unit("b", "Beta", metadata={"date": "2026-01-02"}),
            unit("a", "Alpha", metadata={"date": "2026-01-02"}),
            unit("c", "Charlie", metadata={"date": "2026-01-01"}),
        ]
    )
    text_b = export_units_to_vegalite_timeline(
        [
            unit("c", "Charlie", metadata={"date": "2026-01-01"}),
            unit("a", "Alpha", metadata={"date": "2026-01-02"}),
            unit("b", "Beta", metadata={"date": "2026-01-02"}),
        ]
    )

    assert text_a == text_b
    assert [value["title"] for value in json.loads(text_a)["data"]["values"]] == ["Charlie", "Alpha", "Beta"]


def test_export_units_to_vegalite_timeline_skips_undated_units_for_path_writes(tmp_path):
    path = tmp_path / "timeline.vl.json"

    stats = export_units_to_vegalite_timeline(
        [unit("dated", metadata={"date": "2026-01-01"}), undated_unit("skip")],
        path,
    )

    assert [value["id"] for value in json.loads(path.read_text(encoding="utf-8"))["data"]["values"]] == ["dated"]
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "skipped_count": 1,
        "bytes_written": path.stat().st_size,
    }


def test_export_units_to_vegalite_timeline_embeds_source_project_and_tags():
    value = json.loads(
        export_units_to_vegalite_timeline(
            unit("a", metadata={"date": "2026-01-01"}, source_project=SourceProject.PINBOARD, tags=["zeta", "alpha"])
        )
    )["data"]["values"][0]

    assert value["source_project"] == "pinboard"
    assert value["tags"] == "alpha, zeta"
