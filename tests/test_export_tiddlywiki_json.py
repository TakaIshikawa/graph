from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_units_to_tiddlywiki_json
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, 30, 456000, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 2, 8, 30, 45, 123000, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_id: str | None = None,
    title: str = "Alpha Note",
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        tags=tags or ["zeta", "alpha tag"],
        metadata=metadata or {"rating": 5},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UPDATED_TIME,
    )


def test_export_units_to_tiddlywiki_json_returns_valid_tiddlers():
    parsed = json.loads(
        export_units_to_tiddlywiki_json(
            [
                unit(
                    "unit-a",
                    tags=["plain", "tag with spaces", "plain"],
                    metadata={"nested": {"when": UNIT_TIME}, "kind": ContentType.FINDING},
                )
            ]
        )
    )

    assert parsed == [
        {
            "title": "Alpha Note",
            "text": "Alpha Note content",
            "tags": "plain [[tag with spaces]]",
            "created": "20260501101530456",
            "modified": "20260502083045123",
            "type": "text/markdown",
            "source_project": "max",
            "source_id": "source-unit-a",
            "metadata": {
                "kind": "finding",
                "nested": {"when": "2026-05-01T10:15:30.456000+00:00"},
            },
        }
    ]


def test_export_units_to_tiddlywiki_json_adds_stable_duplicate_title_suffixes():
    parsed = json.loads(
        export_units_to_tiddlywiki_json(
            [
                unit("unit-b", source_id="beta", title="Same"),
                unit("unit-a", source_id="alpha", title="Same"),
            ]
        )
    )

    assert [item["title"] for item in parsed] == ["Same (beta)", "Same (alpha)"]


def test_export_units_to_tiddlywiki_json_writes_path(tmp_path):
    path = tmp_path / "nested" / "tiddlers.json"

    stats = export_units_to_tiddlywiki_json([unit("unit-a")], path)

    assert json.loads(path.read_text(encoding="utf-8"))[0]["title"] == "Alpha Note"
    assert stats == {
        "path": str(path),
        "units_scanned": 1,
        "units_exported": 1,
        "bytes_written": path.stat().st_size,
    }
