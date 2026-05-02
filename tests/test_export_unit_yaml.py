from __future__ import annotations

from datetime import datetime, timezone

import yaml

from graph.export import export_units_to_yaml
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    source_id: str | None = None,
    title: str | None = None,
    content_type: ContentType = ContentType.FINDING,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="insight",
        title=title or f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        tags=tags or ["storage", "solar"],
        confidence=0.8,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata or {},
        embedding=[0.1, 0.2],
    )


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_export_units_yaml_writes_public_unit_fields_without_embedding_and_stats(tmp_path):
    path = tmp_path / "nested" / "units.yaml"

    stats = export_units_to_yaml([unit("unit-a")], path)
    records = read_yaml(path)

    assert stats == {
        "path": str(path),
        "units_scanned": 1,
        "units_exported": 1,
        "bytes_written": path.stat().st_size,
    }
    assert list(records[0]) == [
        "id",
        "source_project",
        "source_id",
        "source_entity_type",
        "title",
        "content",
        "content_type",
        "tags",
        "confidence",
        "utility_score",
        "created_at",
        "ingested_at",
        "updated_at",
        "metadata",
    ]
    assert "embedding" not in records[0]


def test_export_units_yaml_preserves_sequence_order(tmp_path):
    path = tmp_path / "units.yaml"

    export_units_to_yaml([unit("unit-c"), unit("unit-a"), unit("unit-b")], path)

    assert [record["id"] for record in read_yaml(path)] == ["unit-c", "unit-a", "unit-b"]


def test_export_units_yaml_sorts_non_sequence_iterables_by_source_project_id_title(tmp_path):
    path = tmp_path / "units.yaml"
    units = (
        item
        for item in [
            unit("unit-c", source_project=SourceProject.PINBOARD, source_id="2", title="Beta"),
            unit("unit-a", source_project=SourceProject.MAX, source_id="2", title="Beta"),
            unit("unit-b", source_project=SourceProject.MAX, source_id="1", title="Zeta"),
            unit("unit-d", source_project=SourceProject.MAX, source_id="1", title="Alpha"),
        ]
    )

    export_units_to_yaml(units, path)

    assert [record["id"] for record in read_yaml(path)] == [
        "unit-d",
        "unit-b",
        "unit-a",
        "unit-c",
    ]


def test_export_units_yaml_serializes_enums_datetimes_and_metadata_stably(tmp_path):
    path = tmp_path / "units.yaml"

    export_units_to_yaml(
        [
            unit(
                "unit-a",
                source_project=SourceProject.PINBOARD,
                content_type=ContentType.IDEA,
                metadata={
                    "zeta": SourceProject.MAX,
                    "alpha": {"when": UNIT_TIME, "content_type": ContentType.ARTIFACT},
                },
            )
        ],
        path,
    )

    record = read_yaml(path)[0]
    assert record["source_project"] == "pinboard"
    assert record["content_type"] == "idea"
    assert record["created_at"] == "2026-05-01T10:15:00+00:00"
    assert record["ingested_at"] == "2026-05-01T10:15:00+00:00"
    assert record["updated_at"] == "2026-05-01T10:15:00+00:00"
    assert record["metadata"] == {
        "alpha": {
            "content_type": "artifact",
            "when": "2026-05-01T10:15:00+00:00",
        },
        "zeta": "max",
    }


def test_export_units_yaml_empty_export_writes_empty_list(tmp_path):
    path = tmp_path / "empty.yaml"

    stats = export_units_to_yaml([], path)

    assert path.read_text(encoding="utf-8") == "[]\n"
    assert stats == {
        "path": str(path),
        "units_scanned": 0,
        "units_exported": 0,
        "bytes_written": 3,
    }
