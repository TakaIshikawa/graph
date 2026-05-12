from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_source_timeline_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    source_id: str | None = None,
    content_type: ContentType | str = ContentType.INSIGHT,
    confidence: object = None,
    created_at: object = None,
    ingested_at: object = None,
    updated_at: object = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=content_type,
        metadata=metadata or {},
        confidence=confidence,
        created_at=created_at,
        ingested_at=ingested_at,
        updated_at=updated_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_source_timeline_csv_empty_input_has_header_only():
    assert export_unit_source_timeline_csv([]) == (
        "unit_id,unit_type,source_id,source_project,source_entity_type,observation_date,"
        "confidence,metadata_key_count\n"
    )


def test_unit_source_timeline_csv_emits_multiple_dated_observations():
    text = export_unit_source_timeline_csv(
        [
            unit(
                "a",
                confidence=0.75,
                created_at="2024-01-02",
                updated_at="2024-01-05T10:00:00Z",
                metadata={"observed_dates": ["2024-01-03", "2024-01-02"], "label": "A"},
            )
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "unit_type": "insight",
            "source_id": "source-a",
            "source_project": "max",
            "source_entity_type": "note",
            "observation_date": "2024-01-02",
            "confidence": "0.75",
            "metadata_key_count": "2",
        },
        {
            "unit_id": "a",
            "unit_type": "insight",
            "source_id": "source-a",
            "source_project": "max",
            "source_entity_type": "note",
            "observation_date": "2024-01-03",
            "confidence": "0.75",
            "metadata_key_count": "2",
        },
        {
            "unit_id": "a",
            "unit_type": "insight",
            "source_id": "source-a",
            "source_project": "max",
            "source_entity_type": "note",
            "observation_date": "2024-01-05",
            "confidence": "0.75",
            "metadata_key_count": "2",
        },
    ]


def test_unit_source_timeline_csv_represents_undated_metadata_light_units():
    text = export_unit_source_timeline_csv(
        [unit("a", source_project=None, source_entity_type=None, confidence="not numeric")]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "unit_type": "insight",
            "source_id": "source-a",
            "source_project": "Unknown",
            "source_entity_type": "Unknown",
            "observation_date": "",
            "confidence": "",
            "metadata_key_count": "0",
        }
    ]


def test_unit_source_timeline_csv_sorts_deterministically_for_reversed_input():
    units = [
        unit("b", source_project="Source B", created_at="2024-02-01"),
        unit("a", source_project="Source A", created_at="2024-01-01"),
    ]

    assert export_unit_source_timeline_csv(units) == export_unit_source_timeline_csv(reversed(units))


def test_unit_source_timeline_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "timeline.csv"
    units = [unit("a", created_at="2024-01-01")]

    expected = export_unit_source_timeline_csv(units)
    stats = export_unit_source_timeline_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
