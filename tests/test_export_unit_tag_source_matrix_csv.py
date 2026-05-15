from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_tag_source_matrix_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None, tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="item",
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        tags=tags or [],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_tag_source_matrix_csv_empty_input_returns_header():
    assert export_unit_tag_source_matrix_csv([]) == (
        "tag,source_type,source_label,unit_count,source_observation_count\n"
    )


def test_export_unit_tag_source_matrix_csv_counts_units_and_source_observations():
    text = export_unit_tag_source_matrix_csv(
        [
            unit(
                "a",
                tags=["alpha", "beta"],
                metadata={"sources": [{"type": "web", "label": "Docs"}, {"type": "web", "label": "Docs"}]},
            ),
            unit("b", tags=["alpha"], metadata={"source": {"type": "web", "label": "Docs"}}),
        ]
    )

    result = {(row["tag"], row["source_type"], row["source_label"]): row for row in rows(text)}
    assert result[("alpha", "web", "Docs")]["unit_count"] == "2"
    assert result[("alpha", "web", "Docs")]["source_observation_count"] == "3"
    assert result[("beta", "web", "Docs")]["unit_count"] == "1"
    assert result[("beta", "web", "Docs")]["source_observation_count"] == "2"


def test_export_unit_tag_source_matrix_csv_handles_untagged_and_sourceless_units():
    text = export_unit_tag_source_matrix_csv([unit("a")])

    assert rows(text) == [
        {
            "tag": "unknown",
            "source_type": "unknown",
            "source_label": "unknown",
            "unit_count": "1",
            "source_observation_count": "1",
        }
    ]


def test_export_unit_tag_source_matrix_csv_sorts_rows_deterministically():
    text = export_unit_tag_source_matrix_csv(
        [
            unit("z", tags=["zeta"], metadata={"source": {"type": "note", "label": "B"}}),
            unit("a", tags=["alpha"], metadata={"source": {"type": "note", "label": "A"}}),
        ]
    )

    assert [(row["tag"], row["source_label"]) for row in rows(text)] == [("alpha", "A"), ("zeta", "B")]


def test_export_unit_tag_source_matrix_csv_path_mode(tmp_path):
    path = tmp_path / "tag-source.csv"
    stats = export_unit_tag_source_matrix_csv([unit("a", tags=["alpha"])], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["tag"] == "alpha"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
