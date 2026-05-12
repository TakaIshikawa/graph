from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_source_metadata_density_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, source_project: SourceProject | str | None, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_metadata_density_groups_by_source_and_unknown():
    text = export_source_metadata_density_csv(
        [
            unit("b", SourceProject.MAX, {"priority": "high", "status": "done"}),
            unit("a", None, {"priority": "low"}),
            unit("c", SourceProject.MAX, {}),
            unit("d", SourceProject.MAX, {"priority": "medium"}),
        ]
    )

    assert text.splitlines()[0] == (
        "source_project,unit_count,units_with_metadata,metadata_coverage_percent,"
        "distinct_metadata_keys,average_keys_per_unit,top_metadata_keys"
    )
    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "3",
            "units_with_metadata": "2",
            "metadata_coverage_percent": "66.67",
            "distinct_metadata_keys": "2",
            "average_keys_per_unit": "1.00",
            "top_metadata_keys": "priority (2); status (1)",
        },
        {
            "source_project": "Unknown",
            "unit_count": "1",
            "units_with_metadata": "1",
            "metadata_coverage_percent": "100.00",
            "distinct_metadata_keys": "1",
            "average_keys_per_unit": "1.00",
            "top_metadata_keys": "priority (1)",
        },
    ]


def test_source_metadata_density_min_units_filters_small_groups_and_validates():
    text = export_source_metadata_density_csv(
        [
            unit("a", "Source A", {"a": 1}),
            unit("b", "Source B", {"b": 1}),
            unit("c", "Source B", {}),
        ],
        min_units=2,
    )

    assert [row["source_project"] for row in rows(text)] == ["Source B"]

    with pytest.raises(ValueError, match="min_units"):
        export_source_metadata_density_csv([], min_units=0)
    with pytest.raises(ValueError, match="min_units"):
        export_source_metadata_density_csv([], min_units=True)


def test_source_metadata_density_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "density.csv"
    units = [unit("a", "Source A", {"a": 1}), unit("b", "Source A", {})]

    expected = export_source_metadata_density_csv(units)
    stats = export_source_metadata_density_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "source_project_count": 1,
        "rows_exported": 1,
        "min_units": 1,
        "bytes_written": path.stat().st_size,
    }


def test_source_metadata_density_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", {"z": 1}),
        unit("b", "Source A", {"a": 1, "z": 2}),
        unit("c", "Source A", {"a": 3}),
    ]

    assert export_source_metadata_density_csv(units) == export_source_metadata_density_csv(reversed(units))
