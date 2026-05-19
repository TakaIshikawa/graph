from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_location_coverage_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: object = "Source A",
    source_entity_type: str = "note",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_location_coverage_groups_by_source_and_type():
    text = export_unit_location_coverage_csv(
        [
            unit("b", source_entity_type="checkin", metadata={"place": "Cafe", "country": "US"}),
            unit("a", source_entity_type="checkin", metadata={"latitude": "37.7", "longitude": -122.4, "city": "SF"}),
            unit("c", source_entity_type="note", metadata={}),
            unit("d", source_project="Source B", source_entity_type="place", metadata={"geohash": "9q8yy"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Source A",
            "source_entity_type": "checkin",
            "unit_count": "2",
            "coordinate_count": "1",
            "named_place_count": "1",
            "country_count": "1",
            "missing_location_count": "0",
            "representative_unit_ids": "a; b",
        },
        {
            "source_project": "Source A",
            "source_entity_type": "note",
            "unit_count": "1",
            "coordinate_count": "0",
            "named_place_count": "0",
            "country_count": "0",
            "missing_location_count": "1",
            "representative_unit_ids": "c",
        },
        {
            "source_project": "Source B",
            "source_entity_type": "place",
            "unit_count": "1",
            "coordinate_count": "0",
            "named_place_count": "1",
            "country_count": "0",
            "missing_location_count": "0",
            "representative_unit_ids": "d",
        },
    ]


def test_unit_location_coverage_treats_malformed_coordinate_pair_as_missing_coordinates():
    text = export_unit_location_coverage_csv(
        [
            unit("a", metadata={"lat": "not-a-number", "lng": "-122.4"}),
            unit("b", metadata={"lat": "37.7"}),
            unit("c", metadata={"lon": "-122.4"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Source A",
            "source_entity_type": "note",
            "unit_count": "3",
            "coordinate_count": "0",
            "named_place_count": "0",
            "country_count": "0",
            "missing_location_count": "0",
            "representative_unit_ids": "a; b; c",
        }
    ]


def test_unit_location_coverage_supports_mapping_inputs_and_unknown_fallbacks():
    text = export_unit_location_coverage_csv([{"id": "a", "metadata": {"country": "JP"}}])

    assert rows(text) == [
        {
            "source_project": "Unknown",
            "source_entity_type": "Unknown",
            "unit_count": "1",
            "coordinate_count": "0",
            "named_place_count": "0",
            "country_count": "1",
            "missing_location_count": "0",
            "representative_unit_ids": "a",
        }
    ]


def test_unit_location_coverage_path_and_file_like_output(tmp_path):
    units = [unit("a", metadata={"country": "US"})]
    expected = export_unit_location_coverage_csv(units)
    path = tmp_path / "reports" / "unit-location-coverage.csv"

    stats = export_unit_location_coverage_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }

    buffer = StringIO()
    file_stats = export_unit_location_coverage_csv(units, buffer)
    assert buffer.getvalue() == expected
    assert file_stats["bytes_written"] == len(expected)


def test_unit_location_coverage_is_deterministic_for_reversed_input():
    units = [
        unit("b", source_project="Source B", metadata={"country": "US"}),
        unit("a", source_project="Source A", metadata={"place": "Cafe"}),
    ]

    assert export_unit_location_coverage_csv(units) == export_unit_location_coverage_csv(reversed(units))
