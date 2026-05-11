from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_units_to_geojson
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.CSV,
        source_id=f"source-{unit_id}",
        source_entity_type="place",
        title=f"Place {unit_id}",
        content="Place note.",
        content_type=ContentType.ARTIFACT,
        tags=["map"],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata or {},
    )


def test_export_units_to_geojson_emits_feature_collection():
    data = json.loads(
        export_units_to_geojson(
            [
                unit("a", metadata={"latitude": "35.0", "longitude": "139.0", "name": "Tokyo"}),
                unit("b", metadata={"location": {"latitude": 40.7, "longitude": -74.0, "name": "NYC"}}),
                unit("c", metadata={"latitude": "bad", "longitude": "139.0"}),
            ],
            include_summary=True,
        )
    )

    assert data["type"] == "FeatureCollection"
    assert [feature["geometry"]["coordinates"] for feature in data["features"]] == [
        [139.0, 35.0],
        [-74.0, 40.7],
    ]
    assert data["features"][0]["properties"]["metadata"] == {"name": "Tokyo"}
    assert data["features"][1]["properties"]["metadata"] == {"location": {"name": "NYC"}}
    assert data["metadata"]["units_without_coordinates"] == 1


def test_export_units_to_geojson_writes_file(tmp_path):
    path = tmp_path / "units.geojson"

    text = export_units_to_geojson(unit("a", metadata={"lat": 1, "lon": 2}), path)

    assert path.read_text(encoding="utf-8") == text
