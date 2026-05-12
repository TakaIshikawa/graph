from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.inaturalist_observations_csv import INaturalistObservationsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_inaturalist_observations_csv_ingests_observations(tmp_path):
    path = tmp_path / "observations.csv"
    _write_csv(
        path,
        [
            {
                "id": "123",
                "observed_on": "2026-05-01",
                "created_at": "2026-05-02T03:00:00Z",
                "updated_at": "2026-05-03T03:00:00Z",
                "common_name": "California poppy",
                "scientific_name": "Eschscholzia californica",
                "iconic_taxon_name": "Plantae",
                "quality_grade": "research",
                "place_guess": "Berkeley, CA",
                "latitude": "37.8715",
                "longitude": "-122.2730",
                "geoprivacy": "open",
                "url": "https://www.inaturalist.org/observations/123",
                "description": "Blooming near the path.",
                "tags": "garden, orange",
            }
        ],
    )

    result = INaturalistObservationsCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.INATURALIST_OBSERVATIONS_CSV
    assert unit.source_id == "inaturalist_observations_csv:123"
    assert unit.title == "California poppy on 2026-05-01"
    assert unit.metadata["scientific_name"] == "Eschscholzia californica"
    assert unit.metadata["latitude"] == 37.8715
    assert unit.metadata["longitude"] == -122.273
    assert unit.metadata["geoprivacy"] == "open"
    assert unit.metadata["tags"] == ["garden", "orange"]
    assert unit.metadata["row"]["description"] == "Blooming near the path."
    assert unit.updated_at == datetime(2026, 5, 3, 3, tzinfo=timezone.utc)


def test_inaturalist_observations_csv_directory_since_and_invalid_coordinates(tmp_path):
    _write_csv(tmp_path / "old.csv", [{"id": "1", "common_name": "Old", "updated_at": "2026-04-01"}])
    _write_csv(tmp_path / "new.csv", [{"common_name": "New", "updated_at": "2026-05-01", "latitude": "northish", "longitude": "westish"}])
    since = SyncState(source_project="inaturalist_observations_csv", source_entity_type="observation", last_sync_at=datetime(2026, 4, 15, tzinfo=timezone.utc))

    result = INaturalistObservationsCsvAdapter(path=str(tmp_path)).ingest(since=since)

    assert [unit.metadata["common_name"] for unit in result.units] == ["New"]
    assert result.units[0].metadata["latitude"] == "northish"
    assert result.units[0].metadata["longitude"] == "westish"
    assert result.units[0].source_id.startswith("inaturalist_observations_csv:")
    assert get_adapter("inaturalist_observations_csv", path=str(tmp_path)).name == "inaturalist_observations_csv"


def test_inaturalist_observations_csv_filters_and_skips_empty_rows(tmp_path):
    path = tmp_path / "observations.csv"
    _write_csv(path, [{"id": "", "common_name": "", "scientific_name": ""}, {"scientific_name": "Danaus plexippus"}])

    result = INaturalistObservationsCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "Danaus plexippus"
    assert INaturalistObservationsCsvAdapter(path=str(path)).ingest(entity_types=["book"]).units == []
