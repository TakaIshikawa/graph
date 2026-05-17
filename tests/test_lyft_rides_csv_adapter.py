from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.lyft_rides_csv import LyftRidesCsvAdapter
from graph.types.models import SyncState


def test_lyft_rides_csv_ingests_ride_metadata_and_tags(tmp_path):
    export = tmp_path / "lyft.csv"
    export.write_text(
        "Ride ID,Requested Time,Started Time,Completed Time,Ride Type,Status,Pickup Address,Dropoff Address,City,Distance (mi),Duration,Cost,Tip,Currency,Driver Name\n"
        "ride-1,2026-05-01T08:00:00Z,2026-05-01T08:04:00Z,2026-05-01T08:24:00Z,Standard,completed,1 Start St,9 End Ave,Tokyo,7.2,20 min,$18.40,3.00,USD,Ada\n",
        encoding="utf-8",
    )

    unit = LyftRidesCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "lyft_rides_csv"
    assert unit.source_id == "lyft_rides_csv:ride-1"
    assert unit.source_entity_type == "ride"
    assert unit.metadata["requested_at"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["started_at"] == "2026-05-01T08:04:00+00:00"
    assert unit.metadata["completed_at"] == "2026-05-01T08:24:00+00:00"
    assert unit.metadata["cost"] == 18.4
    assert unit.metadata["tip"] == 3.0
    assert unit.metadata["distance"] == 7.2
    assert unit.metadata["distance_unit"] == "mi"
    assert unit.metadata["duration_seconds"] == 1200
    assert unit.created_at == datetime(2026, 5, 1, 8, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 1, 8, 24, tzinfo=timezone.utc)
    assert {"lyft", "ride", "Tokyo", "Standard", "completed"}.issubset(set(unit.tags))
    assert "Cost: 18.4 USD" in unit.content


def test_lyft_rides_csv_directory_aliases_since_and_entity_filter(tmp_path):
    (tmp_path / "old.csv").write_text(
        "Date,End Time,Service,Ride Status,Origin,Destination,Ride City,Total,Fare Currency,Distance (km),Duration Seconds\n"
        "2026-05-01,2026-05-01 00:30:00,Shared,completed,Old pickup,Old dropoff,Paris,12.00,EUR,4.5,1800\n",
        encoding="utf-8",
    )
    (tmp_path / "new.csv").write_text(
        "Request Date,Dropoff Time,Product,Status,From,To,City,Fare,Currency,Distance Unit,Ride Distance,Ride Duration\n"
        "2026-05-03 09:00,2026-05-03 09:18,XL,completed,Station,Hotel,Kyoto,2600,JPY,km,8.1,00:18:00\n",
        encoding="utf-8",
    )
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    sync = SyncState(source_project="lyft_rides_csv", source_entity_type="ride", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))
    adapter = LyftRidesCsvAdapter(path=str(tmp_path))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.metadata["city"] for unit in first.units] == ["Kyoto"]
    assert first.units[0].metadata["duration_seconds"] == 1080
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert [unit.metadata["city"] for unit in adapter.ingest().units] == ["Paris", "Kyoto"]
    assert adapter.ingest(entity_types=["trip"]).units == []


def test_lyft_rides_csv_tolerates_empty_and_sparse_rows(tmp_path):
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    (tmp_path / "sparse.csv").write_text("Ride ID,City,Cost\n, ,\n,New York,$33.10\n", encoding="utf-8")

    result = LyftRidesCsvAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["city"] == "New York"
    assert result.units[0].metadata["cost"] == 33.1
    assert result.units[0].source_id.startswith("lyft_rides_csv:")
