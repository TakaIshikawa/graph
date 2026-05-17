"""Adapter for Lyft ride history CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LyftRidesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "lyft_rides_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["ride"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "ride" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        ride_id = first(row, "Ride ID", "Ride Id", "Ride UUID", "Ride Identifier", "ID")
        requested_text = first(row, "Requested Time", "Requested At", "Request Time", "Request Date", "Date")
        started_text = first(row, "Started Time", "Started At", "Start Time", "Pickup Time", "Begin Time")
        completed_text = first(row, "Completed Time", "Completed At", "End Time", "Dropoff Time", "Drop Off Time")
        requested_at = parse_datetime(requested_text)
        started_at = parse_datetime(started_text)
        completed_at = parse_datetime(completed_text)
        ride_type = first(row, "Ride Type", "Lyft Type", "Product", "Service", "Vehicle Type")
        status = first(row, "Status", "Ride Status")
        pickup_address = first(row, "Pickup Address", "Pickup Location", "Start Address", "Origin", "From")
        dropoff_address = first(row, "Dropoff Address", "Drop Off Address", "Drop-off Address", "Destination", "To")
        city = first(row, "City", "Ride City", "Region")
        distance = parse_float(first(row, "Distance", "Distance (mi)", "Distance (km)", "Ride Distance"))
        duration_seconds = parse_duration_seconds(first(row, "Duration", "Ride Duration", "Duration Seconds", "Duration (seconds)"))
        cost = parse_float(first(row, "Cost", "Total", "Total Paid", "Fare", "Fare Amount"))
        tip = parse_float(first(row, "Tip", "Driver Tip"))
        currency = first(row, "Currency", "Fare Currency", "ISO Currency Code")
        driver_name = first(row, "Driver Name", "Driver")

        if not any([ride_id, requested_text, started_text, completed_text, ride_type, status, pickup_address, dropoff_address, city, distance is not None, cost is not None, tip is not None, driver_name]):
            return None

        now = datetime.now(timezone.utc)
        ride_time = completed_at or started_at or requested_at
        metadata = clean_metadata(
            {
                "ride_id": ride_id,
                "requested_at": requested_at.isoformat() if requested_at else requested_text,
                "started_at": started_at.isoformat() if started_at else started_text,
                "completed_at": completed_at.isoformat() if completed_at else completed_text,
                "ride_type": ride_type,
                "status": status,
                "pickup_address": pickup_address,
                "dropoff_address": dropoff_address,
                "city": city,
                "distance": distance,
                "distance_unit": self._distance_unit(row),
                "duration_seconds": duration_seconds,
                "cost": cost,
                "tip": tip,
                "currency": currency,
                "driver_name": driver_name,
                "source_file": source_file,
            }
        )

        return KnowledgeUnit(
            source_project="lyft_rides_csv",
            source_id=f"lyft_rides_csv:{ride_id}" if ride_id else digest_source_id("lyft_rides_csv", requested_text, started_text, completed_text, pickup_address, dropoff_address, cost, index),
            source_entity_type="ride",
            title=self._title(city, ride_type, pickup_address, dropoff_address),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["lyft", "ride", city, ride_type, status] if tag)),
            created_at=requested_at or started_at or completed_at or now,
            updated_at=ride_time or now,
        )

    def _distance_unit(self, row: dict[str, Any]) -> str:
        for key in row:
            lowered = key.casefold()
            if "distance" not in lowered:
                continue
            if "km" in lowered or "kilometer" in lowered:
                return "km"
            if "mi" in lowered or "mile" in lowered:
                return "mi"
        return first(row, "Distance Unit", "Distance Units")

    def _title(self, city: str, ride_type: str, pickup_address: str, dropoff_address: str) -> str:
        route = " to ".join(part for part in [pickup_address, dropoff_address] if part)
        return " - ".join(part for part in [ride_type, city, route] if part) or "Lyft ride"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("requested_at", "Requested"),
            ("started_at", "Started"),
            ("completed_at", "Completed"),
            ("city", "City"),
            ("ride_type", "Type"),
            ("status", "Status"),
            ("cost", "Cost"),
            ("tip", "Tip"),
            ("distance", "Distance"),
            ("duration_seconds", "Duration seconds"),
            ("pickup_address", "Pickup"),
            ("dropoff_address", "Dropoff"),
            ("driver_name", "Driver"),
        ):
            if metadata.get(key) is not None:
                value = metadata[key]
                if key in {"cost", "tip"} and metadata.get("currency"):
                    value = f"{value} {metadata['currency']}"
                if key == "distance" and metadata.get("distance_unit"):
                    value = f"{value} {metadata['distance_unit']}"
                parts.append(f"{label}: {value}")
        return "\n".join(parts)
