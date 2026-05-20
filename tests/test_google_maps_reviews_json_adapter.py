from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_maps_reviews_json import GoogleMapsReviewsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_google_maps_reviews_json_ingests_container_metadata_and_registry(tmp_path):
    export = tmp_path / "reviews.json"
    export.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "review_id": "rev-1",
                        "place": {
                            "name": "Blue Bottle Coffee",
                            "place_id": "ChIJ123",
                            "address": "1 Ferry Building, San Francisco, CA",
                            "coordinates": {"lat": 37.795, "lng": -122.393},
                            "categories": ["Coffee shop", "Cafe"],
                            "url": "https://maps.google.com/?cid=123",
                        },
                        "review": {
                            "rating": 4,
                            "text": "Good espresso.",
                            "visited_at": "2025-02-01",
                            "created_at": "2025-02-03T04:05:06Z",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = GoogleMapsReviewsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOOGLE_MAPS_REVIEWS_JSON
    assert unit.source_entity_type == "place_review"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.metadata["place_name"] == "Blue Bottle Coffee"
    assert unit.metadata["place_id"] == "ChIJ123"
    assert unit.metadata["address"] == "1 Ferry Building, San Francisco, CA"
    assert unit.metadata["latitude"] == 37.795
    assert unit.metadata["longitude"] == -122.393
    assert unit.metadata["rating"] == 4.0
    assert unit.metadata["review_text"] == "Good espresso."
    assert unit.metadata["categories"] == ["Coffee shop", "Cafe"]
    assert unit.metadata["url"] == "https://maps.google.com/?cid=123"
    assert unit.metadata["reviewed_at"] == "2025-02-03T04:05:06+00:00"
    assert unit.metadata["visited_at"] == "2025-02-01T00:00:00+00:00"
    assert unit.metadata["source_file"] == "reviews.json"
    assert unit.metadata["record"]["review_id"] == "rev-1"
    assert "Place: Blue Bottle Coffee" in unit.content
    assert "Review: Good espresso." in unit.content
    assert unit.updated_at == datetime(2025, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert get_adapter("google_maps_reviews_json", path=str(export)).name == "google_maps_reviews_json"


def test_google_maps_reviews_json_skips_bad_rows_dedupes_sorts_and_since_filters(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "old.json").write_text(
        json.dumps([{"placeName": "Old Place", "rating": 3, "date": "2025-01-01"}]),
        encoding="utf-8",
    )
    (tmp_path / "new.json").write_text(
        json.dumps(
            {
                "data": [
                    {},
                    {"placeName": "Beta", "placeId": "beta", "rating": 5, "date": "2025-01-03", "text": "Nice"},
                    {"placeName": "Alpha", "placeId": "alpha", "rating": 4, "date": "2025-01-04", "text": "Great"},
                    {"placeName": "Alpha", "placeId": "alpha", "rating": 4, "date": "2025-01-04", "text": "Great"},
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = GoogleMapsReviewsJsonAdapter(path=str(tmp_path))
    sync = SyncState(
        source_project="google_maps_reviews_json",
        source_entity_type="place_review",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert sorted(unit.title for unit in first.units) == ["Alpha", "Beta"]
    assert [unit.source_id for unit in first.units] == sorted(unit.source_id for unit in first.units)
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert len(first.units) == 2
    assert adapter.ingest(entity_types=["place_visit"]).units == []
