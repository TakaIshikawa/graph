from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.libby_holds_csv import LibbyHoldsCsvAdapter
from graph.types.models import SyncState


def test_libby_holds_csv_ingests_normalized_hold_metadata(tmp_path):
    export = tmp_path / "holds.csv"
    export.write_text(
        "Hold ID,Title,Authors,Format,Library,Estimated Wait,Queue Position,Placed Date,Suspended Until,Source URL\n"
        "hold-1,The Queue,\"Ada Lovelace; Grace Hopper\",Audiobook,City Library,3 weeks,#12,2026-05-01,\"June 1, 2026\",https://libby.example/hold\n",
        encoding="utf-8",
    )

    result = LibbyHoldsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "libby_holds_csv"
    assert unit.source_entity_type == "libby_hold"
    assert unit.source_id == LibbyHoldsCsvAdapter(path=str(export)).ingest().units[0].source_id
    assert unit.metadata["title"] == "The Queue"
    assert unit.metadata["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert unit.metadata["format"] == "Audiobook"
    assert unit.metadata["library"] == "City Library"
    assert unit.metadata["estimated_wait"] == "3 weeks"
    assert unit.metadata["estimated_wait_days"] == 21
    assert unit.metadata["queue_position"] == 12
    assert unit.metadata["placed_at"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["suspended_until"] == "2026-06-01T00:00:00+00:00"
    assert unit.metadata["source_url"] == "https://libby.example/hold"
    assert unit.created_at == datetime(2026, 5, 1, tzinfo=timezone.utc)


def test_libby_holds_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "holds.csv"
    export.write_text(
        "Title,Author,Placed Date,Queue Position,Estimated Wait\n"
        "Old,Ada,2026-04-01,1,soon\n"
        "New,Ada,2026-05-03,2,14 days\n",
        encoding="utf-8",
    )

    since = SyncState(source_project="libby_holds_csv", source_entity_type="libby_hold", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    result = LibbyHoldsCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New"]
    assert result.units[0].metadata["estimated_wait_days"] == 14
    assert LibbyHoldsCsvAdapter(path=str(export)).ingest(entity_types=["loan"]).units == []


def test_libby_holds_csv_source_id_falls_back_to_title_author_and_placed_date(tmp_path):
    export = tmp_path / "holds.csv"
    export.write_text(
        "Title,Author,Placed Date\nStable,Ada,2026-05-01\n",
        encoding="utf-8",
    )

    first = LibbyHoldsCsvAdapter(path=str(export)).ingest().units[0]
    second = LibbyHoldsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("libby_holds_csv:")


def test_libby_holds_csv_emits_author_aggregates_and_edges(tmp_path):
    export = tmp_path / "holds.csv"
    export.write_text(
        "Title,Authors,Format,Library,Placed Date\n"
        "One,Ada Lovelace,eBook,City,2026-05-01\n"
        "Two,ada lovelace,Audiobook,County,2026-05-03\n"
        "Three,Grace Hopper,eBook,City,2026-05-02\n",
        encoding="utf-8",
    )

    result = LibbyHoldsCsvAdapter(path=str(export)).ingest(entity_types=["libby_hold", "author"])
    authors = {unit.title: unit for unit in result.units if unit.source_entity_type == "author"}

    assert set(authors) == {"Ada Lovelace", "Grace Hopper"}
    ada = authors["Ada Lovelace"]
    assert ada.metadata["hold_count"] == 2
    assert ada.metadata["libraries"] == ["City", "County"]
    assert ada.metadata["formats"] == ["Audiobook", "eBook"]
    assert ada.metadata["earliest_placed_at"] == "2026-05-01T00:00:00+00:00"
    assert ada.metadata["latest_placed_at"] == "2026-05-03T00:00:00+00:00"
    assert ada.metadata["source_files"] == ["holds.csv"]
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "hold_author"]) == 3

    author_only = LibbyHoldsCsvAdapter(path=str(export)).ingest(entity_types=["author"])
    hold_only = LibbyHoldsCsvAdapter(path=str(export)).ingest(entity_types=["libby_hold"])
    assert {unit.source_entity_type for unit in author_only.units} == {"author"}
    assert author_only.edges == []
    assert {unit.source_entity_type for unit in hold_only.units} == {"libby_hold"}
    assert hold_only.edges == []
