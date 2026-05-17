from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.letterboxd import LetterboxdAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_letterboxd_csv_ingests_films_with_full_metadata(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date,Review",
                '2025-01-15,Inception,2010,https://letterboxd.com/film/inception/,4.5,No,"sci-fi,thriller",2025-01-15,"Mind-bending masterpiece!"',
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.LETTERBOXD
    assert unit.source_id == "letterboxd:inception"
    assert unit.source_entity_type == "film"
    assert unit.title == "Inception (2010)"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Film: Inception (2010)" in unit.content
    assert "Rating: 4.5" in unit.content
    assert "Tags: sci-fi, thriller" in unit.content
    assert "Mind-bending masterpiece!" in unit.content
    assert unit.metadata == {
        "name": "Inception",
        "year": "2010",
        "letterboxd_uri": "https://letterboxd.com/film/inception/",
        "rating": "4.5",
        "rewatch": False,
        "tags": ["sci-fi", "thriller"],
        "watched_date": "2025-01-15T00:00:00+00:00",
        "diary_date": "",
        "review_url": "",
        "contains_spoilers": None,
        "review": "Mind-bending masterpiece!",
    }
    assert unit.tags == ["sci-fi", "thriller"]
    assert unit.created_at == datetime(2025, 1, 15, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 15, tzinfo=timezone.utc)


def test_letterboxd_csv_handles_missing_optional_fields(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Watched Date",
                "Minimal Film,2020,2025-01-20",
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Minimal Film (2020)"
    assert unit.metadata["name"] == "Minimal Film"
    assert unit.metadata["year"] == "2020"
    assert unit.metadata["letterboxd_uri"] == ""
    assert unit.metadata["rating"] == ""
    assert unit.metadata["rewatch"] is None
    assert unit.metadata["review"] == ""
    assert unit.metadata["tags"] == []
    assert unit.tags == []
    assert unit.created_at == datetime(2025, 1, 20, tzinfo=timezone.utc)


def test_letterboxd_generates_hash_source_id_when_no_uri(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year",
                "Unknown Film,2021",
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id.startswith("letterboxd:")
    assert len(unit.source_id) == 35  # "letterboxd:" + 24 character hash


def test_letterboxd_json_ingests_films_from_list_or_dict_exports(tmp_path):
    export = tmp_path / "letterboxd.json"
    export.write_text(
        json.dumps(
            {
                "diary": [
                    {
                        "name": "The Matrix",
                        "year": "1999",
                        "letterboxd_uri": "https://letterboxd.com/film/the-matrix/",
                        "rating": "5",
                        "watched_date": "2025-01-25T20:00:00Z",
                        "tags": "action,sci-fi",
                        "review": "Revolutionary film!",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = LetterboxdAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "letterboxd:the-matrix"
    assert unit.title == "The Matrix (1999)"
    assert unit.metadata["name"] == "The Matrix"
    assert unit.metadata["year"] == "1999"
    assert unit.metadata["rating"] == "5"
    assert unit.metadata["review"] == "Revolutionary film!"
    assert unit.metadata["tags"] == ["action", "sci-fi"]
    assert unit.tags == ["action", "sci-fi"]
    assert unit.created_at == datetime(2025, 1, 25, 20, 0, tzinfo=timezone.utc)


def test_letterboxd_since_filters_films_not_newer_than_sync_state(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Watched Date",
                "Old Film,2020,2025-01-01",
                "Equal Film,2021,2025-01-02",
                "New Film,2022,2025-01-03",
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="letterboxd",
        source_entity_type="film",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = LetterboxdAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert "New Film" in result.units[0].title


def test_letterboxd_respects_entity_types(tmp_path):
    export = tmp_path / "letterboxd.json"
    export.write_text(
        json.dumps([{"name": "Test Film", "year": "2020"}]), encoding="utf-8"
    )

    result = LetterboxdAdapter(path=str(export)).ingest(entity_types=["book"])

    assert result.units == []
    assert result.edges == []


def test_letterboxd_skips_films_without_name(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Watched Date",
                ",,2025-01-01",  # Empty name
                "Valid Film,2020,2025-01-02",
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["name"] == "Valid Film"


def test_letterboxd_handles_malformed_csv(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text("This is not a valid CSV file\n<<>><>", encoding="utf-8")

    result = LetterboxdAdapter(path=str(export)).ingest()

    # Should return empty result without crashing
    assert result.units == []


def test_letterboxd_handles_empty_export(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Watched Date",
                # No data rows
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    assert result.units == []


def test_letterboxd_handles_nonexistent_file():
    result = LetterboxdAdapter(path="/nonexistent/path.csv").ingest()

    assert result.units == []


def test_letterboxd_parses_tags_correctly(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Tags,Watched Date",
                "Film 1,2020,\"action,thriller,drama\",2025-01-01",
                "Film 2,2021,comedy,2025-01-02",
                "Film 3,2022,,2025-01-03",  # Empty tags
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    assert len(result.units) == 3
    assert result.units[0].tags == ["action", "thriller", "drama"]
    assert result.units[1].tags == ["comedy"]
    assert result.units[2].tags == []


def test_letterboxd_handles_rewatches(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Rewatch,Watched Date",
                "Film,2020,Yes,2025-01-15",
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.metadata["rewatch"] is True


def test_letterboxd_preserves_watch_review_links_spoilers_and_tags(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Rewatch,Tags,Watched Date,Diary Date,Review URL,Letterboxd URI,Contains Spoilers",
                "Film,2020,Yes,\"favorite; drama | favorite\",2026-05-01,2026-05-02,"
                "https://letterboxd.com/user/film/review/,https://letterboxd.com/film/film/,true",
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.source_id == "letterboxd:film"
    assert unit.metadata["rewatch"] is True
    assert unit.metadata["tags"] == ["favorite", "drama"]
    assert unit.metadata["watched_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["diary_date"] == "2026-05-02T00:00:00+00:00"
    assert unit.metadata["review_url"] == "https://letterboxd.com/user/film/review/"
    assert unit.metadata["letterboxd_uri"] == "https://letterboxd.com/film/film/"
    assert unit.metadata["contains_spoilers"] is True
    assert unit.tags == ["favorite", "drama"]


def test_letterboxd_ingests_director_entities(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Director,Watched Date",
                "Film,2020,Ada Lovelace,2025-01-15",
                "Other Film,2021,\"Ada Lovelace; Grace Hopper\",2025-01-16",
                "No Director,2022,,2025-01-17",
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    directors = [unit for unit in result.units if unit.source_entity_type == "director"]
    assert sorted(unit.title for unit in directors) == ["Ada Lovelace", "Grace Hopper"]
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "film_director"]) == 3


def test_letterboxd_extracts_slug_from_uri(tmp_path):
    export = tmp_path / "letterboxd.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Year,Letterboxd URI",
                "Test Film,2020,https://letterboxd.com/film/test-film-2020/",
            ]
        ),
        encoding="utf-8",
    )

    result = LetterboxdAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.source_id == "letterboxd:test-film-2020"


def test_letterboxd_adapter_is_registered():
    assert "letterboxd" in list_adapters()
    adapter = get_adapter("letterboxd", path="/tmp/letterboxd.csv")
    assert adapter.name == "letterboxd"
