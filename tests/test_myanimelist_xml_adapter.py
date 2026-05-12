from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.myanimelist_xml import MyAnimeListXmlAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_myanimelist_xml_ingests_anime_entries(tmp_path):
    path = tmp_path / "anime.xml"
    path.write_text(
        """<?xml version="1.0"?>
<myanimelist>
  <anime>
    <series_animedb_id>5114</series_animedb_id>
    <series_title>Fullmetal Alchemist: Brotherhood</series_title>
    <series_type>TV</series_type>
    <series_episodes>64</series_episodes>
    <my_watched_episodes>64</my_watched_episodes>
    <my_score>10</my_score>
    <my_status>Completed</my_status>
    <my_start_date>2026-01-01</my_start_date>
    <my_finish_date>2026-02-01</my_finish_date>
    <my_tags>classic, action</my_tags>
    <my_storage>NAS</my_storage>
    <my_rewatching>0</my_rewatching>
    <my_times_watched>1</my_times_watched>
  </anime>
</myanimelist>
""",
        encoding="utf-8",
    )

    result = MyAnimeListXmlAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MYANIMELIST_XML
    assert unit.source_id == "myanimelist_xml:5114"
    assert unit.metadata["watched_episodes"] == 64
    assert unit.metadata["total_episodes"] == 64
    assert unit.metadata["score"] == 10
    assert unit.metadata["tags"] == ["classic", "action"]
    assert unit.metadata["storage"] == "NAS"
    assert unit.updated_at == datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert "Episodes: 64/64" in unit.content


def test_myanimelist_xml_directory_since_and_malformed_file(tmp_path):
    (tmp_path / "bad.xml").write_text("<myanimelist><anime>", encoding="utf-8")
    (tmp_path / "old.xml").write_text("<myanimelist><anime><series_title>Old</series_title><my_finish_date>2026-01-01</my_finish_date></anime></myanimelist>", encoding="utf-8")
    (tmp_path / "new.xml").write_text("<myanimelist><anime><series_title>New</series_title><my_finish_date>2026-03-01</my_finish_date></anime></myanimelist>", encoding="utf-8")
    since = SyncState(source_project="myanimelist_xml", source_entity_type="anime", last_sync_at=datetime(2026, 2, 1, tzinfo=timezone.utc))

    result = MyAnimeListXmlAdapter(path=str(tmp_path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New"]
    assert get_adapter("myanimelist_xml", path=str(tmp_path)).name == "myanimelist_xml"


def test_myanimelist_xml_filters_entity_type_and_hashes_without_mal_id(tmp_path):
    path = tmp_path / "anime.xml"
    path.write_text("<myanimelist><anime><series_title>Hash Me</series_title><my_status>Watching</my_status></anime></myanimelist>", encoding="utf-8")

    units = MyAnimeListXmlAdapter(path=str(path)).ingest().units

    assert len(units) == 1
    assert units[0].source_id.startswith("myanimelist_xml:")
    assert MyAnimeListXmlAdapter(path=str(path)).ingest(entity_types=["manga"]).units == []
