from __future__ import annotations

import os
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.webvtt import WebVttAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def _edge_title_pairs(result) -> list[tuple[str, str]]:
    titles = {unit.source_id: unit.title for unit in result.units}
    return [(titles[edge.from_unit_id], titles[edge.to_unit_id]) for edge in result.edges]


def test_webvtt_ingests_parent_and_cues_with_contains_edges(tmp_path):
    path = tmp_path / "meeting.vtt"
    path.write_text(
        "WEBVTT\n"
        "Kind: captions\n"
        "\n"
        "NOTE intro note ignored\n"
        "This should not become content.\n"
        "\n"
        "intro-cue\n"
        "00:00:01.000 --> 00:00:03.500 align:start\n"
        "<v Alice>Welcome to the meeting.\n"
        "Today we discuss ingestion.</v>\n"
        "\n"
        "00:00:04.000 --> 00:00:05.250\n"
        "Bob: Ship the WebVTT adapter.\n",
        encoding="utf-8",
    )

    result = WebVttAdapter(path=str(path)).ingest()

    assert [unit.source_entity_type for unit in result.units] == [
        "webvtt_transcript",
        "webvtt_cue",
        "webvtt_cue",
    ]
    parent, first, second = result.units
    assert parent.source_project == SourceProject.WEBVTT
    assert parent.source_id == "webvtt:meeting.vtt"
    assert parent.title == "meeting"
    assert parent.content_type == ContentType.ARTIFACT
    assert parent.content == (
        "Welcome to the meeting.\n"
        "Today we discuss ingestion.\n\n"
        "Ship the WebVTT adapter."
    )
    assert parent.metadata["source_path"] == "meeting.vtt"
    assert parent.metadata["cue_count"] == 2
    assert parent.metadata["first_timestamp"] == "00:00:01.000"
    assert parent.metadata["last_timestamp"] == "00:00:05.250"

    assert first.content == "Welcome to the meeting.\nToday we discuss ingestion."
    assert first.content_type == ContentType.INSIGHT
    assert first.metadata["start"] == "00:00:01.000"
    assert first.metadata["end"] == "00:00:03.500"
    assert first.metadata["cue_index"] == 1
    assert first.metadata["cue_id"] == "intro-cue"
    assert first.metadata["speaker"] == "Alice"
    assert first.metadata["source_path"] == "meeting.vtt"

    assert second.content == "Ship the WebVTT adapter."
    assert second.metadata["speaker"] == "Bob"
    assert "cue_id" not in second.metadata

    assert _edge_title_pairs(result) == [
        ("meeting", "meeting 00:00:01.000 Alice"),
        ("meeting", "meeting 00:00:04.000 Bob"),
    ]
    assert [edge.to_unit_id for edge in result.edges] == [first.source_id, second.source_id]
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in result.edges)
    assert all(edge.source == EdgeSource.SOURCE for edge in result.edges)
    assert result.edges[0].metadata["relation_type"] == "webvtt_contains"
    assert result.edges[0].metadata["cue_index"] == 1


def test_webvtt_recurses_directories_and_skips_note_only_files(tmp_path):
    nested = tmp_path / "talks"
    nested.mkdir()
    (tmp_path / "root.vtt").write_text(
        "WEBVTT\n\n00:01.000 --> 00:02.000\nRoot cue.\n",
        encoding="utf-8",
    )
    (nested / "talk.vtt").write_text(
        "WEBVTT\n\ncue-a\n00:00:10.000 --> 00:00:12.000\nNested cue.\n",
        encoding="utf-8",
    )
    (nested / "notes.vtt").write_text("WEBVTT\n\nNOTE only\nignored\n", encoding="utf-8")

    result = WebVttAdapter(path=str(tmp_path)).ingest()

    transcript_source_ids = [
        unit.source_id
        for unit in result.units
        if unit.source_entity_type == "webvtt_transcript"
    ]
    assert transcript_source_ids == [
        "webvtt:root.vtt",
        "webvtt:talks/talk.vtt",
    ]
    assert [unit.content for unit in result.units if unit.source_entity_type == "webvtt_cue"] == [
        "Root cue.",
        "Nested cue.",
    ]


def test_webvtt_filters_entity_types_and_since(tmp_path):
    old_path = tmp_path / "old.vtt"
    new_path = tmp_path / "new.vtt"
    old_path.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nOld.\n", encoding="utf-8")
    new_path.write_text("WEBVTT\n\n00:00:03.000 --> 00:00:04.000\nNew.\n", encoding="utf-8")
    os.utime(old_path, (1_700_000_000, 1_700_000_000))
    os.utime(new_path, (1_700_010_000, 1_700_010_000))

    skipped = WebVttAdapter(path=str(tmp_path)).ingest(entity_types=["markdown_note"])
    assert skipped.units == []
    assert skipped.edges == []

    cues_only = WebVttAdapter(path=str(new_path)).ingest(entity_types=["webvtt_cue"])
    assert [unit.source_entity_type for unit in cues_only.units] == ["webvtt_cue"]
    assert cues_only.edges == []

    since = datetime.fromtimestamp(1_700_005_000, tz=timezone.utc)
    result = WebVttAdapter(path=str(tmp_path)).ingest(
        since=SyncState(
            source_project="webvtt",
            source_entity_type="webvtt_transcript",
            last_sync_at=since,
        )
    )
    cue_contents = [
        unit.content for unit in result.units if unit.source_entity_type == "webvtt_cue"
    ]
    assert cue_contents == ["New."]


def test_webvtt_adapter_is_registered():
    assert "webvtt" in list_adapters()
    adapter = get_adapter("webvtt", path="/tmp/transcript.vtt")
    assert isinstance(adapter, WebVttAdapter)
    assert adapter.name == "webvtt"
