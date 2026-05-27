from graph.store import summarize_unit_audio_timestamp_references


def test_audio_references_group_by_normalized_extension_and_timestamp():
    report = summarize_unit_audio_timestamp_references(
        [
            {"id": "b", "content": "clip at 01:23 https://example.com/talk.MP3"},
            {"id": "a", "content": "local/audio.wav no cue"},
            {"id": "c", "metadata": {"audio_url": "episodes/show.m4a", "source_url": "archive/tape.flac?t=1"}},
        ]
    )

    assert report["extension_counts"] == {"flac": 1, "m4a": 1, "mp3": 1, "wav": 1}
    assert report["timestamped_reference_count"] == 1
    assert report["untimestamped_reference_count"] == 3
    assert report["total_references"] == 4


def test_audio_metadata_aliases_and_hour_timestamp_are_supported():
    report = summarize_unit_audio_timestamp_references(
        [
            {"id": "u1", "media_url": "podcast.mp3", "metadata": {"path": "raw/interview.wav 1:02:03"}},
            {"id": "u2", "source_url": "https://cdn.example/audio.m4a"},
        ]
    )

    assert report["extension_counts"]["mp3"] == 1
    assert report["extension_counts"]["wav"] == 1
    assert report["extension_counts"]["m4a"] == 1
    assert report["timestamped_reference_count"] == 1
    assert report["untimestamped_reference_count"] == 2
