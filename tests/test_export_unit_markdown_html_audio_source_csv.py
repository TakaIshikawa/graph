from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_audio_source_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_audio_source_csv_exports_direct_and_nested_sources():
    text = export_units_to_markdown_html_audio_source_csv(
        [
            {
                "id": "u",
                "title": "Audio",
                "metadata": {"source_path": "notes/audio.md", "source": "archive"},
                "content": '<audio src="intro.mp3" controls preload="metadata">Fallback text</audio>\n'
                '<audio autoplay loop><source src="clip.ogg" type="audio/ogg">No audio support.</audio>',
            }
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "Audio",
            "source_path": "notes/audio.md",
            "source": "archive",
            "line_number": "1",
            "audio_src": "intro.mp3",
            "source_src": "",
            "type": "",
            "controls": "true",
            "autoplay": "false",
            "loop": "false",
            "preload": "metadata",
            "fallback_text": "Fallback text",
        },
        {
            "unit_id": "u",
            "title": "Audio",
            "source_path": "notes/audio.md",
            "source": "archive",
            "line_number": "2",
            "audio_src": "",
            "source_src": "clip.ogg",
            "type": "audio/ogg",
            "controls": "false",
            "autoplay": "true",
            "loop": "true",
            "preload": "",
            "fallback_text": "No audio support.",
        },
    ]


def test_audio_source_csv_ignores_fenced_examples_and_writes_path(tmp_path):
    units = [{"id": "u", "content": '```html\n<audio src="skip.mp3"></audio>\n```\n<audio><source src=ok.mp3></audio>'}]
    expected = export_units_to_markdown_html_audio_source_csv(units)
    path = tmp_path / "audio.csv"

    stats = export_units_to_markdown_html_audio_source_csv(units, path)

    assert [row["source_src"] for row in _rows(expected)] == ["ok.mp3"]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
