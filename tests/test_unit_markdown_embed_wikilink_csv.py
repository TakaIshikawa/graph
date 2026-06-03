from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_embed_wikilink_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_embed_wikilink_csv_exports_image_note_pdf_alias_and_positions():
    text = export_units_to_markdown_embed_wikilink_csv(
        [
            {
                "id": "u",
                "content": "See ![[image.png]] and [[Plain]]\nEmbed ![[Note#Section]]\nPDF ![[file.pdf|Alias]]",
            }
        ]
    )

    assert _rows(text) == [
        {"unit_id": "u", "target": "image.png", "alias": "", "heading": "", "media_type_hint": "image", "line_number": "1", "column_number": "5"},
        {"unit_id": "u", "target": "Note", "alias": "", "heading": "Section", "media_type_hint": "", "line_number": "2", "column_number": "7"},
        {"unit_id": "u", "target": "file.pdf", "alias": "Alias", "heading": "", "media_type_hint": "pdf", "line_number": "3", "column_number": "5"},
    ]


def test_embed_wikilink_csv_ignores_plain_escaped_malformed_and_fenced_links():
    text = export_units_to_markdown_embed_wikilink_csv(
        [
            {
                "id": "u",
                "content": r"[[Plain]] \![[Escaped]] ![[ ]] ![[Bad|Alias|Nope]]" "\n```md\n![[hidden.png]]\n```\n![[ok.svg|Icon]]",
            }
        ]
    )

    assert _rows(text) == [
        {"unit_id": "u", "target": "ok.svg", "alias": "Icon", "heading": "", "media_type_hint": "image", "line_number": "5", "column_number": "1"}
    ]


def test_embed_wikilink_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "embeds.csv"
    units = [{"unit_id": "u1", "content": "![[clip.mp4]]"}]

    expected = export_units_to_markdown_embed_wikilink_csv(units)
    stats = export_units_to_markdown_embed_wikilink_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
