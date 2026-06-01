from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_embed_inventory_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_embed_inventory_csv_parses_obsidian_embeds_outside_fences():
    text = export_units_to_markdown_embed_inventory_csv(
        [{"id": "u", "title": "Doc", "content": "![[image.png|Alt]] and [[plain]]\n![[Note#Section]]\n```md\n![[ignored.png]]\n```"}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Doc", "line_number": "1", "target": "image.png", "fragment": "", "alias": "Alt", "is_image_embed": "true"},
        {"unit_id": "u", "title": "Doc", "line_number": "2", "target": "Note", "fragment": "Section", "alias": "", "is_image_embed": "false"},
    ]


def test_markdown_embed_inventory_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "embeds.csv"
    units = [{"id": "u", "content": "![[target]]"}]

    expected = export_units_to_markdown_embed_inventory_csv(units)
    stats = export_units_to_markdown_embed_inventory_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
