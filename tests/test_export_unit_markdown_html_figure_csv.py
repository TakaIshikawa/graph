from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_figure_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_figure_csv_exports_inline_caption_and_image_source():
    text = export_unit_markdown_html_figure_csv([{"id": "u", "title": "T", "source": "s", "content": '<figure><img src="a.png"><figcaption>Alpha</figcaption></figure>'}])

    assert _rows(text) == [{"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "has_caption": "True", "caption_text": "Alpha", "image_sources": "a.png"}]


def test_figure_csv_exports_multiline_multiple_images_and_missing_caption():
    text = export_unit_markdown_html_figure_csv(
        [{"id": "u", "content": '<figure>\n<img src="a.png">\n<img alt=x src=b.jpg>\n</figure>\n<figure><img src="c.webp"></figure>'}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "", "source": "", "line_number": "1", "has_caption": "False", "caption_text": "", "image_sources": "a.png; b.jpg"},
        {"unit_id": "u", "title": "", "source": "", "line_number": "5", "has_caption": "False", "caption_text": "", "image_sources": "c.webp"},
    ]


def test_figure_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "figures.csv"
    units = [{"id": "u", "content": "<figure><figcaption>Only caption</figcaption></figure>"}]

    expected = export_unit_markdown_html_figure_csv(units)
    stats = export_unit_markdown_html_figure_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
