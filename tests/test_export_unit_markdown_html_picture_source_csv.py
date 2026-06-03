from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_picture_source_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_picture_source_csv_exports_sources_and_fallback_img():
    text = export_unit_markdown_html_picture_source_csv(
        [
            {
                "id": "u",
                "title": "Responsive",
                "content": """
<picture>
  <source srcset="hero.avif 1x, hero@2x.avif 2x" media="(min-width: 900px)" type="image/avif" sizes="50vw">
  <source srcset='hero.webp' type='image/webp'>
  <img src="hero.jpg" alt="Hero image" srcset="hero.jpg 1x, hero@2x.jpg 2x" sizes="100vw">
</picture>
""".strip(),
            }
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "Responsive",
            "line_number": "1",
            "tag": "img",
            "src": "hero.jpg",
            "srcset": "hero.jpg 1x, hero@2x.jpg 2x",
            "media": "",
            "type": "",
            "sizes": "100vw",
            "alt": "Hero image",
            "raw_html": '<img src="hero.jpg" alt="Hero image" srcset="hero.jpg 1x, hero@2x.jpg 2x" sizes="100vw">',
        },
        {
            "unit_id": "u",
            "title": "Responsive",
            "line_number": "1",
            "tag": "source",
            "src": "",
            "srcset": "hero.avif 1x, hero@2x.avif 2x",
            "media": "(min-width: 900px)",
            "type": "image/avif",
            "sizes": "50vw",
            "alt": "",
            "raw_html": '<source srcset="hero.avif 1x, hero@2x.avif 2x" media="(min-width: 900px)" type="image/avif" sizes="50vw">',
        },
        {
            "unit_id": "u",
            "title": "Responsive",
            "line_number": "1",
            "tag": "source",
            "src": "",
            "srcset": "hero.webp",
            "media": "",
            "type": "image/webp",
            "sizes": "",
            "alt": "",
            "raw_html": "<source srcset='hero.webp' type='image/webp'>",
        },
    ]


def test_picture_source_csv_ignores_fenced_examples_and_sorts_units():
    text = export_unit_markdown_html_picture_source_csv(
        [
            {"id": "b", "content": "```html\n<picture><source srcset=\"skip.webp\"></picture>\n```\n<picture><img src=b.jpg alt=B></picture>"},
            {"id": "a", "metadata": {"title": "Meta"}, "content": "<picture><source srcset=a.webp media=(print)></picture>"},
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "a",
            "title": "Meta",
            "line_number": "1",
            "tag": "source",
            "src": "",
            "srcset": "a.webp",
            "media": "(print)",
            "type": "",
            "sizes": "",
            "alt": "",
            "raw_html": "<source srcset=a.webp media=(print)>",
        },
        {
            "unit_id": "b",
            "title": "",
            "line_number": "4",
            "tag": "img",
            "src": "b.jpg",
            "srcset": "",
            "media": "",
            "type": "",
            "sizes": "",
            "alt": "B",
            "raw_html": "<img src=b.jpg alt=B>",
        },
    ]


def test_picture_source_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "pictures.csv"
    units = [{"id": "u", "content": '<picture><img src="hero.jpg"></picture>'}]

    expected = export_unit_markdown_html_picture_source_csv(units)
    stats = export_unit_markdown_html_picture_source_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
