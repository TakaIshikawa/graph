import csv
from io import StringIO

from graph.export import export_units_to_markdown_image_alt_csv


def _rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_markdown_image_alt_csv_exports_images_outside_fences():
    content = "\n".join(["![Diagram](https://img.test/a.png)", "```", "![Ignored](ignored.png)", "```", "![Logo](logo.svg \"Logo\")"])

    rows = _rows(export_units_to_markdown_image_alt_csv([{"id": "u1", "title": "Unit One", "content": content}]))

    assert rows == [
        {"unit_id": "u1", "title": "Unit One", "image_url": "https://img.test/a.png", "alt_text": "Diagram", "has_alt_text": "True", "image_count": "2"},
        {"unit_id": "u1", "title": "Unit One", "image_url": "logo.svg", "alt_text": "Logo", "has_alt_text": "True", "image_count": "2"},
    ]


def test_markdown_image_alt_csv_includes_empty_alt_images_and_path_stats(tmp_path):
    path = tmp_path / "image-alt.csv"
    units = [{"id": "u", "content": "![](missing.png)\n![Chart](chart.png)"}]

    expected = export_units_to_markdown_image_alt_csv(units)
    stats = export_units_to_markdown_image_alt_csv(units, path)

    assert _rows(expected) == [
        {"unit_id": "u", "title": "", "image_url": "chart.png", "alt_text": "Chart", "has_alt_text": "True", "image_count": "2"},
        {"unit_id": "u", "title": "", "image_url": "missing.png", "alt_text": "", "has_alt_text": "False", "image_count": "2"},
    ]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 2
    assert stats["bytes_written"] == path.stat().st_size
