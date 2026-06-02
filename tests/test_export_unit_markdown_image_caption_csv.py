from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_image_caption_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_image_followed_by_italic_caption():
    result = rows(export_units_to_markdown_image_caption_csv([{"id": "u", "title": "T", "content": "![Alt](img/chart.png)\n*Quarterly trend*"}]))

    assert result == [
        {
            "unit_id": "u",
            "title": "T",
            "image_target": "img/chart.png",
            "line_number": "2",
            "caption": "Quarterly trend",
            "caption_style": "italic",
        }
    ]


def test_exports_figure_caption_lines_adjacent_to_images():
    result = rows(
        export_units_to_markdown_image_caption_csv(
            [{"id": "u", "content": "Figure: System overview\n![Diagram](diagram.svg)\n\n![Photo](photo.jpg)\nFigure: Launch day"}]
        )
    )

    assert [(row["image_target"], row["line_number"], row["caption"], row["caption_style"]) for row in result] == [
        ("diagram.svg", "1", "System overview", "figure"),
        ("photo.jpg", "5", "Launch day", "figure"),
    ]


def test_extracts_simple_html_figure_caption():
    result = rows(
        export_units_to_markdown_image_caption_csv(
            [{"id": "u", "content": '<figure>\n<img src="plot.png" alt="Plot">\n<figcaption>Forecast <strong>range</strong></figcaption>\n</figure>'}]
        )
    )

    assert [(row["image_target"], row["line_number"], row["caption"], row["caption_style"]) for row in result] == [
        ("plot.png", "1", "Forecast range", "html_figcaption")
    ]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "captions.csv"

    result = export_units_to_markdown_image_caption_csv([{"id": "u", "content": "![Alt](image.png)\n_Caption_"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
