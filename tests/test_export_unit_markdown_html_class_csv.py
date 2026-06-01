import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_class_csv


def _rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_markdown_html_class_csv_splits_class_attributes_and_skips_fences():
    content = "\n".join(
        [
            '<div class="alpha beta">Block</div>',
            "Inline <span class='gamma'>text</span>",
            "<mark class=delta>hit</mark>",
            "```",
            '<p class="ignored">No</p>',
            "```",
        ]
    )

    rows = _rows(export_units_to_markdown_html_class_csv([{"id": "u1", "title": "Unit One", "content": content}]))

    assert rows == [
        {"unit_id": "u1", "title": "Unit One", "line_number": "1", "tag": "div", "class_name": "alpha", "class_count": "2"},
        {"unit_id": "u1", "title": "Unit One", "line_number": "1", "tag": "div", "class_name": "beta", "class_count": "2"},
        {"unit_id": "u1", "title": "Unit One", "line_number": "2", "tag": "span", "class_name": "gamma", "class_count": "1"},
        {"unit_id": "u1", "title": "Unit One", "line_number": "3", "tag": "mark", "class_name": "delta", "class_count": "1"},
    ]


def test_markdown_html_class_csv_path_mode_reports_write_metadata(tmp_path):
    path = tmp_path / "html_classes.csv"
    units = [{"id": "u", "content": '<section class="note">Body</section>'}]

    expected = export_units_to_markdown_html_class_csv(units)
    stats = export_units_to_markdown_html_class_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
