from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_meta_tag_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_meta_tag_csv_exports_name_property_and_http_equiv_variants():
    text = export_unit_markdown_html_meta_tag_csv(
        [{"id": "u", "title": "T", "source": "s", "content": '<meta content="width=device-width" name=viewport>\n<meta property="og:title" content=\'Hello\'>\n<meta http-equiv="refresh" content="5; url=/next">'}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "name": "viewport", "property": "", "http_equiv": "", "content_value": "width=device-width"},
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "2", "name": "", "property": "og:title", "http_equiv": "", "content_value": "Hello"},
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "3", "name": "", "property": "", "http_equiv": "refresh", "content_value": "5; url=/next"},
    ]


def test_meta_tag_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "meta.csv"
    units = [{"id": "u", "content": '<meta name="description" content="Keep unknown-safe content">'}]

    expected = export_unit_markdown_html_meta_tag_csv(units)
    stats = export_unit_markdown_html_meta_tag_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
