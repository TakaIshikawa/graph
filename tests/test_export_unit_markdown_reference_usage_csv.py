from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_reference_usage_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_markdown_reference_usage_csv_reports_all_usage_forms_with_lines():
    rows = _rows(
        export_unit_markdown_reference_usage_csv(
            [
                {
                    "id": "u1",
                    "title": "Refs",
                    "content": "\n".join(
                        [
                            "See [Docs][docs] and [Guide][].",
                            "[shortcut] appears here.",
                            "[docs]: https://example.com",
                            "Inline [skip](https://example.com) and ![image][img].",
                        ]
                    ),
                }
            ]
        )
    )

    assert rows == [
        {"unit_id": "u1", "title": "Refs", "label": "docs", "link_text": "Docs", "usage_type": "full", "line_number": "1"},
        {"unit_id": "u1", "title": "Refs", "label": "Guide", "link_text": "Guide", "usage_type": "collapsed", "line_number": "1"},
        {"unit_id": "u1", "title": "Refs", "label": "shortcut", "link_text": "shortcut", "usage_type": "shortcut", "line_number": "2"},
    ]


def test_export_unit_markdown_reference_usage_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reference-usages.csv"
    units = [{"id": "u1", "content": "[Docs][docs]\n[docs]: https://example.com"}]
    expected = export_unit_markdown_reference_usage_csv(units)

    stats = export_unit_markdown_reference_usage_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "rows_exported": 1, "bytes_written": len(expected.encode("utf-8"))}
