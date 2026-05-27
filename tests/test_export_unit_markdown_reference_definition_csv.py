from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_reference_definition_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_markdown_reference_definition_csv_parses_destinations_titles_and_lines():
    rows = _rows(
        export_unit_markdown_reference_definition_csv(
            [
                {
                    "id": "u1",
                    "title": "Reference Note",
                    "content": "\n".join(
                        [
                            "Intro [docs][d].",
                            '[d]: https://example.com/docs "Docs title"',
                            "  [guide]: <https://example.com/guide> 'Guide title'",
                            "[plain]: ./local.md",
                        ]
                    ),
                }
            ]
        )
    )

    assert rows == [
        {
            "unit_id": "u1",
            "title": "Reference Note",
            "label": "d",
            "destination": "https://example.com/docs",
            "link_title": "Docs title",
            "line_number": "2",
            "duplicate_label": "false",
        },
        {
            "unit_id": "u1",
            "title": "Reference Note",
            "label": "guide",
            "destination": "https://example.com/guide",
            "link_title": "Guide title",
            "line_number": "3",
            "duplicate_label": "false",
        },
        {
            "unit_id": "u1",
            "title": "Reference Note",
            "label": "plain",
            "destination": "./local.md",
            "link_title": "",
            "line_number": "4",
            "duplicate_label": "false",
        },
    ]


def test_export_unit_markdown_reference_definition_csv_flags_duplicate_labels_within_unit():
    rows = _rows(
        export_unit_markdown_reference_definition_csv(
            [
                {"id": "u1", "content": "[Docs]: https://example.com/a\n[docs]: https://example.com/b"},
                {"id": "u2", "content": "[Docs]: https://example.com/c"},
            ]
        )
    )

    assert [(row["unit_id"], row["label"], row["duplicate_label"]) for row in rows] == [
        ("u1", "Docs", "true"),
        ("u1", "docs", "true"),
        ("u2", "Docs", "false"),
    ]


def test_export_unit_markdown_reference_definition_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reference-definitions.csv"
    units = [{"id": "u1", "content": "[docs]: https://example.com"}]
    expected = export_unit_markdown_reference_definition_csv(units)

    stats = export_unit_markdown_reference_definition_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": len(expected.encode("utf-8")),
    }
