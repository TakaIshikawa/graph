from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_link_text_quality_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_link_text_quality_csv_detects_weak_labels_case_insensitively():
    rows = _rows(
        export_units_to_link_text_quality_csv(
            [
                {
                    "id": "u1",
                    "content": "\n".join(
                        [
                            "[Here](https://example.com/a)",
                            "[THIS](https://example.com/b)",
                            "[link](https://example.com/c)",
                            "[Click   Here](https://example.com/d)",
                        ]
                    ),
                }
            ]
        )
    )

    assert [(row["link_text"], row["issue"], row["line_number"]) for row in rows] == [
        ("Here", "weak_label", "1"),
        ("THIS", "weak_label", "2"),
        ("link", "weak_label", "3"),
        ("Click Here", "weak_label", "4"),
    ]


def test_export_units_to_link_text_quality_csv_flags_bare_url_text_separately_from_weak_labels():
    rows = _rows(
        export_units_to_link_text_quality_csv(
            [{"id": "u1", "content": "[https://Example.com/a](https://example.com/b)\n[](https://example.com/c)"}]
        )
    )

    assert [(row["link_text"], row["target"], row["issue"]) for row in rows] == [
        ("https://Example.com/a", "https://example.com/b", "bare_url_text"),
        ("", "https://example.com/c", "empty_text"),
    ]


def test_export_units_to_link_text_quality_csv_flags_repeated_text_with_different_destinations_within_unit():
    rows = _rows(
        export_units_to_link_text_quality_csv(
            [
                {
                    "id": "u1",
                    "content": "[Docs](https://example.com/a)\n[docs](https://example.com/b)\n[Docs](https://example.com/a)",
                },
                {"id": "u2", "content": "[Docs](https://example.com/a)"},
            ]
        )
    )

    assert [(row["unit_id"], row["link_text"], row["target"], row["issue"], row["line_number"]) for row in rows] == [
        ("u1", "Docs", "https://example.com/a", "repeated_text_different_target", "1"),
        ("u1", "docs", "https://example.com/b", "repeated_text_different_target", "2"),
        ("u1", "Docs", "https://example.com/a", "repeated_text_different_target", "3"),
    ]


def test_export_units_to_link_text_quality_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "link-text-quality.csv"
    units = [{"id": "u1", "content": "[here](https://example.com)"}]
    expected = export_units_to_link_text_quality_csv(units)

    stats = export_units_to_link_text_quality_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": len(expected.encode("utf-8")),
    }
