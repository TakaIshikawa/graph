from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_details_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_details_csv_exports_closed_open_and_multiline_summary():
    text = export_unit_markdown_html_details_csv(
        [
            {
                "id": "u",
                "title": "Doc",
                "source": "vault",
                "content": "<details><summary>Closed</summary><p>Body</p></details>\n<details open>\n<summary>\nOpen note\n</summary>\n</details>",
            }
        ]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Doc", "source": "vault", "line_number": "1", "is_open": "False", "summary_text": "Closed"},
        {"unit_id": "u", "title": "Doc", "source": "vault", "line_number": "2", "is_open": "True", "summary_text": "Open note"},
    ]


def test_details_csv_sorts_by_unit_id_and_line_number():
    text = export_unit_markdown_html_details_csv(
        [
            {"id": "b", "content": "<details><summary>B</summary></details>"},
            {"id": "a", "content": "x\n<details open><summary>A2</summary></details>\n<details><summary>A3</summary></details>"},
        ]
    )

    assert [(row["unit_id"], row["line_number"], row["summary_text"]) for row in _rows(text)] == [("a", "2", "A2"), ("a", "3", "A3"), ("b", "1", "B")]


def test_details_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "details.csv"
    units = [{"unit_id": "u1", "content": "<details><summary>One</summary></details>"}]

    expected = export_unit_markdown_html_details_csv(units)
    stats = export_unit_markdown_html_details_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
