from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_code_fence_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_code_fence_csv_exports_backtick_and_tilde_fences():
    text = export_units_to_markdown_code_fence_csv(
        [
            {"id": "b", "title": "Beta", "content": "```python linenums\nprint(1)\n```\n~~~ sql\nselect 1\n~~~"},
            {"id": "a", "title": "Alpha", "content": "plain"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "b", "title": "Beta", "start_line": "1", "end_line": "3", "fence_char": "`", "language": "python", "info_string": "python linenums", "line_count": "1"},
        {"unit_id": "b", "title": "Beta", "start_line": "4", "end_line": "6", "fence_char": "~", "language": "sql", "info_string": "sql", "line_count": "1"},
    ]


def test_markdown_code_fence_csv_tolerates_unterminated_fences_and_path_mode(tmp_path):
    path = tmp_path / "fences.csv"
    units = [{"id": "a", "content": "```js\nconsole.log(1)"}]

    stats = export_units_to_markdown_code_fence_csv(units, path)

    assert rows(path.read_text(encoding="utf-8"))[0]["end_line"] == "2"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
