import csv
from io import StringIO

from graph.export import export_unit_markdown_code_fence_meta_to_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_code_fence_meta_csv_exports_backtick_and_tilde_metadata():
    content = "```python linenos title='Demo'\nprint(1)\n```\n~~~js\nx\n~~~\n~~~go meta\n"

    result = rows(export_unit_markdown_code_fence_meta_to_csv([{"id": "u", "title": "T", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "line_number": "1", "fence_marker": "```", "language": "python", "meta": "linenos title='Demo'", "raw_info_string": "python linenos title='Demo'"},
        {"unit_id": "u", "title": "T", "line_number": "7", "fence_marker": "~~~", "language": "go", "meta": "meta", "raw_info_string": "go meta"},
    ]


def test_code_fence_meta_csv_writes_path(tmp_path):
    path = tmp_path / "fences.csv"
    stats = export_unit_markdown_code_fence_meta_to_csv([{"id": "u", "content": "```python meta\n```"}], path)

    assert stats["rows_exported"] == 1
    assert rows(path.read_text(encoding="utf-8"))[0]["meta"] == "meta"
