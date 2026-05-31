import csv
from io import StringIO

from graph.export import export_units_to_markdown_code_fence_attribute_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_code_fence_attribute_csv_exports_language_and_attributes():
    content = "```python {#demo .fast key=value}\nprint(1)\n```\n```js\nx\n```"

    result = rows(export_units_to_markdown_code_fence_attribute_csv([{"id": "u", "title": "T", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "line": "1", "language": "python", "attribute_type": "class", "attribute_name": "fast", "attribute_value": ""},
        {"unit_id": "u", "title": "T", "line": "1", "language": "python", "attribute_type": "id", "attribute_name": "demo", "attribute_value": ""},
        {"unit_id": "u", "title": "T", "line": "1", "language": "python", "attribute_type": "key_value", "attribute_name": "key", "attribute_value": "value"},
    ]


def test_code_fence_attribute_csv_writes_path(tmp_path):
    path = tmp_path / "attrs.csv"
    stats = export_units_to_markdown_code_fence_attribute_csv([{"id": "u", "content": "```python {.x}\n```"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["attribute_name"] == "x"
    assert stats["rows_exported"] == 1
