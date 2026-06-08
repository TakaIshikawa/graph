import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_contenteditable_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_contenteditable_csv_classifies_values_and_skips_fences():
    content = """```
<div contenteditable>Skip</div>
```
<div id="empty" class="edit" contenteditable><span>Edit</span> me</div>
<p contenteditable="true">True</p>
<section contenteditable="false">False</section>
<article contenteditable="plaintext-only">Plain</article>"""

    result = rows(export_units_to_markdown_html_contenteditable_csv([{"id": "u", "source_path": "doc.md", "source": "s", "content": content}]))

    assert [row["contenteditable"] for row in result] == ["", "true", "false", "plaintext-only"]
    assert result[0]["is_true"] == "true"
    assert result[0]["is_empty"] == "true"
    assert result[0]["id"] == "empty"
    assert result[0]["class"] == "edit"
    assert result[0]["text_preview"] == "Edit me"
    assert result[1]["is_true"] == "true"
    assert result[2]["is_false"] == "true"
    assert result[3]["is_plaintext_only"] == "true"


def test_contenteditable_csv_writes_optional_path(tmp_path):
    path = tmp_path / "contenteditable.csv"

    meta = export_units_to_markdown_html_contenteditable_csv([{"id": "u", "content": "<div contenteditable></div>"}], path)

    assert meta["rows_exported"] == 1
    assert path.read_text().startswith("unit_id,title,source_path,source,line_number")
