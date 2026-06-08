import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_aria_attribute_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_aria_attribute_csv_exports_attributes_skips_fences_and_sorts():
    content = """```html
<button aria-label="Skip">Skip</button>
```
<section id="panel" class="region" aria-HIDDEN="false" aria-label='Main panel'><strong>Panel</strong> text</section>
<button aria-expanded>Toggle</button>"""

    result = rows(
        export_units_to_markdown_html_aria_attribute_csv(
            [
                {"id": "b", "title": "B", "content": '<div aria-live="polite">Later</div>'},
                {"id": "a", "title": "A", "source_path": "doc.md", "source": "notes", "content": content},
            ]
        )
    )

    assert [(row["unit_id"], row["line_number"], row["tag_name"], row["attribute_name"]) for row in result] == [
        ("a", "4", "section", "aria-hidden"),
        ("a", "4", "section", "aria-label"),
        ("a", "5", "button", "aria-expanded"),
        ("b", "1", "div", "aria-live"),
    ]
    assert result[0]["attribute_value"] == "false"
    assert result[0]["is_empty"] == "false"
    assert result[0]["id"] == "panel"
    assert result[0]["class"] == "region"
    assert result[0]["text_preview"] == "Panel text"
    assert result[2]["attribute_value"] == ""
    assert result[2]["is_empty"] == "true"
    assert result[0]["source_path"] == "doc.md"
    assert result[0]["source"] == "notes"


def test_aria_attribute_csv_writes_optional_path(tmp_path):
    path = tmp_path / "aria.csv"

    meta = export_units_to_markdown_html_aria_attribute_csv([{"id": "u", "content": '<div aria-label="Name"></div>'}], path)

    assert meta["path"] == str(path)
    assert meta["unit_count"] == 1
    assert meta["rows_exported"] == 1
    assert meta["bytes_written"] == path.stat().st_size
    assert rows(path.read_text())[0]["attribute_name"] == "aria-label"
