import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_draggable_spellcheck_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_draggable_spellcheck_csv_classifies_known_values_and_skips_fences():
    content = """```
<div draggable="true">Skip</div>
```
<div id="drag" draggable="true" spellcheck="false"><span>Drag</span> me</div>
<p draggable="false">No drag</p>
<textarea spellcheck="true">Words</textarea>
<section draggable="auto" spellcheck></section>"""

    result = rows(export_units_to_markdown_html_draggable_spellcheck_csv([{"id": "u", "content": content}]))

    assert [row["tag_name"] for row in result] == ["div", "p", "textarea", "section"]
    assert result[0]["is_draggable_true"] == "true"
    assert result[0]["is_spellcheck_false"] == "true"
    assert result[0]["text_preview"] == "Drag me"
    assert result[1]["is_draggable_false"] == "true"
    assert result[2]["is_spellcheck_true"] == "true"
    assert result[3]["draggable"] == "auto"
    assert result[3]["spellcheck"] == ""
