import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_tabindex_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_tabindex_csv_classifies_values_and_skips_fences():
    content = """```
<div tabindex="9">Skip</div>
```
<button id="start" class="focus" tabindex="2">Start</button>
<a tabindex="0">Link</a>
<section tabindex="-1">Panel</section>
<div tabindex="later">Bad</div>"""

    result = rows(export_units_to_markdown_html_tabindex_csv([{"id": "u", "source_path": "doc.md", "source": "s", "content": content}]))

    assert [row["tabindex"] for row in result] == ["2", "0", "-1", "later"]
    assert result[0]["tabindex_int"] == "2"
    assert result[0]["is_positive"] == "true"
    assert result[0]["id"] == "start"
    assert result[0]["class"] == "focus"
    assert result[0]["text_preview"] == "Start"
    assert result[1]["is_zero"] == "true"
    assert result[2]["is_negative"] == "true"
    assert result[3]["tabindex_int"] == ""
    assert result[3]["is_invalid"] == "true"


def test_tabindex_csv_writes_optional_path(tmp_path):
    path = tmp_path / "tabindex.csv"

    meta = export_units_to_markdown_html_tabindex_csv([{"id": "u", "content": '<div tabindex="-1"></div>'}], path)

    assert meta["rows_exported"] == 1
    assert rows(path.read_text())[0]["tabindex_int"] == "-1"
