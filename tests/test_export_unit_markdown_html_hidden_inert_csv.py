import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_hidden_inert_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_hidden_inert_csv_exports_visibility_signals_and_skips_fences():
    content = """```
<div hidden>Skip</div>
```
<section id="a" hidden><span>Hidden</span> text</section>
<div hidden="until-found" inert aria-hidden="true">Findable</div>
<p aria-hidden="false">Visible to AT</p>"""

    result = rows(export_units_to_markdown_html_hidden_inert_csv([{"id": "u", "content": content}]))

    assert [row["tag_name"] for row in result] == ["section", "div", "p"]
    assert result[0]["hidden"] == ""
    assert result[0]["text_preview"] == "Hidden text"
    assert result[1]["hidden"] == "until-found"
    assert result[1]["inert"] == "true"
    assert result[1]["aria_hidden"] == "true"
    assert result[1]["is_hidden_until_found"] == "true"
    assert result[2]["aria_hidden"] == "false"
