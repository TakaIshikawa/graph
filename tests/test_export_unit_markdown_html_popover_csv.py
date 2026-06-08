import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_popover_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_popover_csv_exports_elements_and_triggers_skipping_fences():
    content = """```html
<button popovertarget="skip">Skip</button>
```
<div id="menu" class="panel" popover="manual"><strong>Menu</strong> body</div>
<button popovertarget="menu" popovertargetaction="show">Open</button>
<button popovertarget="menu" popovertargetaction="hide">Close</button>"""

    result = rows(export_units_to_markdown_html_popover_csv([{"id": "u", "content": content}]))

    assert [row["tag_name"] for row in result] == ["div", "button", "button"]
    assert result[0]["is_popover_element"] == "true"
    assert result[0]["is_popover_trigger"] == "false"
    assert result[0]["popover"] == "manual"
    assert result[0]["id"] == "menu"
    assert result[0]["class"] == "panel"
    assert result[0]["text_preview"] == "Menu body"
    assert result[1]["is_popover_trigger"] == "true"
    assert result[1]["popovertarget"] == "menu"
    assert result[1]["target_action"] == "show"
    assert result[2]["target_action"] == "hide"


def test_popover_csv_writes_optional_path(tmp_path):
    path = tmp_path / "popover.csv"

    meta = export_units_to_markdown_html_popover_csv([{"id": "u", "content": '<div id="p" popover></div>'}], path)

    assert meta["rows_exported"] == 1
    assert rows(path.read_text())[0]["is_popover_element"] == "true"
