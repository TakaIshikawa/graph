import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_menu_command_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_menu_command_csv_exports_commands_and_skips_fences(tmp_path):
    content = """```
<menu><command label="Skip"></menu>
```
<menu>
  <command type="checkbox" label="Bold &amp; bright" command="bold" icon="bold.svg" checked disabled>
  <menuitem type="radio" label="Left" radiogroup="align">
  <button type="button" disabled>Save &amp; close</button>
</menu>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_menu_command_csv(units)
    result = rows(text)

    assert [row["tag"] for row in result] == ["command", "menuitem", "button"]
    assert result[0]["label"] == "Bold & bright"
    assert result[0]["type"] == "checkbox"
    assert result[0]["command"] == "bold"
    assert result[0]["icon"] == "bold.svg"
    assert result[0]["checked"] == "true"
    assert result[0]["disabled"] == "true"
    assert result[1]["radiogroup"] == "align"
    assert result[2]["label"] == "Save & close"
    assert result[2]["text_preview"] == "Save & close"

    output = tmp_path / "menu.csv"
    stats = export_units_to_markdown_html_menu_command_csv(units, output)
    assert stats["rows_exported"] == 3
    assert output.read_text() == text
