import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_accesskey_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_accesskey_csv_counts_keys_sorts_previews_and_skips_fences():
    content = """<button accesskey="s a s">Save</button>
```
<a accesskey="x">Skip</a>
```
<a id="home" class="nav" accesskey="h"><span>Home</span> link</a>
<input accesskey="">"""

    result = rows(export_units_to_markdown_html_accesskey_csv([{"id": "u", "content": content}]))

    assert [row["tag_name"] for row in result] == ["button", "a", "input"]
    assert result[0]["accesskey"] == "s a s"
    assert result[0]["key_count"] == "3"
    assert result[0]["first_key"] == "s"
    assert result[0]["has_multiple_keys"] == "true"
    assert result[1]["id"] == "home"
    assert result[1]["class"] == "nav"
    assert result[1]["text_preview"] == "Home link"
    assert result[2]["key_count"] == "0"
    assert result[2]["first_key"] == ""
