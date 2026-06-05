import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_link_rel_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_link_rel_csv_exports_link_metadata_unquoted_attrs_and_skips_fences():
    content = """```
<link rel="stylesheet" href="skip.css">
```
<link rel=stylesheet href=https://example.com/app.css media="screen">
<link rel="preload" as="script" href="/app.js" type="text/javascript" integrity="sha" crossorigin referrerpolicy="no-referrer">
<link rel="canonical" href="https://docs.example.org/page" hreflang="en" sizes="any">"""

    result = rows(export_units_to_markdown_html_link_rel_csv([{"id": "u", "content": content}]))

    assert [row["rel"] for row in result] == ["stylesheet", "preload", "canonical"]
    assert result[2]["domain"] == "docs.example.org"
    assert result[1]["as"] == "script"
    assert result[1]["crossorigin"] == ""
    assert result[0]["href"] == "https://example.com/app.css"
    assert result[0]["domain"] == "example.com"
