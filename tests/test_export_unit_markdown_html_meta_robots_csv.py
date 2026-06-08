import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_meta_robots_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_meta_robots_csv_parses_crawler_directives_and_skips_fences():
    content = """```
<meta name="robots" content="noindex">
```
<meta name="description" content="skip">
<meta id="r" name="Robots" content="NOINDEX, nofollow, max-snippet:20">
<meta name="googlebot" content="noarchive,nosnippet">
<meta name="bingbot" content="index,follow">"""

    result = rows(export_units_to_markdown_html_meta_robots_csv([{"id": "u", "content": content}]))

    assert [row["name"] for row in result] == ["Robots", "googlebot", "bingbot"]
    assert result[0]["directives"] == "noindex|nofollow|max-snippet:20"
    assert result[0]["has_noindex"] == "true"
    assert result[0]["has_nofollow"] == "true"
    assert result[0]["id"] == "r"
    assert result[1]["has_noarchive"] == "true"
    assert result[1]["has_nosnippet"] == "true"
    assert result[2]["has_noindex"] == "false"
