import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_image_dimension_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_image_dimension_csv_exports_metadata_empty_src_domains_and_skips_fences():
    content = """```
<img src="skip.png">
```
<img src="https://img.example.com/a.png" alt="A" width="640" height="480" loading="lazy" decoding="async" srcset="a 1x" sizes="100vw" usemap="#m" ismap>
<img alt="missing">"""

    result = rows(export_units_to_markdown_html_image_dimension_csv([{"id": "u", "content": content}]))

    assert [row["src"] for row in result] == ["https://img.example.com/a.png", ""]
    assert result[0]["width"] == "640"
    assert result[0]["height"] == "480"
    assert result[0]["srcset_present"] == "true"
    assert result[0]["ismap"] == "true"
    assert result[0]["domain"] == "img.example.com"
    assert result[1]["alt"] == "missing"
