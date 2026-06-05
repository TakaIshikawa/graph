import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_anchor_download_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_anchor_download_csv_exports_download_metadata_and_skips_fences():
    content = """```
<a href="skip.zip" download>Skip</a>
```
<a href="https://example.com/file.zip" download rel="nofollow" target="_blank" type="application/zip" hreflang="en"><strong>Download</strong> file</a>
<a href="/empty" download>Empty</a>
<a href="/plain">Plain</a>
<a href="https://cdn.example.org/report.pdf" type="application/pdf">Report &amp; data</a>"""

    result = rows(export_units_to_markdown_html_anchor_download_csv([{"id": "u", "title": "T", "source_path": "doc.md", "source": "s", "content": content}]))

    assert [row["href"] for row in result] == ["https://example.com/file.zip", "/empty", "https://cdn.example.org/report.pdf"]
    assert result[0]["download"] == ""
    assert result[0]["rel"] == "nofollow"
    assert result[0]["target"] == "_blank"
    assert result[0]["hreflang"] == "en"
    assert result[0]["text_preview"] == "Download file"
    assert result[0]["domain"] == "example.com"
    assert result[2]["text_preview"] == "Report & data"
    assert result[2]["domain"] == "cdn.example.org"
