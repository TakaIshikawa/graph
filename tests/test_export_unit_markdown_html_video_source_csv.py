import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_video_source_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_video_source_csv_exports_nested_sources_direct_src_path_mode_and_skips_fences(tmp_path):
    content = """```
<video src="skip.mp4"></video>
```
<video poster="/p.jpg" controls autoplay muted loop preload="metadata" width="640" height="360">
<source src="https://cdn.example.com/a.webm" type="video/webm">
<source src="/a.mp4" type="video/mp4">
</video>
<video src="https://video.example.org/direct.mp4" controls></video>"""

    units = [{"id": "u", "content": content}]
    text = export_units_to_markdown_html_video_source_csv(units)
    result = rows(text)

    assert [row["source_src"] for row in result] == ["https://cdn.example.com/a.webm", "/a.mp4", ""]
    assert result[0]["poster"] == "/p.jpg"
    assert result[0]["controls"] == "true"
    assert result[0]["autoplay"] == "true"
    assert result[0]["domain"] == "cdn.example.com"
    assert result[2]["video_src"] == "https://video.example.org/direct.mp4"
    assert result[2]["domain"] == "video.example.org"

    output = tmp_path / "video.csv"
    stats = export_units_to_markdown_html_video_source_csv(units, output)
    assert stats["rows_exported"] == 3
    assert output.read_text() == text
