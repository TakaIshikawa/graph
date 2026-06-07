import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_embed_object_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_embed_object_csv_exports_media_params_and_skips_fences(tmp_path):
    content = """```
<embed src="https://skip.example/file.pdf">
```
<embed src="https://cdn.example.com/file.pdf" type="application/pdf" width="600" height="400" name="pdf">
<object data="https://media.example.org/movie.swf" type="application/x-shockwave-flash" width="320">
  <param name="movie" value="movie.swf">
  <param name="quality" value="high">
  Fallback <strong>player</strong> &amp; link
</object>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_embed_object_csv(units)
    result = rows(text)

    assert [row["tag"] for row in result] == ["embed", "object", "object"]
    assert result[0]["src_or_data"] == "https://cdn.example.com/file.pdf"
    assert result[0]["domain"] == "cdn.example.com"
    assert result[0]["name"] == "pdf"
    assert result[1]["param_name"] == "movie"
    assert result[1]["param_value"] == "movie.swf"
    assert result[1]["fallback_preview"] == "Fallback player & link"
    assert result[1]["domain"] == "media.example.org"
    assert result[2]["param_name"] == "quality"

    output = tmp_path / "embed_object.csv"
    stats = export_units_to_markdown_html_embed_object_csv(units, output)
    assert stats["rows_exported"] == 3
    assert output.read_text() == text
