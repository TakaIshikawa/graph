import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_track_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_track_csv_exports_parent_track_metadata_and_skips_fences(tmp_path):
    content = """```
<video><track src="skip.vtt"></video>
```
<video src="movie.mp4"><track src="https://cdn.example.com/en.vtt" kind="subtitles" srclang="en" label="English" default><track src="/fr.vtt" kind="captions" srclang="fr" label="French"></video>
<audio><source src="song.mp3"><track src="https://audio.example.org/chapters.vtt" kind="chapters" label="Chapters"></audio>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_track_csv(units)
    result = rows(text)

    assert [row["parent_tag"] for row in result] == ["video", "video", "audio"]
    assert result[0]["parent_src"] == "movie.mp4"
    assert result[0]["src"] == "https://cdn.example.com/en.vtt"
    assert result[0]["domain"] == "cdn.example.com"
    assert result[0]["kind"] == "subtitles"
    assert result[0]["srclang"] == "en"
    assert result[0]["label"] == "English"
    assert result[0]["default"] == "true"
    assert result[1]["track_index"] == "2"
    assert result[2]["parent_src"] == "song.mp3"
    assert result[2]["domain"] == "audio.example.org"

    output = tmp_path / "track.csv"
    stats = export_units_to_markdown_html_track_csv(units, output)
    assert stats["rows_exported"] == 3
    assert output.read_text() == text
