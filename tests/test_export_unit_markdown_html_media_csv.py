import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_media_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_media_csv_exports_audio_video_booleans_fallback_and_skips_fences():
    content = """```
<audio src="skip.mp3"></audio>
```
<audio src="song.mp3" controls preload="metadata">Fallback <strong>audio</strong></audio>
<video id="hero" src="clip.mp4" autoplay loop muted playsinline width="640" height="360">Fallback video</video>"""

    result = rows(export_units_to_markdown_html_media_csv([{"id": "u", "content": content}]))

    assert [row["tag_name"] for row in result] == ["audio", "video"]
    assert result[0]["controls"] == "true"
    assert result[0]["preload"] == "metadata"
    assert result[0]["text_preview"] == "Fallback audio"
    assert result[1]["autoplay"] == "true"
    assert result[1]["loop"] == "true"
    assert result[1]["muted"] == "true"
    assert result[1]["playsinline"] == "true"
    assert result[1]["width"] == "640"
    assert result[1]["height"] == "360"
