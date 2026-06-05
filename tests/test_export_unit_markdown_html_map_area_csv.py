import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_map_area_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_map_area_csv_exports_area_metadata_and_skips_fences(tmp_path):
    content = """```
<map name="skip"><area href="skip.html" alt="Skip"></map>
```
<map name="floor" id="floor-map">
<area shape="rect" coords="0,0,100,50" href="https://example.com/room?a=1&amp;b=2" target="_blank" alt="Room &amp; details" rel="nofollow" media="screen" type="text/html" hreflang="en" referrerpolicy="no-referrer" ping="/track">
<area shape="circle" coords="50,50,20" href="/lobby" download="lobby.pdf" alt="Lobby">
<area shape="poly" coords="0,0,10,10,20,0" nohref alt="Closed">
</map>"""

    units = [{"id": "u", "title": "Plan", "source_path": "doc.md", "source": "manual", "content": content}]
    text = export_units_to_markdown_html_map_area_csv(units)
    result = rows(text)

    assert [row["shape"] for row in result] == ["rect", "circle", "poly"]
    assert result[0]["map_name"] == "floor"
    assert result[0]["map_id"] == "floor-map"
    assert result[0]["href"] == "https://example.com/room?a=1&b=2"
    assert result[0]["alt"] == "Room & details"
    assert result[0]["target"] == "_blank"
    assert result[0]["rel"] == "nofollow"
    assert result[0]["media"] == "screen"
    assert result[0]["type"] == "text/html"
    assert result[0]["hreflang"] == "en"
    assert result[0]["referrerpolicy"] == "no-referrer"
    assert result[0]["ping_present"] == "true"
    assert result[0]["domain"] == "example.com"
    assert result[1]["download"] == "lobby.pdf"
    assert result[2]["nohref"] == "true"

    output = tmp_path / "map_area.csv"
    stats = export_units_to_markdown_html_map_area_csv(units, output)
    assert stats["rows_exported"] == 3
    assert output.read_text() == text
