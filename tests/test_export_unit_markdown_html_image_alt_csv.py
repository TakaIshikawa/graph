from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_image_alt_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_image_alt_states_and_skips_fences():
    content = """```
<img src="skip.png">
```
<img id="hero" class="wide" src="hero.png" alt="Hero" title="Cover" width="640" height="480" loading="lazy" decoding="async">
<img src="decor.svg" alt="">
<img src="missing.png">"""

    rows = _rows(export_units_to_markdown_html_image_alt_csv([{"id": "u", "content": content}]))

    assert [row["src"] for row in rows] == ["hero.png", "decor.svg", "missing.png"]
    assert rows[0]["missing_alt"] == "false"
    assert rows[0]["empty_alt"] == "false"
    assert rows[0]["width"] == "640"
    assert rows[0]["loading"] == "lazy"
    assert rows[1]["empty_alt"] == "true"
    assert rows[2]["missing_alt"] == "true"
