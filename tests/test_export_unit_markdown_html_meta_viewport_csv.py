from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_meta_viewport_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_viewport_directives_and_skips_fences():
    content = """```
<meta name="viewport" content="width=skip">
```
<meta name="description" content="Nope">
<meta id="vp" class="mobile" name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="viewport" content="user-scalable=no">"""

    rows = _rows(export_units_to_markdown_html_meta_viewport_csv([{"id": "u", "content": content}]))

    assert len(rows) == 2
    assert rows[0]["width_value"] == "device-width"
    assert rows[0]["initial_scale"] == "1"
    assert rows[0]["maximum_scale"] == "1"
    assert rows[0]["disables_zoom"] == "true"
    assert rows[0]["id"] == "vp"
    assert rows[1]["user_scalable"] == "no"
    assert rows[1]["disables_zoom"] == "true"
