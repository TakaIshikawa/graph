from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_iframe_sandbox_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_iframe_sandbox_security_and_skips_fences():
    content = """```
<iframe src="skip"></iframe>
```
<iframe id="p" class="embed" src="video" title="Player" sandbox="allow-scripts allow-same-origin" allow="fullscreen" referrerpolicy="no-referrer" loading="lazy"></iframe>
<iframe src="untitled" sandbox></iframe>"""

    rows = _rows(export_units_to_markdown_html_iframe_sandbox_csv([{"id": "u", "content": content}]))

    assert rows[0]["src"] == "video"
    assert rows[0]["sandbox_token_count"] == "2"
    assert rows[0]["allows_scripts"] == "true"
    assert rows[0]["allows_same_origin"] == "true"
    assert rows[0]["missing_title"] == "false"
    assert rows[0]["referrerpolicy"] == "no-referrer"
    assert rows[1]["missing_title"] == "true"
