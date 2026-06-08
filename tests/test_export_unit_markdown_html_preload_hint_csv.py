from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_preload_hint_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_preload_hint_csv_exports_supported_hints_and_skips_fences():
    content = """```
<link rel="preload" href="skip.js">
```
<link id="p" class="hint" rel="preload" href="/a.js" as="script" type="text/javascript" crossorigin media="all">
<link rel="modulepreload" href="/m.js">
<link rel="preconnect" href="https://cdn.example">
<link rel="dns-prefetch" href="//dns.example">
<link rel="prefetch" href="/next">
<link rel="prerender" href="/page">
<link rel="stylesheet" href="/style.css">"""

    rows = _rows(export_units_to_markdown_html_preload_hint_csv([{"id": "u", "content": content}]))

    assert [row["hint_kind"] for row in rows] == ["preload", "modulepreload", "preconnect", "dns-prefetch", "prefetch", "prerender"]
    assert rows[0]["as_attr"] == "script"
    assert rows[0]["crossorigin"] == ""
    assert rows[0]["media"] == "all"
    assert rows[0]["id"] == "p"


def test_preload_hint_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "hints.csv"
    units = [{"id": "u", "content": '<link rel="PRELOAD" href="/x">'}]

    expected = export_units_to_markdown_html_preload_hint_csv(units)
    stats = export_units_to_markdown_html_preload_hint_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
