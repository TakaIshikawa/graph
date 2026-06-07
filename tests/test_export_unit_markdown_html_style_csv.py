from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_style_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_markdown_html_style_elements_and_skips_fences(tmp_path):
    content = """```html
<style>.skip { background: url(skip.png); }</style>
```
<style id="critical" class="theme" media="screen" type="text/css" nonce="abc">
@import url("base.css");
.hero { background-image: url('/hero.png'); }
</style>
<style media="print">   </style>"""
    units = [{"id": "u", "title": "Unit", "source": "manual", "content": content}]

    text = export_units_to_markdown_html_style_csv(units)
    rows = _rows(text)

    assert len(rows) == 2
    assert rows[0]["unit_id"] == "u"
    assert rows[0]["title"] == "Unit"
    assert rows[0]["source"] == "manual"
    assert rows[0]["line_number"] == "4"
    assert rows[0]["media"] == "screen"
    assert rows[0]["type"] == "text/css"
    assert rows[0]["nonce"] == "abc"
    assert rows[0]["css_preview"] == '@import url("base.css"); .hero { background-image: url(\'/hero.png\'); }'
    assert rows[0]["character_count"] == "70"
    assert rows[0]["import_rule_count"] == "1"
    assert rows[0]["url_reference_count"] == "2"
    assert rows[0]["empty_style"] == "false"
    assert rows[0]["id"] == "critical"
    assert rows[0]["class"] == "theme"
    assert rows[1]["media"] == "print"
    assert rows[1]["empty_style"] == "true"
    assert rows[1]["character_count"] == "0"

    output = tmp_path / "style.csv"
    result = export_units_to_markdown_html_style_csv(units, output)
    assert result["rows_exported"] == 2
    assert output.read_text() == text
