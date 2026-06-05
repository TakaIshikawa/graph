import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_template_slot_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_template_slot_csv_exports_previews_shadowroot_and_skips_fences():
    content = """```
<slot name="skip"></slot>
```
<template id="card" shadowrootmode="open">
<p>Hello &amp; <strong>world</strong></p>
</template>
<slot name="actions">Default</slot>
<slot name="standalone">"""

    result = rows(export_units_to_markdown_html_template_slot_csv([{"id": "u", "content": content}]))

    assert [row["tag"] for row in result] == ["template", "slot", "slot"]
    assert result[0]["id"] == "card"
    assert result[0]["shadowrootmode"] == "open"
    assert result[0]["content_preview"] == "Hello & world"
    assert result[0]["multiline"] == "true"
    assert result[1]["name"] == "actions"
    assert result[1]["content_preview"] == "Default"
    assert result[2]["name"] == "standalone"
