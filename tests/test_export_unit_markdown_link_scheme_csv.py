import csv
from io import StringIO

from graph.export import export_units_to_markdown_link_scheme_csv


def test_link_scheme_export_classifies_links_and_excludes_images_and_fences():
    content = "\n".join(["[web](https://example.com)", "[mail](mailto:a@example.com)", "[file](file:///tmp/a)", "[app](obsidian://open)", "[rel](docs/page.md)", "![img](https://image)", "```", "[hidden](http://hidden)", "```"])
    rows = list(csv.DictReader(StringIO(export_units_to_markdown_link_scheme_csv([{"id": "u1", "content": content}]))))

    assert [(row["link_text"], row["scheme"], row["scheme_type"]) for row in rows] == [
        ("web", "https", "web"),
        ("mail", "mailto", "mail"),
        ("file", "file", "file"),
        ("app", "obsidian", "app"),
        ("rel", "relative", "unknown"),
    ]
