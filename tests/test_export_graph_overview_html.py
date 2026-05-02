from __future__ import annotations

from graph.export import render_graph_overview_html


def test_render_graph_overview_html_returns_standalone_html():
    text = render_graph_overview_html({"counts": {"units": 2}})

    assert text.startswith("<!doctype html>\n<html lang=\"en\">")
    assert "<meta charset=\"utf-8\">" in text
    assert "<style>" in text
    assert "<title>Graph overview</title>" in text
    assert text.endswith("</html>\n")


def test_render_graph_overview_html_includes_all_summary_sections():
    text = render_graph_overview_html(
        {
            "counts": {"units": 10, "edges": 14},
            "top_tags": [{"tag": "solar", "count": 5}, {"tag": "storage", "count": 3}],
            "top_sources": {"max": 6, "pinboard": 4},
            "central_units": [
                {
                    "id": "unit-a",
                    "title": "Alpha",
                    "score": 0.91,
                    "degree": 7,
                    "source_project": "max",
                    "tags": ["solar", "planning"],
                }
            ],
            "components": [
                {
                    "id": "component-1",
                    "unit_count": 8,
                    "edge_count": 9,
                    "density": 0.32,
                    "sample_units": ["unit-a", "unit-b"],
                }
            ],
            "warnings": ["2 isolated units"],
        }
    )

    for heading in (
        "Counts",
        "Top Tags",
        "Top Sources",
        "Central Units",
        "Components",
        "Warnings",
    ):
        assert f"<h2>{heading}</h2>" in text
    for value in (
        "units",
        "10",
        "solar",
        "storage",
        "max",
        "pinboard",
        "Alpha",
        "unit-a",
        "component-1",
        "unit-b",
        "2 isolated units",
    ):
        assert value in text


def test_render_graph_overview_html_escapes_user_values():
    text = render_graph_overview_html(
        {
            "counts": {"<units>": "<script>alert(1)</script>"},
            "top_tags": [{"tag": "alpha & beta", "count": "5 > 3"}],
            "top_sources": [{"source": "\"quoted\"", "count": 1}],
            "central_units": [
                {
                    "id": "unit<1>",
                    "title": "Title <b>bold</b>",
                    "score": "0.9",
                    "degree": "7",
                    "source_project": "max & pinboard",
                    "tags": ["x<y"],
                }
            ],
            "components": [
                {
                    "id": "component<1>",
                    "unit_count": "2",
                    "edge_count": "1",
                    "sample_units": ["sample<script>"],
                }
            ],
            "warnings": ["Disconnected <unit> & source"],
        }
    )

    assert "<script>alert(1)</script>" not in text
    assert "<b>bold</b>" not in text
    assert "sample<script>" not in text
    assert "&lt;units&gt;" in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "alpha &amp; beta" in text
    assert "5 &gt; 3" in text
    assert "&quot;quoted&quot;" in text
    assert "Title &lt;b&gt;bold&lt;/b&gt;" in text
    assert "Disconnected &lt;unit&gt; &amp; source" in text


def test_render_graph_overview_html_handles_missing_optional_sections():
    text = render_graph_overview_html({})

    assert "No counts provided." in text
    assert "No top tags provided." in text
    assert "No top sources provided." in text
    assert "No central units provided." in text
    assert "No components provided." in text
    assert "No warnings provided." in text


def test_render_graph_overview_html_renders_deterministically_except_generated_time():
    summary = {
        "counts": {"units": 3, "edges": 2},
        "top_tags": [{"tag": "zeta", "count": 1}, {"tag": "alpha", "count": 4}],
        "top_sources": [("pinboard", 1), ("max", 4)],
        "central_units": [
            {"id": "unit-b", "title": "B", "score": 0.1},
            {"id": "unit-a", "title": "A", "score": 0.9},
        ],
        "components": [
            {"id": "small", "unit_count": 2},
            {"id": "large", "unit_count": 5},
        ],
        "warnings": ["z warning", "a warning"],
    }

    first = _without_generated_time(render_graph_overview_html(summary))
    second = _without_generated_time(render_graph_overview_html(summary))

    assert first == second
    assert first.index(">edges<") < first.index(">units<")
    assert first.index(">alpha<") < first.index(">zeta<")
    assert first.index(">max<") < first.index(">pinboard<")
    assert first.index(">A<") < first.index(">B<")
    assert first.index(">large<") < first.index(">small<")
    assert first.index("a warning") < first.index("z warning")


def test_render_graph_overview_html_is_importable_from_graph_export():
    from graph.export import render_graph_overview_html as imported

    assert imported is render_graph_overview_html


def _without_generated_time(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "Generated:" in line:
            lines.append("<span>Generated: TIMESTAMP</span>")
        else:
            lines.append(line)
    return "\n".join(lines)
