from __future__ import annotations

from graph.export import render_search_html_report


def test_render_search_html_report_is_importable_from_graph_export():
    from graph.export.html_report import render_search_html_report as direct

    assert render_search_html_report is direct


def test_render_search_html_report_escapes_user_controlled_fields():
    text = render_search_html_report(
        {
            "query": "<script>alert(1)</script>",
            "mode": "hybrid",
            "sort": "relevance",
            "filters": {"owner": "Ada <admin>"},
            "results": [
                {
                    "title": "Title <b>bold</b>",
                    "snippet": "Snippet with <img src=x onerror=alert(1)>",
                    "tags": ["alpha & beta", "tag<script>"],
                    "metadata": {"raw": "<unsafe>", "quote": '"quoted"'},
                }
            ],
        }
    )

    assert "<script>alert(1)</script>" not in text
    assert "<b>bold</b>" not in text
    assert "<img src=x onerror=alert(1)>" not in text
    assert "tag<script>" not in text
    assert "Ada <admin>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "Title &lt;b&gt;bold&lt;/b&gt;" in text
    assert "Snippet with &lt;img src=x onerror=alert(1)&gt;" in text
    assert "alpha &amp; beta" in text
    assert "tag&lt;script&gt;" in text
    assert "Ada &lt;admin&gt;" in text
    assert "&lt;unsafe&gt;" in text
    assert "&quot;\\&quot;quoted\\&quot;&quot;" in text


def test_render_search_html_report_renders_empty_results_section():
    text = render_search_html_report({"query": "missing", "results": []})

    assert "<span>0 results</span>" in text
    assert '<section class="empty">No results found.</section>' in text


def test_render_search_html_report_formats_score_tags_and_metadata_details():
    text = render_search_html_report(
        {
            "query": "solar",
            "results": [
                {
                    "title": "Solar storage",
                    "snippet": "Battery research",
                    "score": 0.87654,
                    "tags": ["solar", "storage"],
                    "metadata": {"source": "paper", "priority": 2},
                }
            ],
        }
    )

    assert "<span>Score 0.877</span>" in text
    assert '<span class="tag">solar</span>' in text
    assert '<span class="tag">storage</span>' in text
    assert "<details><summary>Metadata</summary><pre>" in text
    assert "&quot;priority&quot;: 2" in text
    assert "&quot;source&quot;: &quot;paper&quot;" in text


def test_render_search_html_report_renders_no_tags_fallback():
    text = render_search_html_report(
        {
            "query": "untagged",
            "results": [{"title": "No tags unit", "snippet": "Plain result"}],
        }
    )

    assert '<span class="muted">No tags</span>' in text
