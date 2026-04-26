"""Self-contained HTML reports for graph search results."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone


def _escape(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _format_metadata(value: object) -> str:
    return html.escape(json.dumps(value, sort_keys=True, default=str), quote=True)


def render_search_html_report(payload: dict) -> str:
    """Render a standalone HTML search report.

    All user-controlled values are escaped before interpolation.
    """
    query = payload.get("query", "")
    mode = payload.get("mode", "")
    sort = payload.get("sort", "relevance")
    filters = payload.get("filters", {})
    results = payload.get("results", [])
    generated_at = datetime.now(timezone.utc).isoformat()
    title = f"Graph search: {query}"

    filter_items = "".join(
        f"<li><span>{_escape(key)}</span><code>{_format_metadata(value)}</code></li>"
        for key, value in sorted(filters.items())
    )
    if not filter_items:
        filter_items = "<li><span>filters</span><code>{}</code></li>"

    result_items = []
    for index, result in enumerate(results, start=1):
        tags = "".join(f"<span class=\"tag\">{_escape(tag)}</span>" for tag in result.get("tags", []))
        if not tags:
            tags = "<span class=\"muted\">No tags</span>"

        score = ""
        if "score" in result:
            score_value = f"{float(result['score']):.3f}"
            score = f"<span>Score {_escape(score_value)}</span>"

        link_counts = []
        if "link_count" in result:
            link_counts.append(f"{_escape(result['link_count'])} links")
        if "backlink_count" in result:
            link_counts.append(f"{_escape(result['backlink_count'])} backlinks")
        link_summary = ""
        if link_counts:
            link_summary = f"<span>{_escape(' / '.join(link_counts))}</span>"

        metadata = result.get("metadata") or {}
        metadata_block = ""
        if metadata:
            metadata_block = f"<details><summary>Metadata</summary><pre>{_format_metadata(metadata)}</pre></details>"

        snippet = result.get("snippet") or result.get("content", "")
        result_items.append(
            "\n".join(
                [
                    "<article class=\"result\">",
                    f"<div class=\"rank\">{index}</div>",
                    "<div class=\"body\">",
                    f"<h2>{_escape(result.get('title', 'Untitled'))}</h2>",
                    "<div class=\"meta\">",
                    f"<span>{_escape(result.get('source_project'))}</span>",
                    f"<span>{_escape(result.get('content_type'))}</span>",
                    f"<span>{_escape(result.get('created_at'))}</span>",
                    score,
                    link_summary,
                    "</div>",
                    f"<p class=\"snippet\">{_escape(snippet)}</p>",
                    f"<div class=\"tags\">{tags}</div>",
                    metadata_block,
                    "</div>",
                    "</article>",
                ]
            )
        )

    if not result_items:
        result_items.append("<section class=\"empty\">No results found.</section>")

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"<title>{_escape(title)}</title>",
            "<style>",
            """
:root { color-scheme: light; --fg: #202124; --muted: #5f6368; --line: #d9dde3; --bg: #f6f7f9; --accent: #0b57d0; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.45; }
header { background: #fff; border-bottom: 1px solid var(--line); padding: 28px max(24px, calc((100vw - 980px) / 2)); }
main { max-width: 980px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 12px; font-size: 28px; font-weight: 650; }
h2 { margin: 0 0 8px; font-size: 19px; }
.summary { display: flex; flex-wrap: wrap; gap: 10px; color: var(--muted); font-size: 14px; }
.filters, .result { background: #fff; border: 1px solid var(--line); border-radius: 8px; }
.filters { margin-bottom: 18px; padding: 16px 18px; }
.filters h2 { font-size: 15px; margin-bottom: 10px; }
.filters ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
.filters li { display: flex; gap: 12px; align-items: baseline; }
.filters span { min-width: 140px; color: var(--muted); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }
.result { display: grid; grid-template-columns: 42px 1fr; gap: 14px; padding: 18px; margin-bottom: 14px; }
.rank { width: 32px; height: 32px; border-radius: 50%; background: #e8f0fe; color: var(--accent); display: grid; place-items: center; font-weight: 700; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 14px; color: var(--muted); font-size: 13px; margin-bottom: 10px; }
.snippet { margin: 0 0 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.tag { border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; font-size: 12px; background: #fbfcfe; }
.muted { color: var(--muted); }
details { margin-top: 8px; }
summary { cursor: pointer; color: var(--accent); }
pre { background: #f8fafd; border: 1px solid var(--line); border-radius: 6px; padding: 10px; }
.empty { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 24px; color: var(--muted); }
""".strip(),
            "</style>",
            "</head>",
            "<body>",
            "<header>",
            f"<h1>{_escape(title)}</h1>",
            "<div class=\"summary\">",
            f"<span>{_escape(len(results))} results</span>",
            f"<span>Mode: {_escape(mode)}</span>",
            f"<span>Sort: {_escape(sort)}</span>",
            f"<span>Generated: {_escape(generated_at)}</span>",
            "</div>",
            "</header>",
            "<main>",
            "<section class=\"filters\">",
            "<h2>Search Metadata</h2>",
            "<ul>",
            f"<li><span>query</span><code>{_escape(query)}</code></li>",
            f"<li><span>mode</span><code>{_escape(mode)}</code></li>",
            f"<li><span>sort</span><code>{_escape(sort)}</code></li>",
            *filter_items.splitlines(),
            "</ul>",
            "</section>",
            *result_items,
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )
