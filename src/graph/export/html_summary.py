"""Standalone HTML summary export for knowledge graphs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit


def export_graph_html_summary(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    recent_limit: int = 10,
    top_limit: int = 10,
) -> str | dict[str, Any]:
    """Return or write a dependency-free HTML summary for a graph."""
    if not isinstance(recent_limit, int) or isinstance(recent_limit, bool) or recent_limit < 0:
        raise ValueError("recent_limit must be a non-negative integer")
    if not isinstance(top_limit, int) or isinstance(top_limit, bool) or top_limit < 0:
        raise ValueError("top_limit must be a non-negative integer")

    unit_list = sorted(list(units), key=_unit_key)
    edge_list = sorted(list(edges), key=_edge_key)
    html = _render_html(unit_list, edge_list, recent_limit=recent_limit, top_limit=top_limit)

    if path is None:
        return html

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_exported": len(unit_list),
        "edges_exported": len(edge_list),
        "bytes_written": output_path.stat().st_size,
    }


def _render_html(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
    *,
    recent_limit: int,
    top_limit: int,
) -> str:
    source_counts = Counter(_field_value(unit.source_project) for unit in units)
    type_counts = Counter(_field_value(unit.content_type) for unit in units)
    tag_counts = Counter(tag for unit in units for tag in _unit_tags(unit))
    recent_units = _recent_units(units, recent_limit)

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            "  <title>Graph Summary</title>",
            "  <style>",
            _css(),
            "  </style>",
            "</head>",
            "<body>",
            '  <main class="page">',
            "    <header>",
            "      <h1>Graph Summary</h1>",
            "    </header>",
            '    <section class="metrics" aria-label="Graph counts">',
            _metric("Units", len(units)),
            _metric("Edges", len(edges)),
            _metric("Sources", len(source_counts)),
            _metric("Tags", len(tag_counts)),
            "    </section>",
            '    <section class="grid">',
            _breakdown_section("Source Breakdown", source_counts, top_limit),
            _breakdown_section("Content Types", type_counts, top_limit),
            _breakdown_section("Top Tags", tag_counts, top_limit),
            "    </section>",
            _recent_units_section(recent_units),
            "  </main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _metric(label: str, value: int) -> str:
    return "\n".join(
        [
            '      <article class="metric">',
            f"        <div>{escape(label)}</div>",
            f"        <strong>{value}</strong>",
            "      </article>",
        ]
    )


def _breakdown_section(title: str, counts: Counter[str], limit: int) -> str:
    rows = _top_items(counts, limit)
    if not rows:
        body = '        <p class="empty">No data</p>'
    else:
        total = sum(counts.values()) or 1
        body = "\n".join(
            [
                "        <table>",
                "          <thead><tr><th>Name</th><th>Count</th><th>Share</th></tr></thead>",
                "          <tbody>",
                *[
                    (
                        "            <tr>"
                        f"<td>{escape(name)}</td>"
                        f"<td>{count}</td>"
                        f"<td>{count / total:.0%}</td>"
                        "</tr>"
                    )
                    for name, count in rows
                ],
                "          </tbody>",
                "        </table>",
            ]
        )
    return "\n".join(
        [
            "      <section>",
            f"        <h2>{escape(title)}</h2>",
            body,
            "      </section>",
        ]
    )


def _recent_units_section(units: list[tuple[datetime, KnowledgeUnit]]) -> str:
    if not units:
        body = '      <p class="empty">No dated units</p>'
    else:
        body = "\n".join(
            [
                "      <table>",
                "        <thead><tr><th>Date</th><th>Title</th><th>Source</th><th>Type</th></tr></thead>",
                "        <tbody>",
                *[
                    (
                        "          <tr>"
                        f"<td>{escape(timestamp.isoformat())}</td>"
                        f"<td>{escape(_text(unit.title))}</td>"
                        f"<td>{escape(_field_value(unit.source_project))}</td>"
                        f"<td>{escape(_field_value(unit.content_type))}</td>"
                        "</tr>"
                    )
                    for timestamp, unit in units
                ],
                "        </tbody>",
                "      </table>",
            ]
        )
    return "\n".join(["    <section>", "      <h2>Recent Units</h2>", body, "    </section>"])


def _recent_units(units: list[KnowledgeUnit], limit: int) -> list[tuple[datetime, KnowledgeUnit]]:
    dated = [(timestamp, unit) for unit in units if (timestamp := _unit_datetime(unit)) is not None]
    return sorted(dated, key=lambda item: (item[0], _unit_key(item[1])), reverse=True)[:limit]


def _unit_datetime(unit: KnowledgeUnit) -> datetime | None:
    for value in (unit.updated_at, unit.created_at, unit.ingested_at):
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return None
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day)


def _top_items(counts: Counter[str], limit: int) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))[:limit]


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted({_text(tag) for tag in unit.tags if _text(tag)}, key=_sort_key)


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (_text(unit.id), _text(unit.source_id))


def _edge_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        _text(edge.from_unit_id),
        _text(edge.to_unit_id),
        _field_value(edge.relation),
        _text(edge.id),
    )


def _field_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)


def _text(value: object) -> str:
    return str(value or "")


def _css() -> str:
    return """    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #202124;
      --muted: #5f6368;
      --line: #d7d9d2;
      --accent: #0b6b61;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .page { width: min(1120px, calc(100vw - 32px)); margin: 32px auto; }
    header { margin-bottom: 20px; }
    h1 { margin: 0; font-size: 28px; font-weight: 700; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    section, .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      padding: 0;
      border: 0;
      background: transparent;
      margin-bottom: 12px;
    }
    .metric div { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .metric strong { display: block; margin-top: 4px; font-size: 30px; color: var(--accent); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      padding: 0;
      border: 0;
      background: transparent;
      margin-bottom: 12px;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; }
    tr:last-child td { border-bottom: 0; }
    .empty { margin: 0; color: var(--muted); }"""
