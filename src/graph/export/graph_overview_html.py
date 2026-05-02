"""Self-contained HTML overview reports for graph summaries."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone


def render_graph_overview_html(summary: dict) -> str:
    """Render a standalone HTML graph overview report.

    The summary is intentionally a plain dictionary so callers can pass audit
    snapshots without depending on graph model classes. All interpolated
    summary values are escaped before rendering.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    counts = _summary_mapping(summary.get("counts"))
    top_tags = _ranked_items(summary.get("top_tags"), label_keys=("tag", "name", "id"))
    top_sources = _ranked_items(
        summary.get("top_sources"),
        label_keys=("source_project", "source", "name", "id"),
    )
    central_units = _central_units(summary.get("central_units"))
    components = _components(summary.get("components"))
    warnings = _text_items(summary.get("warnings"))

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "<title>Graph overview</title>",
            "<style>",
            _STYLES,
            "</style>",
            "</head>",
            "<body>",
            "<header>",
            "<h1>Graph Overview</h1>",
            "<div class=\"summary\">",
            f"<span>Generated: {_escape(generated_at)}</span>",
            f"<span>{_escape(len(counts))} count metrics</span>",
            f"<span>{_escape(len(warnings))} warnings</span>",
            "</div>",
            "</header>",
            "<main>",
            "<section class=\"panel\">",
            "<h2>Counts</h2>",
            _render_metric_grid(counts),
            "</section>",
            "<section class=\"panel\">",
            "<h2>Top Tags</h2>",
            _render_ranked_list(top_tags, empty_text="No top tags provided."),
            "</section>",
            "<section class=\"panel\">",
            "<h2>Top Sources</h2>",
            _render_ranked_list(top_sources, empty_text="No top sources provided."),
            "</section>",
            "<section class=\"panel\">",
            "<h2>Central Units</h2>",
            _render_central_units(central_units),
            "</section>",
            "<section class=\"panel\">",
            "<h2>Components</h2>",
            _render_components(components),
            "</section>",
            "<section class=\"panel warnings\">",
            "<h2>Warnings</h2>",
            _render_warnings(warnings),
            "</section>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _summary_mapping(value: object) -> list[tuple[str, object]]:
    if not isinstance(value, Mapping):
        return []
    return sorted(
        ((_text(key), item) for key, item in value.items()),
        key=lambda item: (item[0].casefold(), item[0]),
    )


def _ranked_items(value: object, *, label_keys: Sequence[str]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    if isinstance(value, Mapping):
        source: object = list(value.items())
    else:
        source = value
    if not _is_sequence(source):
        return []

    for entry in source:
        label: object = ""
        count: object = ""
        if isinstance(entry, Mapping):
            label = _first_present(entry, label_keys)
            count = _first_present(entry, ("count", "unit_count", "units", "value"))
        elif _is_sequence(entry):
            entry_values = list(entry)
            if len(entry_values) >= 2:
                label, count = entry_values[0], entry_values[1]
        items.append({"label": label, "count": count})

    return sorted(
        items,
        key=lambda item: (-_numeric_sort(item["count"]), _text(item["label"]).casefold(), _text(item["label"])),
    )


def _central_units(value: object) -> list[Mapping[str, object]]:
    if not _is_sequence(value):
        return []
    records = [entry for entry in value if isinstance(entry, Mapping)]
    return sorted(
        records,
        key=lambda item: (
            -_numeric_sort(_first_present(item, ("score", "centrality", "degree"))),
            _text(_first_present(item, ("id", "unit_id", "title"))).casefold(),
            _text(_first_present(item, ("id", "unit_id", "title"))),
        ),
    )


def _components(value: object) -> list[Mapping[str, object]]:
    if not _is_sequence(value):
        return []
    records = [entry for entry in value if isinstance(entry, Mapping)]
    return sorted(
        records,
        key=lambda item: (
            -_numeric_sort(_first_present(item, ("unit_count", "units", "size"))),
            _text(_first_present(item, ("id", "component_id", "name"))).casefold(),
            _text(_first_present(item, ("id", "component_id", "name"))),
        ),
    )


def _text_items(value: object) -> list[str]:
    if isinstance(value, Mapping):
        return [_text(f"{key}: {item}") for key, item in sorted(value.items())]
    if _is_sequence(value):
        return sorted((_text(item) for item in value), key=lambda item: (item.casefold(), item))
    text = _text(value)
    return [text] if text else []


def _render_metric_grid(items: list[tuple[str, object]]) -> str:
    if not items:
        return "<p class=\"empty\">No counts provided.</p>"
    metrics = "".join(
        "\n".join(
            [
                "<div class=\"metric\">",
                f"<span>{_escape(label)}</span>",
                f"<strong>{_escape(value)}</strong>",
                "</div>",
            ]
        )
        for label, value in items
    )
    return f"<div class=\"metrics\">{metrics}</div>"


def _render_ranked_list(items: list[dict[str, object]], *, empty_text: str) -> str:
    if not items:
        return f"<p class=\"empty\">{_escape(empty_text)}</p>"
    rows = "".join(
        f"<li><span>{_escape(item['label'])}</span><strong>{_escape(item['count'])}</strong></li>"
        for item in items
    )
    return f"<ol class=\"ranked\">{rows}</ol>"


def _render_central_units(units: list[Mapping[str, object]]) -> str:
    if not units:
        return "<p class=\"empty\">No central units provided.</p>"
    cards = []
    for unit in units:
        tags = _text_list(unit.get("tags"))
        tag_html = "".join(f"<span class=\"tag\">{_escape(tag)}</span>" for tag in tags)
        if not tag_html:
            tag_html = "<span class=\"muted\">No tags</span>"
        cards.append(
            "\n".join(
                [
                    "<article class=\"record\">",
                    f"<h3>{_escape(_first_present(unit, ('title', 'name', 'id', 'unit_id')) or 'Untitled')}</h3>",
                    "<dl>",
                    f"<dt>ID</dt><dd>{_escape(_first_present(unit, ('id', 'unit_id')))}</dd>",
                    f"<dt>Score</dt><dd>{_escape(_first_present(unit, ('score', 'centrality')))}</dd>",
                    f"<dt>Degree</dt><dd>{_escape(unit.get('degree'))}</dd>",
                    f"<dt>Source</dt><dd>{_escape(_first_present(unit, ('source_project', 'source')))}</dd>",
                    "</dl>",
                    f"<div class=\"tags\">{tag_html}</div>",
                    "</article>",
                ]
            )
        )
    return "".join(cards)


def _render_components(components: list[Mapping[str, object]]) -> str:
    if not components:
        return "<p class=\"empty\">No components provided.</p>"
    cards = []
    for component in components:
        samples = _text_list(_first_present(component, ("sample_units", "units", "examples")))
        sample_html = "".join(f"<li>{_escape(sample)}</li>" for sample in samples)
        if not sample_html:
            sample_html = "<li class=\"muted\">No sample units</li>"
        cards.append(
            "\n".join(
                [
                    "<article class=\"record\">",
                    f"<h3>{_escape(_first_present(component, ('id', 'component_id', 'name')) or 'Component')}</h3>",
                    "<dl>",
                    f"<dt>Units</dt><dd>{_escape(_first_present(component, ('unit_count', 'size')))}</dd>",
                    f"<dt>Edges</dt><dd>{_escape(component.get('edge_count'))}</dd>",
                    f"<dt>Density</dt><dd>{_escape(component.get('density'))}</dd>",
                    "</dl>",
                    "<h4>Sample Units</h4>",
                    f"<ul class=\"samples\">{sample_html}</ul>",
                    "</article>",
                ]
            )
        )
    return "".join(cards)


def _render_warnings(warnings: list[str]) -> str:
    if not warnings:
        return "<p class=\"empty\">No warnings provided.</p>"
    items = "".join(f"<li>{_escape(warning)}</li>" for warning in warnings)
    return f"<ul class=\"warning-list\">{items}</ul>"


def _first_present(record: Mapping[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        if key in record:
            return record[key]
    return ""


def _text_list(value: object) -> list[str]:
    if not _is_sequence(value):
        text = _text(value)
        return [text] if text else []
    return sorted({_text(item) for item in value if _text(item)}, key=lambda item: (item.casefold(), item))


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _numeric_sort(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _escape(value: object) -> str:
    return html.escape(_text(value), quote=True)


_STYLES = """
:root { color-scheme: light; --fg: #202124; --muted: #5f6368; --line: #d9dde3; --bg: #f6f7f9; --accent: #0b57d0; --warn: #9a3412; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.45; }
header { background: #fff; border-bottom: 1px solid var(--line); padding: 28px max(24px, calc((100vw - 1040px) / 2)); }
main { max-width: 1040px; margin: 0 auto; padding: 24px; display: grid; gap: 18px; }
h1 { margin: 0 0 12px; font-size: 28px; font-weight: 650; }
h2 { margin: 0 0 14px; font-size: 19px; }
h3 { margin: 0 0 10px; font-size: 16px; }
h4 { margin: 12px 0 6px; font-size: 13px; color: var(--muted); text-transform: uppercase; }
.summary { display: flex; flex-wrap: wrap; gap: 10px 16px; color: var(--muted); font-size: 14px; }
.panel, .record { background: #fff; border: 1px solid var(--line); border-radius: 8px; }
.panel { padding: 18px; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
.metric { border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: #fbfcfe; }
.metric span, dt, .muted, .empty { color: var(--muted); }
.metric strong { display: block; margin-top: 6px; font-size: 22px; }
.ranked, .warning-list, .samples { margin: 0; padding-left: 22px; }
.ranked li { padding: 7px 0; }
.ranked span { display: inline-block; min-width: 180px; }
.record { padding: 14px; margin-top: 10px; }
dl { display: grid; grid-template-columns: 120px 1fr; gap: 6px 12px; margin: 0; }
dd { margin: 0; overflow-wrap: anywhere; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.tag { border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; font-size: 12px; background: #fbfcfe; }
.warning-list li { color: var(--warn); padding: 4px 0; }
""".strip()
