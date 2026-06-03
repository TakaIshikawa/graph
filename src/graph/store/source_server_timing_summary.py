"""Summarize Server-Timing headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import re
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "server-timing"
_METRIC_NAME_RE = re.compile(r"^[a-z0-9_.:-]+$")


def summarize_source_server_timings(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    metric_counts: Counter[str] = Counter()
    rows_by_metric: dict[str, dict[str, Any]] = {}
    duration_buckets: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    sample_order = 0
    sources_with = missing_duration_count = invalid_duration_count = description_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        metrics = _metrics(value)
        if metrics:
            sources_with += 1
        for metric in metrics:
            name = metric["name"]
            metric_counts[name] += 1
            row = rows_by_metric.setdefault(name, {"metric": name, "count": 0, "source_ids": [], "examples": []})
            row["count"] += 1
            if sid not in row["source_ids"] and len(row["source_ids"]) < limit:
                row["source_ids"].append(sid)
            if metric["raw"] not in row["examples"] and len(row["examples"]) < limit:
                row["examples"].append(metric["raw"])
            dur = metric["params"].get("dur")
            if dur is None:
                missing_duration_count += 1
            else:
                try:
                    duration_buckets[_duration_bucket(float(dur))] += 1
                except ValueError:
                    invalid_duration_count += 1
            desc = field_value(metric["params"].get("desc"))
            if desc:
                description_count += 1
            sample: dict[str, Any] = {"source_id": sid, "metric": name, "value": metric["raw"], "_order": sample_order}
            sample_order += 1
            if dur is not None:
                sample["dur"] = field_value(dur)
            if desc:
                sample["desc"] = desc
            samples.append(sample)

    samples.sort(key=lambda row: (sort_key(row["source_id"]), row["_order"]))
    bounded_samples = samples[:limit]
    for sample in bounded_samples:
        sample.pop("_order", None)
    rows = [rows_by_metric[key] for key in sorted(rows_by_metric, key=sort_key)]
    return {
        "total_sources": len(source_list),
        "sources_with_server_timing": sources_with,
        "metric_counts": {key: metric_counts[key] for key in sorted(metric_counts, key=sort_key)},
        "rows": rows,
        "duration_buckets": {key: duration_buckets[key] for key in ("lt_100ms", "100_499ms", "500_999ms", "gte_1000ms") if duration_buckets[key]},
        "missing_duration_count": missing_duration_count,
        "invalid_duration_count": invalid_duration_count,
        "description_count": description_count,
        "missing_server_timing_count": len(source_list) - sources_with,
        "samples": bounded_samples,
    }


def _metrics(value: str) -> list[dict[str, Any]]:
    metrics = []
    for entry in _split_quoted(value, ","):
        raw = entry.strip()
        if not raw:
            continue
        parts = _split_quoted(raw, ";")
        name = field_value(parts[0]).casefold()
        if not name or not _METRIC_NAME_RE.fullmatch(name):
            continue
        params = {}
        for part in parts[1:]:
            key, sep, raw_value = part.partition("=")
            if sep:
                params[field_value(key).casefold()] = _unquote(raw_value)
        metrics.append({"name": name, "params": params, "raw": raw})
    return metrics


def _duration_bucket(value: float) -> str:
    if value < 100:
        return "lt_100ms"
    if value < 500:
        return "100_499ms"
    if value < 1000:
        return "500_999ms"
    return "gte_1000ms"


def _split_quoted(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote = False
    escape = False
    for char in value:
        if escape:
            buf.append(char)
            escape = False
            continue
        if char == "\\" and quote:
            buf.append(char)
            escape = True
            continue
        if char == '"':
            quote = not quote
        if char == delimiter and not quote:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return parts


def _unquote(value: str) -> str:
    text = field_value(value)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1].replace('\\"', '"')
    return text


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
