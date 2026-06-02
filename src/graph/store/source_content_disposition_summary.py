"""Summarize Content-Disposition headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import unquote

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "content-disposition"


def summarize_source_content_dispositions(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    disposition_type_counts: Counter[str] = Counter()
    filename_extension_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        raw = _lookup_header(source, _HEADER)
        if not raw:
            continue
        sources_with += 1
        disposition_type, filename = _parse_content_disposition(raw)
        disposition_type_counts[disposition_type] += 1
        extension = _filename_extension(filename)
        if extension:
            filename_extension_counts[extension] += 1
        if len(samples) < limit:
            samples.append({"source_id": sid, "disposition_type": disposition_type, "filename": filename, "raw": raw})

    return {
        "total_sources": len(source_list),
        "sources_with_content_disposition": sources_with,
        "missing_content_disposition_count": len(source_list) - sources_with,
        "disposition_type_counts": {key: disposition_type_counts[key] for key in sorted(disposition_type_counts, key=sort_key)},
        "filename_extension_counts": {key: filename_extension_counts[key] for key in sorted(filename_extension_counts, key=sort_key)},
        "samples": samples,
    }


def _parse_content_disposition(value: str) -> tuple[str, str]:
    parts = [field_value(part) for part in value.split(";") if field_value(part)]
    disposition_type = parts[0].casefold() if parts else "unknown"
    params: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        params[key.strip().casefold()] = raw_value.strip().strip("\"'")
    filename = _decode_filename_star(params.get("filename*") or "") or field_value(params.get("filename"))
    return disposition_type or "unknown", filename


def _decode_filename_star(value: str) -> str:
    if not value:
        return ""
    pieces = value.split("'", 2)
    encoded = pieces[2] if len(pieces) == 3 else value
    try:
        return field_value(unquote(encoded))
    except Exception:
        return field_value(encoded)


def _filename_extension(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in name:
        return ""
    extension = name.rsplit(".", 1)[-1].strip().casefold()
    return f".{extension}" if extension else ""


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
