"""Summarize WWW-Authenticate challenges in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "www-authenticate"
_KNOWN_SCHEMES = {"basic", "bearer", "digest", "negotiate"}


def summarize_source_www_authenticate_challenges(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    scheme_counts: Counter[str] = Counter()
    realm_presence_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    sources_with = empty_value_count = malformed_challenge_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        found, raw_values = _lookup_header(source, _HEADER)
        if not found:
            continue
        values = [field_value(value).strip() for value in raw_values if field_value(value).strip()]
        if not values:
            empty_value_count += 1
            continue

        challenges = [challenge for value in values for challenge in _parse_challenges(value)]
        valid = [challenge for challenge in challenges if challenge["scheme"]]
        malformed_count = len(challenges) - len(valid)
        if not valid:
            malformed_challenge_count += max(1, malformed_count)
            rows.append({"source_id": sid, "schemes": [], "realm_schemes": [], "malformed_challenge_count": max(1, malformed_count)})
            continue

        sources_with += 1
        schemes = [challenge["scheme"] for challenge in valid]
        realm_schemes = [challenge["scheme"] for challenge in valid if challenge["has_realm"]]
        scheme_counts.update(schemes)
        realm_presence_counts.update(realm_schemes)
        malformed_challenge_count += malformed_count
        row = {
            "source_id": sid,
            "schemes": sorted(dict.fromkeys(schemes), key=sort_key),
            "realm_schemes": sorted(dict.fromkeys(realm_schemes), key=sort_key),
            "malformed_challenge_count": malformed_count,
        }
        rows.append(row)
        if len(samples) < limit:
            samples.append({"source_id": sid, "schemes": row["schemes"]})

    rows.sort(key=lambda row: sort_key(row["source_id"]))
    samples = sorted(samples, key=lambda row: sort_key(row["source_id"]))[:limit]
    return {
        "total_sources": len(source_list),
        "sources_with_www_authenticate": sources_with,
        "missing_header_count": len(source_list) - sources_with - empty_value_count - sum(1 for row in rows if not row["schemes"]),
        "empty_value_count": empty_value_count,
        "scheme_counts": {key: scheme_counts[key] for key in sorted(scheme_counts, key=sort_key)},
        "realm_presence_counts": {key: realm_presence_counts[key] for key in sorted(realm_presence_counts, key=sort_key)},
        "bearer_count": scheme_counts["bearer"],
        "basic_count": scheme_counts["basic"],
        "digest_count": scheme_counts["digest"],
        "negotiate_count": scheme_counts["negotiate"],
        "unknown_scheme_count": sum(count for scheme, count in scheme_counts.items() if scheme not in _KNOWN_SCHEMES),
        "malformed_challenge_count": malformed_challenge_count,
        "rows": rows,
        "samples": samples,
    }


def summarize_source_www_authenticate_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Compatibility wrapper for the older header summary name."""
    return summarize_source_www_authenticate_challenges(sources, sample_limit=sample_limit)


def _parse_challenges(value: str) -> list[dict[str, Any]]:
    parts = _split_quoted_commas(value)
    challenges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for part in parts:
        if not part:
            continue
        token, rest = _first_token(part)
        if _starts_challenge(token, rest):
            current = {"scheme": token.casefold().rstrip(":"), "has_realm": _has_realm(rest)}
            challenges.append(current)
        elif current is not None and _looks_like_parameter(part):
            current["has_realm"] = bool(current["has_realm"] or _has_realm(part))
        else:
            challenges.append({"scheme": "", "has_realm": False})
            current = None
    return challenges or [{"scheme": "", "has_realm": False}]


def _split_quoted_commas(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    in_quote = False
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_quote:
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if char == "," and not in_quote:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


def _first_token(value: str) -> tuple[str, str]:
    pieces = value.strip().split(None, 1)
    if not pieces:
        return "", ""
    return pieces[0], pieces[1] if len(pieces) > 1 else ""


def _starts_challenge(token: str, rest: str) -> bool:
    clean = token.rstrip(":")
    return bool(clean and clean.replace("-", "").isalnum() and "=" not in clean and (not rest or not token.casefold().startswith("realm=")))


def _looks_like_parameter(value: str) -> bool:
    token, _rest = _first_token(value)
    name, separator, remainder = token.partition("=")
    return bool(separator and name and remainder and "=" not in remainder)


def _has_realm(value: str) -> bool:
    return any(part.strip().casefold().startswith("realm=") for part in _split_quoted_commas(value))


def _values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [field_value(item) for item in value]
    return [field_value(value)]


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> tuple[bool, list[str]]:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            raw = get(container, key) if container_name == "source" else container.get(key)
            if raw is not None:
                return True, _values(raw)
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return True, _values(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return True, _values(value)
    return False, []
