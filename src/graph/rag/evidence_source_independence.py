"""Analyze independence of evidence sources."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from graph.rag._record_text import first, record_id


def analyze_evidence_source_independence(evidence: Iterable[Any] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    records = list(evidence or ())
    for index, item in enumerate(records):
        groups[_source_key(item) or f"unknown-{index + 1}"].append(record_id(item, index, "evidence"))
    sizes = [len(members) for members in groups.values()]
    largest = max(sizes, default=0)
    duplicate_groups = {key: members for key, members in sorted(groups.items()) if len(members) > 1}
    return {
        "record_count": len(records),
        "independent_source_count": len(groups),
        "duplicate_source_group_count": len(duplicate_groups),
        "largest_group_size": largest,
        "concentration_ratio": round(largest / len(records), 3) if records else 0.0,
        "groups": dict(sorted(groups.items())),
        "samples": [{"source": key, "members": members} for key, members in list(duplicate_groups.items())[:sample_limit]],
    }


def _source_key(item: Any) -> str | None:
    for key in ("url", "domain", "source", "author"):
        value = first(item, (key,))
        if not value:
            continue
        text = str(value).strip().lower()
        if key == "url" or "://" in text:
            parsed = urlparse(text if "://" in text else f"//{text}")
            text = parsed.netloc or parsed.path.split("/")[0]
        return text.removeprefix("www.")
    return None
