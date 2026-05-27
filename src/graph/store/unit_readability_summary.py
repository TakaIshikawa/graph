"""Summarize unit readability by source."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_LONG_SENTENCE_WORDS = 25


def summarize_unit_readability(units: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        grouped[_source(unit)].append(unit)

    rows = [_row(source, grouped[source]) for source in sorted(grouped, key=sort_key)]
    return {"total_units": total_units, "rows": rows, "source_summaries": rows}


def _row(source: str, units: list[Any]) -> dict[str, Any]:
    total_words = total_sentences = long_sentence_units = question_sentences = 0
    readable: list[tuple[int, str]] = []

    for index, unit in enumerate(units):
        sentences = _sentences(_content(unit))
        sentence_word_counts = [_word_count(sentence) for sentence in sentences]
        words = sum(sentence_word_counts)
        total_words += words
        total_sentences += len(sentences)
        question_sentences += sum(1 for sentence in sentences if sentence.rstrip().endswith("?"))
        if any(count >= _LONG_SENTENCE_WORDS for count in sentence_word_counts):
            long_sentence_units += 1
        if words:
            readable.append((words, unit_id(unit) or str(index)))

    shortest = sorted(readable, key=lambda item: (item[0], sort_key(item[1])))[0][1] if readable else ""
    return {
        "source": source,
        "unit_count": len(units),
        "average_words_per_sentence": f"{(total_words / total_sentences) if total_sentences else 0:.2f}",
        "long_sentence_unit_count": long_sentence_units,
        "question_sentence_count": question_sentences,
        "shortest_readable_unit_id": shortest,
    }


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"


def _content(unit: Any) -> str:
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)


def _sentences(content: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(content) if _word_count(match.group(0))]


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))
