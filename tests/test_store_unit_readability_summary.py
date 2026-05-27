from __future__ import annotations

from dataclasses import dataclass

from graph.store.unit_readability_summary import summarize_unit_readability


@dataclass
class Unit:
    id: str
    content: str
    source_project: str
    metadata: dict[str, str] | None = None


def test_summarize_unit_readability_groups_counts_and_shortest_readable_unit():
    summary = summarize_unit_readability(
        [
            {
                "id": "long",
                "source_project": "notes",
                "content": "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty one two three four five.",
            },
            {"id": "short", "source_project": "notes", "content": "Tiny note. Why now?"},
            {"id": "empty", "source_project": "notes", "content": ""},
        ]
    )

    assert summary["total_units"] == 3
    assert summary["rows"] == [
        {
            "source": "notes",
            "unit_count": 3,
            "average_words_per_sentence": "9.67",
            "long_sentence_unit_count": 1,
            "question_sentence_count": 1,
            "shortest_readable_unit_id": "short",
        }
    ]
    assert summary["source_summaries"] is summary["rows"]


def test_summarize_unit_readability_supports_objects_metadata_source_and_sorted_rows():
    summary = summarize_unit_readability(
        [
            Unit(id="b", source_project="", metadata={"source": "Beta"}, content="Hello there."),
            {"id": "a", "metadata": {"source": "alpha"}, "content": ""},
        ]
    )

    assert summary["rows"][0]["source"] == "alpha"
    assert summary["rows"][0]["average_words_per_sentence"] == "0.00"
    assert summary["rows"][0]["shortest_readable_unit_id"] == ""
    assert summary["rows"][1]["source"] == "Beta"
    assert summary["rows"][1]["average_words_per_sentence"] == "2.00"
    assert summary["rows"][1]["shortest_readable_unit_id"] == "b"
