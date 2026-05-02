from __future__ import annotations

import pytest

from graph.rag import estimate_reading_time
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    content: str = "",
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
    )


def test_estimate_reading_time_counts_prose_and_metadata_summary():
    result = estimate_reading_time(
        [
            unit(
                "alpha",
                "Alpha title",
                "One two three four.",
                metadata={"summary": "Summary adds two."},
                tags=["planning"],
            )
        ],
        words_per_minute=5,
    )

    assert result["units"] == [
        {
            "unit_id": "alpha",
            "title": "Alpha title",
            "word_count": 9,
            "estimated_minutes": 1.8,
            "tags": ["planning"],
        }
    ]
    assert result["totals"] == {
        "unit_count": 1,
        "word_count": 9,
        "estimated_minutes": 1.8,
    }
    assert result["settings"] == {"words_per_minute": 5, "include_code": True}


def test_estimate_reading_time_handles_empty_content():
    result = estimate_reading_time([unit("empty", "Only title")], words_per_minute=10)

    assert result["units"][0]["word_count"] == 2
    assert result["units"][0]["estimated_minutes"] == 0.2
    assert result["totals"]["word_count"] == 2


def test_estimate_reading_time_can_exclude_fenced_and_code_like_blocks():
    code_heavy = unit(
        "code",
        "Code note",
        """Read this prose.

```python
def build_widget(input_value):
    return input_value + 1
```

    indented helper words
const value = runThing();
Final prose line.""",
    )

    with_code = estimate_reading_time(
        [code_heavy], words_per_minute=10, include_code=True
    )
    without_code = estimate_reading_time(
        [code_heavy], words_per_minute=10, include_code=False
    )

    assert with_code["units"][0]["word_count"] == 23
    assert without_code["units"][0]["word_count"] == 8
    assert without_code["units"][0]["estimated_minutes"] == 0.8


def test_estimate_reading_time_totals_and_order_preservation():
    units = [
        unit("second", "Second unit", "three four"),
        unit("first", "First unit", "three"),
    ]

    result = estimate_reading_time(units, words_per_minute=4)

    assert [entry["unit_id"] for entry in result["units"]] == ["second", "first"]
    assert [entry["word_count"] for entry in result["units"]] == [4, 3]
    assert result["totals"] == {
        "unit_count": 2,
        "word_count": 7,
        "estimated_minutes": 1.75,
    }


def test_estimate_reading_time_is_importable_from_graph_rag():
    from graph.rag import estimate_reading_time as imported

    assert imported is estimate_reading_time


@pytest.mark.parametrize("words_per_minute", [0, -1, 1.5, True])
def test_estimate_reading_time_validates_words_per_minute(words_per_minute):
    with pytest.raises(ValueError, match="words_per_minute must be a positive integer"):
        estimate_reading_time([], words_per_minute=words_per_minute)
