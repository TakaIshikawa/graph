from __future__ import annotations

from types import SimpleNamespace

from graph.store.unit_code_block_language_summary import summarize_unit_code_block_languages


def test_code_block_language_summary_counts_string_mapping_and_object_units():
    report = summarize_unit_code_block_languages(
        [
            "Intro\n```Python\nprint('a')\n```\n``` js\nx()\n```",
            {"id": "m1", "content": "No language\n```\nplain\n```"},
            SimpleNamespace(id="o1", content="~~~PYTHON\nprint('b')\n~~~"),
            {"id": "empty", "content": None},
        ]
    )

    assert report == {
        "total_units": 4,
        "units_with_code_blocks": 3,
        "total_code_blocks": 4,
        "missing_language_count": 1,
        "language_counts": [{"language": "js", "count": 1}, {"language": "python", "count": 2}],
    }


def test_code_block_language_summary_handles_empty_inputs():
    report = summarize_unit_code_block_languages([{}, SimpleNamespace(metadata={"content": ""})])

    assert report["total_units"] == 2
    assert report["units_with_code_blocks"] == 0
    assert report["total_code_blocks"] == 0
    assert report["language_counts"] == []
