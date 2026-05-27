from __future__ import annotations

from graph.store.unit_code_fence_density_summary import summarize_unit_code_fence_density


def test_code_fence_density_summary_counts_backticks_tildes_languages_and_unterminated():
    report = summarize_unit_code_fence_density(
        [
            {"content": " ".join(["word"] * 500) + "\n```python\nx\n```"},
            {"content": "~~~js\nx\n~~~\n```sql\nselect 1"},
            {"content": "no code here"},
        ]
    )

    assert report["total_units"] == 3
    assert report["total_code_blocks"] == 3
    assert report["language_counts"] == [
        {"language": "js", "count": 1},
        {"language": "python", "count": 1},
        {"language": "sql", "count": 1},
    ]
    assert report["density_buckets"] == [
        {"bucket": "none", "count": 1},
        {"bucket": "low", "count": 1},
        {"bucket": "medium", "count": 0},
        {"bucket": "high", "count": 1},
    ]
