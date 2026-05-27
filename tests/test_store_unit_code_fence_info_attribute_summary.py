from __future__ import annotations

from graph.store.unit_code_fence_info_attribute_summary import summarize_unit_code_fence_info_attributes


def test_summarize_unit_code_fence_info_attributes_parses_common_attributes():
    summary = summarize_unit_code_fence_info_attributes(
        [{"id": "u1", "content": "```python {.demo #ex title=\"Example\" filename=app.py}\npass\n```\n~~~js file=index.js\n~~~"}]
    )

    assert summary["total_fences_with_attributes"] == 2
    assert summary["attribute_counts"] == [
        {"attribute": "filename", "count": 2},
        {"attribute": "class", "count": 1},
        {"attribute": "id", "count": 1},
        {"attribute": "title", "count": 1},
    ]
    assert summary["language_counts"] == [{"language": "js", "count": 1}, {"language": "python", "count": 1}]
    assert summary["examples"][0] == {"unit_id": "u1", "line": 1, "language": "python", "attributes": ["class", "filename", "id", "title"]}
