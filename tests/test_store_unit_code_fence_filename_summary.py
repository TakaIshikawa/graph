from graph.store import summarize_unit_code_fence_filenames


def test_code_fence_filename_summary_counts_extensions_and_duplicates():
    summary = summarize_unit_code_fence_filenames([{"id": "u1", "content": "```python filename=app.py\nx\n```\n`file=inline.py`\n```js title=app.py\n```"}])

    assert summary["filename_fence_count"] == 2
    assert summary["extension_counts"] == [{"extension": "py", "count": 2}]
    assert summary["duplicate_filename_counts"] == [{"filename": "app.py", "count": 2}]
