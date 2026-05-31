from graph.store import summarize_unit_markdown_task_due_dates


def test_task_due_date_summary_scans_task_lines_and_normalizes_dates():
    content = "- [ ] Task due:: 2026-05-01\n- [x] Done 📅 2026-06-01\nNot a task #due/2026-01-01\n- [ ] Later #due/2026-05-01"

    result = summarize_unit_markdown_task_due_dates([{"id": "u", "content": content}], as_of="2026-05-31")

    assert result["total_tasks"] == 3
    assert result["tasks_with_due_dates"] == 3
    assert result["overdue_count"] == 2
    assert result["date_counts"] == [{"date": "2026-05-01", "count": 2}, {"date": "2026-06-01", "count": 1}]
