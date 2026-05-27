from graph.store import summarize_unit_task_inventory


def test_task_inventory_summary_counts_states_tags_and_overdue_samples():
    content = "- [ ] open #work due:2000-01-01\n- [x] done #work\n- [?] maybe #later\n```\n- [ ] hidden\n```"
    summary = summarize_unit_task_inventory([{"id": "u1", "content": content}])

    assert summary["total_tasks"] == 3
    assert summary["completed_count"] == 1
    assert summary["open_count"] == 1
    assert summary["unknown_count"] == 1
    assert summary["top_tags"][0] == {"tag": "work", "count": 2}
    assert summary["overdue_due_date_samples"][0]["unit_id"] == "u1"
