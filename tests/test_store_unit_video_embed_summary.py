from __future__ import annotations

from graph.store import summarize_unit_video_embeds


def test_summarize_unit_video_embeds_groups_provider_and_mode():
    summary = summarize_unit_video_embeds(
        [
            {"id": "a", "content": '<iframe src="https://www.youtube.com/embed/abc"></iframe>'},
            {"id": "b", "content": "[clip](https://vimeo.com/123) https://youtu.be/xyz"},
            {"id": "c", "metadata": {"url": "media/demo.mp4"}},
        ]
    )

    assert summary["total_units"] == 3
    assert summary["units_with_video"] == 3
    assert summary["embedded_count"] == 1
    assert summary["linked_count"] == 3
    assert summary["provider_counts"] == [
        {"provider": "local", "count": 1},
        {"provider": "vimeo", "count": 1},
        {"provider": "youtube", "count": 2},
    ]
