from types import SimpleNamespace

from graph.store import summarize_collection_duplicate_titles


def test_collection_duplicate_titles_groups_title_name_and_label_aliases():
    report = summarize_collection_duplicate_titles(
        [
            {"id": "a", "title": "  Quarterly   Plan "},
            {"id": "b", "name": "quarterly plan"},
            {"id": "c", "metadata": {"label": "Quarterly Plan"}},
            {"id": "d", "title": "Unique"},
            {"id": "blank", "title": "   "},
            {"id": "missing"},
        ]
    )

    assert report == {
        "collection_count": 6,
        "duplicate_title_group_count": 1,
        "duplicate_collection_count": 3,
        "groups": [
            {
                "normalized_title": "quarterly plan",
                "collection_ids": ["a", "b", "c"],
                "titles": ["Quarterly   Plan", "quarterly plan", "Quarterly Plan"],
            }
        ],
    }


def test_collection_duplicate_titles_supports_objects_and_ignores_blank_titles():
    report = summarize_collection_duplicate_titles(
        [
            SimpleNamespace(id="a", title="Roadmap"),
            SimpleNamespace(id="b", metadata={"name": " roadmap "}),
            SimpleNamespace(id="c", label=" "),
        ]
    )

    assert report["groups"] == [{"normalized_title": "roadmap", "collection_ids": ["a", "b"], "titles": ["Roadmap", "roadmap"]}]
