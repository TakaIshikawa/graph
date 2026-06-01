from graph.store.unit_markdown_link_target_extension_summary import summarize_unit_markdown_link_target_extensions


def test_link_target_extension_summary_classifies_local_external_and_extensionless_targets():
    summary = summarize_unit_markdown_link_target_extensions(
        [
            {"id": "b", "content": "[Doc](notes/Readme.MD?x=1#top) ![Img](img/photo.PNG) [Web](https://example.com/a.pdf)"},
            {"id": "a", "content": "[No ext](/path/page) [Frag](#local)\n```\n[Skip](x.txt)\n```"},
        ],
        sample_limit=3,
    )

    assert summary["total_units"] == 2
    assert summary["target_count"] == 5
    assert summary["extension_counts"] == {"md": 1, "png": 1}
    assert summary["extensionless_count"] == 2
    assert summary["external_url_count"] == 1
    assert summary["local_path_count"] == 4
    assert summary["examples"] == [
        {"unit_id": "a", "line": 1, "target": "#local", "extension": "", "is_image": False},
        {"unit_id": "a", "line": 1, "target": "/path/page", "extension": "", "is_image": False},
        {"unit_id": "b", "line": 1, "target": "https://example.com/a.pdf", "extension": "", "is_image": False},
    ]
