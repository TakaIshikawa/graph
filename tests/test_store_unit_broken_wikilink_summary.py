from __future__ import annotations

from types import SimpleNamespace

from graph.store.unit_broken_wikilink_summary import summarize_unit_broken_wikilinks


def test_unit_broken_wikilinks_resolves_alias_id_title_and_slug():
    summary = summarize_unit_broken_wikilinks(
        [
            {"id": "u1", "title": "Source", "content": "[[Target Page|label]] [[u2]] [[target-page]] [[Missing]] [[Missing]]"},
            {"id": "u2", "title": "Target Page", "slug": "target-page", "content": ""},
        ]
    )

    assert summary == {
        "total_units": 2,
        "total_wikilinks": 5,
        "broken_link_count": 1,
        "rows": [{"unit_id": "u1", "missing_targets": ["Missing"], "missing_count": 1}],
    }


def test_unit_broken_wikilinks_supports_objects_and_metadata_content():
    summary = summarize_unit_broken_wikilinks([SimpleNamespace(unit_id="a", metadata={"content": "[[Nope]]"})])

    assert summary["rows"] == [{"unit_id": "a", "missing_targets": ["Nope"], "missing_count": 1}]
