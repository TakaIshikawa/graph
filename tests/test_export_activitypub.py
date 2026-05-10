from __future__ import annotations

import json

from graph.exports.activitypub import export_units_to_activitypub
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _make_unit(
    title: str = "Test Unit",
    content: str = "Test content",
    tags: list[str] | None = None,
    **kwargs,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=SourceProject.TODOIST,
        source_id=f"test:{title.lower().replace(' ', '_')}",
        source_entity_type="task",
        title=title,
        content=content,
        content_type=ContentType.ARTIFACT,
        tags=tags or [],
        **kwargs,
    )


def test_returns_valid_json():
    units = [_make_unit()]
    result = export_units_to_activitypub(units)
    data = json.loads(result)
    assert isinstance(data, dict)


def test_ordered_collection_structure():
    units = [_make_unit("A"), _make_unit("B")]
    data = json.loads(export_units_to_activitypub(units))

    assert data["type"] == "OrderedCollection"
    assert data["totalItems"] == 2
    assert len(data["orderedItems"]) == 2


def test_note_fields():
    unit = _make_unit(title="My Note", content="Hello <world>")
    data = json.loads(export_units_to_activitypub([unit]))

    note = data["orderedItems"][0]
    assert note["type"] == "Note"
    assert note["name"] == "My Note"
    # Content should be HTML-escaped
    assert "&lt;" in note["content"] or "Hello" in note["content"]


def test_tags_become_hashtag_objects():
    unit = _make_unit(tags=["python", "coding"])
    data = json.loads(export_units_to_activitypub([unit]))

    note = data["orderedItems"][0]
    assert "tag" in note
    assert len(note["tag"]) == 2
    assert note["tag"][0]["type"] == "Hashtag"
    assert note["tag"][0]["name"] == "#python"


def test_source_has_plaintext():
    unit = _make_unit(content="Plain text here")
    data = json.loads(export_units_to_activitypub([unit]))

    note = data["orderedItems"][0]
    assert note["source"]["content"] == "Plain text here"
    assert note["source"]["mediaType"] == "text/plain"


def test_custom_context_url():
    data = json.loads(export_units_to_activitypub(
        [_make_unit()],
        context_url="https://custom.example.com",
    ))
    assert data["@context"] == "https://custom.example.com"


def test_default_context_url():
    data = json.loads(export_units_to_activitypub([_make_unit()]))
    assert data["@context"] == "https://www.w3.org/ns/activitystreams"


def test_empty_units():
    data = json.loads(export_units_to_activitypub([]))
    assert data["totalItems"] == 0
    assert data["orderedItems"] == []


def test_registry_lookup():
    from graph.exports.registry import get_exporter

    func = get_exporter("activitypub")
    assert func is export_units_to_activitypub


def test_published_from_metadata():
    unit = _make_unit(metadata={"date": "2024-06-15"})
    data = json.loads(export_units_to_activitypub([unit]))

    note = data["orderedItems"][0]
    assert note["published"] == "2024-06-15"
