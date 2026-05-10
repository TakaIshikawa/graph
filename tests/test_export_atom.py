from __future__ import annotations

import xml.etree.ElementTree as ET

from graph.export.atom import export_units_to_atom
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

ATOM_NS = "http://www.w3.org/2005/Atom"


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


def _parse_atom(xml_str: str) -> ET.Element:
    return ET.fromstring(xml_str)


def _ns(tag: str) -> str:
    return f"{{{ATOM_NS}}}{tag}"


def test_single_unit_produces_valid_xml():
    xml = export_units_to_atom(_make_unit())
    root = _parse_atom(xml)
    assert root.tag == _ns("feed")


def test_feed_has_title_and_id():
    xml = export_units_to_atom(
        [_make_unit()],
        feed_title="My Feed",
        feed_id="urn:test:feed",
    )
    root = _parse_atom(xml)
    assert root.find(_ns("title")).text == "My Feed"
    assert root.find(_ns("id")).text == "urn:test:feed"


def test_entries_created_for_each_unit():
    units = [_make_unit("A"), _make_unit("B"), _make_unit("C")]
    root = _parse_atom(export_units_to_atom(units))
    entries = root.findall(_ns("entry"))
    assert len(entries) == 3


def test_entry_has_title_and_content():
    xml = export_units_to_atom(_make_unit(title="My Title", content="My content"))
    root = _parse_atom(xml)
    entry = root.find(_ns("entry"))
    assert entry.find(_ns("title")).text == "My Title"
    assert entry.find(_ns("content")).text == "My content"
    assert entry.find(_ns("content")).get("type") == "html"


def test_tags_become_categories():
    xml = export_units_to_atom(_make_unit(tags=["python", "coding"]))
    root = _parse_atom(xml)
    entry = root.find(_ns("entry"))
    cats = entry.findall(_ns("category"))
    terms = {c.get("term") for c in cats}
    assert "python" in terms
    assert "coding" in terms


def test_empty_units():
    xml = export_units_to_atom([])
    root = _parse_atom(xml)
    entries = root.findall(_ns("entry"))
    assert len(entries) == 0


def test_writes_to_file(tmp_path):
    out = tmp_path / "feed.xml"
    xml = export_units_to_atom([_make_unit()], path=str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == xml


def test_function_is_importable():
    """Verify the export function can be imported directly."""
    from graph.export.atom import export_units_to_atom as fn
    assert fn is export_units_to_atom
