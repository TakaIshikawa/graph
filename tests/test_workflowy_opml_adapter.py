from __future__ import annotations

from graph.adapters.registry import get_adapter
from graph.adapters.workflowy_opml import WorkflowyOpmlAdapter


def test_workflowy_opml_emits_nested_bullets_with_parent_metadata(tmp_path):
    path = tmp_path / "workflowy.opml"
    path.write_text("""<opml><body><outline text="Parent" _note="Note"><outline text="Child" _complete="true"/></outline></body></opml>""", encoding="utf-8")

    units = WorkflowyOpmlAdapter(str(path)).ingest().units
    parent = next(unit for unit in units if unit.title == "Parent")
    child = next(unit for unit in units if unit.title == "Child")

    assert parent.metadata["child_count"] == 1
    assert child.metadata["depth"] == 2
    assert child.metadata["parent_path"] == "1"
    assert child.metadata["completed"] is True
    assert child.metadata["sibling_order"] == 1
    assert isinstance(get_adapter("workflowy_opml"), WorkflowyOpmlAdapter)
