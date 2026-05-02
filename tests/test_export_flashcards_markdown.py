from __future__ import annotations

from graph.export import export_units_to_flashcards_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    title: str = "Solar storage",
    content: str = "Batteries smooth evening demand.",
    tags: list[str] | None = None,
    metadata: dict | None = None,
    source_id: str | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        metadata=metadata or {},
    )


def test_export_flashcards_markdown_writes_title_content_cards_and_stats(tmp_path):
    path = tmp_path / "deck.md"

    stats = export_units_to_flashcards_markdown(
        [
            unit(
                "unit-a",
                tags=["storage", "Energy", "storage"],
            )
        ],
        path,
    )
    text = path.read_text(encoding="utf-8")

    assert stats == {
        "path": str(path),
        "units_exported": 1,
        "cards_exported": 1,
    }
    assert text == (
        "# Flashcards\n"
        "\n"
        "## Index\n"
        "\n"
        "- [Solar storage](#card-max-source-unit-a)\n"
        "\n"
        "## Cards\n"
        "\n"
        '<a id="card-max-source-unit-a"></a>\n'
        "\n"
        "### Solar storage\n"
        "\n"
        "**Question**\n"
        "\n"
        "```\n"
        "Solar storage\n"
        "```\n"
        "\n"
        "**Answer**\n"
        "\n"
        "```\n"
        "Batteries smooth evening demand.\n"
        "```\n"
        "\n"
        "**Tags**\n"
        "\n"
        "`Energy`, `storage`\n"
        "\n"
        "**Source**\n"
        "\n"
        "- Project: `max`\n"
        "- ID: `source-unit-a`\n"
    )


def test_export_flashcards_markdown_extracts_metadata_dotted_fields(tmp_path):
    path = tmp_path / "deck.md"
    units = [
        unit(
            "unit-a",
            title="Fallback question",
            content="Fallback answer",
            metadata={
                "study": {
                    "prompt": "What does storage smooth?",
                    "answer": "Evening demand.",
                }
            },
        )
    ]

    export_units_to_flashcards_markdown(
        units,
        path,
        question_field="metadata.study.prompt",
        answer_field="study.answer",
    )
    text = path.read_text(encoding="utf-8")

    assert "What does storage smooth?" in text
    assert "Evening demand." in text
    assert "Fallback question" not in text
    assert "Fallback answer" not in text


def test_export_flashcards_markdown_falls_back_when_metadata_fields_are_missing(tmp_path):
    path = tmp_path / "deck.md"

    export_units_to_flashcards_markdown(
        [unit("unit-a", title="Fallback question", content="Fallback answer")],
        path,
        question_field="metadata.study.prompt",
        answer_field="metadata.study.answer",
    )
    text = path.read_text(encoding="utf-8")

    assert "Fallback question" in text
    assert "Fallback answer" in text


def test_export_flashcards_markdown_can_omit_tags(tmp_path):
    path = tmp_path / "deck.md"

    export_units_to_flashcards_markdown(
        [unit("unit-a", tags=["storage"])],
        path,
        include_tags=False,
    )
    text = path.read_text(encoding="utf-8")

    assert "**Tags**" not in text
    assert "`storage`" not in text
    assert "**Source**" in text


def test_export_flashcards_markdown_escapes_headings_links_and_fences(tmp_path):
    path = tmp_path / "deck.md"

    export_units_to_flashcards_markdown(
        [
            unit(
                "unit-a/b",
                title="# API [draft](v2) \\ notes",
                content="Answer with ``` fenced text",
                source_id="source/unit-a",
            )
        ],
        path,
    )
    text = path.read_text(encoding="utf-8")

    assert r"[# API \[draft\]\(v2\) \\ notes](#card-max-source-unit-a)" in text
    assert r"### \# API [draft](v2) \\ notes" in text
    assert "````\nAnswer with ``` fenced text\n````" in text


def test_export_flashcards_markdown_empty_input_writes_empty_document(tmp_path):
    path = tmp_path / "empty.md"

    stats = export_units_to_flashcards_markdown([], path)

    assert path.read_text(encoding="utf-8") == (
        "# Flashcards\n"
        "\n"
        "## Index\n"
        "\n"
        "_No flashcards exported._\n"
        "\n"
        "## Cards\n"
        "\n"
        "_No flashcards exported._\n"
    )
    assert stats == {
        "path": str(path),
        "units_exported": 0,
        "cards_exported": 0,
    }
