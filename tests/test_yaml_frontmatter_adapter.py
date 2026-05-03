from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from graph.adapters.yaml_frontmatter import YamlFrontmatterAdapter
from graph.types.enums import ContentType, SourceProject


def test_ingests_markdown_with_frontmatter(tmp_path):
    """Test ingesting a markdown file with YAML frontmatter."""
    content = """---
title: My Document
tags: [python, testing, automation]
author: John Doe
date: 2024-01-15
---

# Introduction

This is the document content.
"""
    md_file = tmp_path / "document.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.YAML_FRONTMATTER
    assert unit.source_entity_type == "yaml_frontmatter"
    assert unit.title == "My Document"
    assert "# Introduction" in unit.content
    assert "This is the document content." in unit.content
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.metadata["source_file"] == "document.md"
    assert unit.metadata["frontmatter"]["title"] == "My Document"
    assert unit.metadata["frontmatter"]["author"] == "John Doe"
    assert "python" in unit.tags
    assert "testing" in unit.tags
    assert "automation" in unit.tags
    assert isinstance(unit.created_at, datetime)
    assert isinstance(unit.updated_at, datetime)


def test_ingests_text_file_with_frontmatter(tmp_path):
    """Test ingesting a plain text file with YAML frontmatter."""
    content = """---
title: Notes on Testing
category: Development
---

These are my notes on testing strategies.
Multiple lines of content.
"""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(txt_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Notes on Testing"
    assert "These are my notes on testing strategies." in unit.content
    assert unit.metadata["categories"] == ["development"]


def test_handles_nested_yaml_structures(tmp_path):
    """Test handling nested YAML structures in frontmatter."""
    content = """---
title: Complex Document
metadata:
  author: Jane Smith
  version: 1.2.3
  status: draft
settings:
  publish: false
  notifications:
    - email
    - slack
tags:
  - documentation
  - advanced
---

Content here.
"""
    md_file = tmp_path / "complex.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Complex Document"
    frontmatter = unit.metadata["frontmatter"]
    assert isinstance(frontmatter["metadata"], dict)
    assert frontmatter["metadata"]["author"] == "Jane Smith"
    assert frontmatter["metadata"]["version"] == "1.2.3"
    assert isinstance(frontmatter["settings"], dict)
    assert frontmatter["settings"]["publish"] is False
    assert isinstance(frontmatter["settings"]["notifications"], list)


def test_handles_arrays_in_frontmatter(tmp_path):
    """Test handling arrays in frontmatter."""
    content = """---
title: Array Test
authors:
  - Alice Johnson
  - Bob Williams
  - Charlie Brown
categories: [tech, science, research]
---

Content with multiple authors.
"""
    md_file = tmp_path / "arrays.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["authors"] == ["alice johnson", "bob williams", "charlie brown"]
    assert unit.metadata["categories"] == ["tech", "science", "research"]


def test_handles_file_without_frontmatter(tmp_path):
    """Test handling a file without frontmatter."""
    content = """# Simple Document

This file has no frontmatter.
Just regular content.
"""
    md_file = tmp_path / "simple.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Title should be extracted from first heading
    assert unit.title == "Simple Document"
    assert unit.content == content
    assert unit.metadata["frontmatter"] == {}
    assert unit.tags == []


def test_handles_file_with_empty_frontmatter(tmp_path):
    """Test handling a file with empty frontmatter."""
    content = """---
---

Content after empty frontmatter.
"""
    md_file = tmp_path / "empty_front.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["frontmatter"] == {}
    assert "Content after empty frontmatter." in unit.content


def test_extracts_title_from_frontmatter(tmp_path):
    """Test that title is extracted from frontmatter."""
    content = """---
title: Frontmatter Title
---

# Body Heading

Content here.
"""
    md_file = tmp_path / "title_test.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "Frontmatter Title"


def test_extracts_title_from_heading_if_no_frontmatter_title(tmp_path):
    """Test that title is extracted from heading if frontmatter has no title."""
    content = """---
tags: [test]
---

# Heading Title

Content here.
"""
    md_file = tmp_path / "heading_title.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "Heading Title"


def test_uses_filename_as_fallback_title(tmp_path):
    """Test that filename is used as fallback title."""
    content = """---
tags: [test]
---

No heading in this content.
"""
    md_file = tmp_path / "fallback_title.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "fallback_title"


def test_handles_datetime_fields(tmp_path):
    """Test handling datetime fields in frontmatter."""
    content = """---
title: Dated Document
created: 2024-01-15T10:30:00Z
updated: 2024-02-20T15:45:00+00:00
---

Content with dates.
"""
    md_file = tmp_path / "dates.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.created_at == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 2, 20, 15, 45, 0, tzinfo=timezone.utc)


def test_handles_alternative_date_field_names(tmp_path):
    """Test handling alternative date field names."""
    content = """---
title: Alternative Dates
date: 2024-01-10
modified: 2024-01-20
---

Content.
"""
    md_file = tmp_path / "alt_dates.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.created_at.year == 2024
    assert unit.created_at.month == 1
    assert unit.created_at.day == 10
    assert unit.updated_at.year == 2024
    assert unit.updated_at.month == 1
    assert unit.updated_at.day == 20


def test_handles_multiple_file_types(tmp_path):
    """Test handling multiple file types."""
    dir1 = tmp_path / "docs"
    dir1.mkdir()

    # Markdown file
    md_file = dir1 / "doc1.md"
    md_file.write_text("---\ntitle: Markdown\n---\nContent", encoding="utf-8")

    # Text file
    txt_file = dir1 / "doc2.txt"
    txt_file.write_text("---\ntitle: Text\n---\nContent", encoding="utf-8")

    # Unsupported file type
    py_file = dir1 / "script.py"
    py_file.write_text("# Python script", encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(dir1)).ingest()

    assert len(result.units) == 2
    titles = sorted([unit.title for unit in result.units])
    assert titles == ["Markdown", "Text"]


def test_custom_file_extensions(tmp_path):
    """Test using custom file extensions."""
    dir1 = tmp_path / "docs"
    dir1.mkdir()

    rst_file = dir1 / "doc.rst"
    rst_file.write_text("---\ntitle: ReStructuredText\n---\nContent", encoding="utf-8")

    md_file = dir1 / "doc.md"
    md_file.write_text("---\ntitle: Markdown\n---\nContent", encoding="utf-8")

    # Only process .rst files
    result = YamlFrontmatterAdapter(
        path=str(dir1),
        file_extensions=[".rst"]
    ).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "ReStructuredText"


def test_handles_malformed_yaml(tmp_path):
    """Test handling malformed YAML in frontmatter."""
    content = """---
title: Test
invalid: [unclosed array
---

Content here.
"""
    md_file = tmp_path / "malformed.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should return the original content when frontmatter parsing fails
    assert "---" in unit.content
    assert "frontmatter_parse_error" in unit.metadata


def test_handles_unicode_content(tmp_path):
    """Test handling Unicode characters in frontmatter and content."""
    content = """---
title: 日本語ドキュメント
author: 田中太郎
tags: [テスト, ドキュメント]
---

# はじめに

これは日本語のコンテンツです。🎌
"""
    md_file = tmp_path / "unicode.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "日本語ドキュメント"
    assert unit.metadata["authors"] == ["田中太郎"]
    assert "🎌" in unit.content


def test_ingests_multiple_files_from_directory(tmp_path):
    """Test ingesting multiple files from a directory."""
    dir1 = tmp_path / "notes"
    dir1.mkdir()

    file1 = dir1 / "note1.md"
    file1.write_text("---\ntitle: Note 1\n---\nContent 1", encoding="utf-8")

    file2 = dir1 / "note2.md"
    file2.write_text("---\ntitle: Note 2\n---\nContent 2", encoding="utf-8")

    subdir = dir1 / "archived"
    subdir.mkdir()
    file3 = subdir / "note3.md"
    file3.write_text("---\ntitle: Note 3\n---\nContent 3", encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(dir1)).ingest()

    assert len(result.units) == 3
    titles = sorted([unit.title for unit in result.units])
    assert titles == ["Note 1", "Note 2", "Note 3"]


def test_nonexistent_path_returns_empty_result(tmp_path):
    """Test that a nonexistent path returns an empty result."""
    nonexistent = tmp_path / "nonexistent"

    result = YamlFrontmatterAdapter(path=str(nonexistent)).ingest()

    assert result.units == []
    assert result.edges == []


def test_entity_type_filter_returns_empty_result(tmp_path):
    """Test that filtering by a different entity type returns empty result."""
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: Test\n---\nContent", encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(tmp_path)).ingest(entity_types=["other_type"])

    assert result.units == []
    assert result.edges == []


def test_entity_type_filter_includes_yaml_frontmatter(tmp_path):
    """Test that filtering by yaml_frontmatter entity type includes results."""
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: Test\n---\nContent", encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(tmp_path)).ingest(entity_types=["yaml_frontmatter"])

    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "yaml_frontmatter"


def test_sync_state_filters_unmodified_files(tmp_path):
    """Test that sync state filtering skips unmodified files."""
    from graph.types.models import SyncState

    md_file = tmp_path / "old.md"
    md_file.write_text("---\ntitle: Old\n---\nContent", encoding="utf-8")

    # Get the file's timestamp
    old_mtime = md_file.stat().st_mtime

    # Create a sync state with a timestamp after the file modification
    sync_state = SyncState(
        source_project="yaml_frontmatter",
        source_entity_type="yaml_frontmatter",
        last_sync_at=datetime.fromtimestamp(old_mtime + 1, tz=timezone.utc),
    )

    result = YamlFrontmatterAdapter(path=str(tmp_path)).ingest(since=sync_state)

    # File should be filtered out since it wasn't modified after sync
    assert result.units == []


def test_uses_source_id_root_for_relative_paths(tmp_path):
    """Test that source_id_root is used for computing relative paths."""
    subdir = tmp_path / "project" / "docs"
    subdir.mkdir(parents=True)

    md_file = subdir / "note.md"
    md_file.write_text("---\ntitle: Test\n---\nContent", encoding="utf-8")

    result = YamlFrontmatterAdapter(
        path=str(md_file),
        source_id_root=str(tmp_path)
    ).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["source_file"] == "project/docs/note.md"


def test_root_path_parameter_works(tmp_path):
    """Test that root_path parameter works as an alternative to path."""
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: Test\n---\nContent", encoding="utf-8")

    result = YamlFrontmatterAdapter(root_path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "Test"


def test_source_id_generation(tmp_path):
    """Test that source_id is generated deterministically."""
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: Test\n---\nContent", encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id.startswith("yaml_frontmatter:")
    # Verify it's deterministic
    expected_digest = hashlib.sha256("test.md".encode("utf-8")).hexdigest()[:16]
    assert unit.source_id == f"yaml_frontmatter:{expected_digest}"


def test_handles_description_field(tmp_path):
    """Test handling description field in frontmatter."""
    content = """---
title: Document with Description
description: This is a summary of the document content
---

Full content here.
"""
    md_file = tmp_path / "desc.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["description"] == "This is a summary of the document content"


def test_handles_source_url_field(tmp_path):
    """Test handling source_url field in frontmatter."""
    content = """---
title: External Document
url: https://example.com/original
---

Content.
"""
    md_file = tmp_path / "url.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["source_url"] == "https://example.com/original"


def test_preserves_newlines_in_content(tmp_path):
    """Test that newlines in content are preserved."""
    content = """---
title: Test
---

Line 1

Line 2

Line 3
"""
    md_file = tmp_path / "newlines.md"
    md_file.write_text(content, encoding="utf-8")

    result = YamlFrontmatterAdapter(path=str(md_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Content should preserve empty lines between text
    assert "\n\n" in unit.content
    assert "Line 1" in unit.content
    assert "Line 2" in unit.content
    assert "Line 3" in unit.content
