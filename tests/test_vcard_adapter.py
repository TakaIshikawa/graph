from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from graph.adapters.vcard import VCardAdapter
from graph.types.enums import ContentType, SourceProject


def test_ingests_single_vcard_file(tmp_path):
    """Test ingesting a single vCard file with one contact."""
    vcard_content = """BEGIN:VCARD
VERSION:3.0
FN:John Doe
N:Doe;John;;;
ORG:Acme Corporation
TITLE:Software Engineer
EMAIL:john.doe@example.com
TEL:+1-555-1234
NOTE:Met at conference 2024
END:VCARD
"""
    vcard_file = tmp_path / "contact.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.VCARD
    assert unit.source_entity_type == "contact"
    assert unit.title == "John Doe"
    assert "Name: John Doe" in unit.content
    assert "Organization: Acme Corporation" in unit.content
    assert "Title: Software Engineer" in unit.content
    assert "Email: john.doe@example.com" in unit.content
    assert "Phone: +1-555-1234" in unit.content
    assert "Notes:" in unit.content
    assert "Met at conference 2024" in unit.content
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.metadata["fn"] == "John Doe"
    assert unit.metadata["org"] == "Acme Corporation"
    assert unit.metadata["title"] == "Software Engineer"
    assert unit.metadata["email"] == "john.doe@example.com"
    assert unit.metadata["tel"] == "+1-555-1234"
    assert unit.metadata["vcard_version"] == "3.0"
    assert unit.metadata["source_file"] == "contact.vcf"
    assert isinstance(unit.created_at, datetime)
    assert isinstance(unit.updated_at, datetime)


def test_ingests_vcard_4_0_format(tmp_path):
    """Test ingesting a vCard 4.0 format file."""
    vcard_content = """BEGIN:VCARD
VERSION:4.0
FN:Jane Smith
N:Smith;Jane;Marie;Dr.;PhD
ORG:Tech Solutions Inc.
TITLE:Chief Technology Officer
EMAIL;TYPE=work:jane.smith@techsolutions.com
EMAIL;TYPE=home:jane@personal.com
TEL;TYPE=cell:+1-555-5678
TEL;TYPE=work:+1-555-9999
ADR;TYPE=work:;;123 Main St;Springfield;IL;62701;USA
URL:https://janesmith.com
BDAY:19850315
END:VCARD
"""
    vcard_file = tmp_path / "jane.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Jane Smith"
    assert unit.metadata["vcard_version"] == "4.0"
    # Multiple emails stored as list
    assert isinstance(unit.metadata["email"], list)
    assert len(unit.metadata["email"]) == 2
    assert "jane.smith@techsolutions.com" in unit.metadata["email"]
    assert "jane@personal.com" in unit.metadata["email"]
    # Multiple phones stored as list
    assert isinstance(unit.metadata["tel"], list)
    assert len(unit.metadata["tel"]) == 2
    # Check content includes both emails and phones
    assert "Emails:" in unit.content
    assert "Phones:" in unit.content


def test_ingests_multiple_vcards_from_one_file(tmp_path):
    """Test ingesting multiple vCard entries from a single file."""
    vcard_content = """BEGIN:VCARD
VERSION:3.0
FN:Alice Johnson
EMAIL:alice@example.com
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:Bob Williams
EMAIL:bob@example.com
TEL:+1-555-0000
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:Charlie Brown
ORG:Peanuts Inc
END:VCARD
"""
    vcard_file = tmp_path / "contacts.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 3
    # Check titles
    titles = [unit.title for unit in result.units]
    assert "Alice Johnson" in titles
    assert "Bob Williams" in titles
    assert "Charlie Brown" in titles


def test_ingests_multiple_vcard_files_from_directory(tmp_path):
    """Test ingesting multiple vCard files from a directory."""
    dir1 = tmp_path / "contacts"
    dir1.mkdir()

    file1 = dir1 / "person1.vcf"
    file1.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Person One\nEND:VCARD\n", encoding="utf-8")

    file2 = dir1 / "person2.vcf"
    file2.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Person Two\nEND:VCARD\n", encoding="utf-8")

    subdir = dir1 / "work"
    subdir.mkdir()
    file3 = subdir / "person3.vcard"
    file3.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Person Three\nEND:VCARD\n", encoding="utf-8")

    result = VCardAdapter(path=str(dir1)).ingest()

    assert len(result.units) == 3
    titles = sorted([unit.title for unit in result.units])
    assert titles == ["Person One", "Person Three", "Person Two"]
    # Check source files
    source_files = [unit.metadata["source_file"] for unit in result.units]
    assert "person1.vcf" in source_files
    assert "person2.vcf" in source_files
    assert "work/person3.vcard" in source_files


def test_handles_vcard_without_fn_field(tmp_path):
    """Test handling vCard without FN field (uses N field instead)."""
    vcard_content = """BEGIN:VCARD
VERSION:3.0
N:Last;First;Middle;Dr.;Jr.
EMAIL:test@example.com
END:VCARD
"""
    vcard_file = tmp_path / "no_fn.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should construct name from N field: Prefix Given Middle Family Suffix
    assert unit.title == "Dr. First Middle Last Jr."


def test_handles_line_folding(tmp_path):
    """Test handling vCard line folding (continuation lines)."""
    vcard_content = """BEGIN:VCARD
VERSION:3.0
FN:Long Name With
 Folded Line
NOTE:This is a very long note that spans
 multiple lines because it was folded
 for readability
EMAIL:test@example.com
END:VCARD
"""
    vcard_file = tmp_path / "folded.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Long Name WithFolded Line"
    # Note should be unfolded
    assert "This is a very long note that spansmultiple lines because it was foldedfor readability" in unit.metadata["note"]


def test_handles_escaped_characters(tmp_path):
    """Test handling escaped characters in vCard fields."""
    vcard_content = r"""BEGIN:VCARD
VERSION:3.0
FN:Test User
NOTE:Line 1\nLine 2\nLine 3
ORG:Company\, Inc.
ADR:;;123\, Main St\;Apt 4;City;State;12345;Country
END:VCARD
"""
    vcard_file = tmp_path / "escaped.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Escaped newlines should be converted
    assert "\n" in unit.metadata["note"]
    assert "Line 1\nLine 2\nLine 3" == unit.metadata["note"]
    # Escaped commas and semicolons
    assert "Company, Inc." == unit.metadata["org"]


def test_handles_unicode_content(tmp_path):
    """Test handling Unicode characters in vCard fields."""
    vcard_content = """BEGIN:VCARD
VERSION:3.0
FN:田中太郎
ORG:日本株式会社
EMAIL:tanaka@example.jp
NOTE:こんにちは世界 🌍
END:VCARD
"""
    vcard_file = tmp_path / "unicode.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "田中太郎"
    assert unit.metadata["org"] == "日本株式会社"
    assert "🌍" in unit.metadata["note"]


def test_skips_non_vcard_files(tmp_path):
    """Test that non-vCard files are ignored."""
    dir1 = tmp_path / "mixed"
    dir1.mkdir()

    vcard_file = tmp_path / "mixed" / "contact.vcf"
    vcard_file.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Valid Contact\nEND:VCARD\n", encoding="utf-8")

    txt_file = tmp_path / "mixed" / "notes.txt"
    txt_file.write_text("Some text", encoding="utf-8")

    csv_file = tmp_path / "mixed" / "data.csv"
    csv_file.write_text("col1,col2\nval1,val2\n", encoding="utf-8")

    result = VCardAdapter(path=str(dir1)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "Valid Contact"


def test_nonexistent_path_returns_empty_result(tmp_path):
    """Test that a nonexistent path returns an empty result."""
    nonexistent = tmp_path / "nonexistent"

    result = VCardAdapter(path=str(nonexistent)).ingest()

    assert result.units == []
    assert result.edges == []


def test_entity_type_filter_returns_empty_result(tmp_path):
    """Test that filtering by a different entity type returns empty result."""
    vcard_file = tmp_path / "contact.vcf"
    vcard_file.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Test\nEND:VCARD\n", encoding="utf-8")

    result = VCardAdapter(path=str(tmp_path)).ingest(entity_types=["other_type"])

    assert result.units == []
    assert result.edges == []


def test_entity_type_filter_includes_contact(tmp_path):
    """Test that filtering by contact entity type includes results."""
    vcard_file = tmp_path / "contact.vcf"
    vcard_file.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Test\nEND:VCARD\n", encoding="utf-8")

    result = VCardAdapter(path=str(tmp_path)).ingest(entity_types=["contact"])

    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "contact"


def test_sync_state_filters_unmodified_files(tmp_path):
    """Test that sync state filtering skips unmodified files."""
    from graph.types.models import SyncState

    vcard_file = tmp_path / "old.vcf"
    vcard_file.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Old Contact\nEND:VCARD\n", encoding="utf-8")

    # Get the file's timestamp
    old_mtime = vcard_file.stat().st_mtime

    # Create a sync state with a timestamp after the file modification
    sync_state = SyncState(
        source_project="vcard",
        source_entity_type="contact",
        last_sync_at=datetime.fromtimestamp(old_mtime + 1, tz=timezone.utc),
    )

    result = VCardAdapter(path=str(tmp_path)).ingest(since=sync_state)

    # File should be filtered out since it wasn't modified after sync
    assert result.units == []


def test_uses_source_id_root_for_relative_paths(tmp_path):
    """Test that source_id_root is used for computing relative paths."""
    subdir = tmp_path / "project" / "contacts"
    subdir.mkdir(parents=True)

    vcard_file = subdir / "person.vcf"
    vcard_file.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Test Person\nEND:VCARD\n", encoding="utf-8")

    result = VCardAdapter(
        path=str(vcard_file),
        source_id_root=str(tmp_path)
    ).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["source_file"] == "project/contacts/person.vcf"


def test_root_path_parameter_works(tmp_path):
    """Test that root_path parameter works as an alternative to path."""
    vcard_file = tmp_path / "contact.vcf"
    vcard_file.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Test\nEND:VCARD\n", encoding="utf-8")

    result = VCardAdapter(root_path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "Test"


def test_handles_vcard_with_multiple_fields(tmp_path):
    """Test vCard with many different fields."""
    vcard_content = """BEGIN:VCARD
VERSION:4.0
FN:John Q. Public
N:Public;John;Quinlan;Mr.;Esq.
NICKNAME:Johnny
ORG:ABC Corporation;Marketing Department
TITLE:Marketing Manager
ROLE:Executive
EMAIL;TYPE=work:john.public@abc.com
EMAIL;TYPE=home:jqp@home.net
TEL;TYPE=work,voice:+1-555-1234
TEL;TYPE=cell:+1-555-5678
ADR;TYPE=work:;;123 Business Ave;Big City;CA;90210;USA
URL:https://johnqpublic.com
BDAY:19750823
CATEGORIES:VIP,Client
NOTE:Important client from 2020
END:VCARD
"""
    vcard_file = tmp_path / "detailed.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "John Q. Public"
    assert unit.metadata["nickname"] == "Johnny"
    assert unit.metadata["role"] == "Executive"
    assert unit.metadata["bday"] == "19750823"
    assert isinstance(unit.metadata["categories"], str)
    assert "VIP,Client" in unit.metadata["categories"]
    assert isinstance(unit.metadata["url"], str)


def test_handles_empty_vcard_file(tmp_path):
    """Test handling an empty vCard file."""
    vcard_file = tmp_path / "empty.vcf"
    vcard_file.write_text("", encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    # Empty file should produce no units
    assert result.units == []


def test_handles_malformed_vcard(tmp_path):
    """Test handling a malformed vCard (missing END:VCARD)."""
    vcard_content = """BEGIN:VCARD
VERSION:3.0
FN:Incomplete Contact
EMAIL:test@example.com
"""
    vcard_file = tmp_path / "malformed.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    # Malformed vCard should not produce a unit
    assert result.units == []


def test_source_id_generation(tmp_path):
    """Test that source_id is generated deterministically."""
    vcard_content = """BEGIN:VCARD
VERSION:3.0
FN:Test Contact
EMAIL:test@example.com
END:VCARD
"""
    vcard_file = tmp_path / "test.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Source ID should be in format "vcard:{hash}"
    assert unit.source_id.startswith("vcard:")
    # Verify it's deterministic
    source_id_base = "test.vcf:Test Contact"
    expected_digest = hashlib.sha256(source_id_base.encode("utf-8")).hexdigest()[:16]
    assert unit.source_id == f"vcard:{expected_digest}"


def test_handles_windows_line_endings(tmp_path):
    """Test handling vCard with Windows line endings (CRLF)."""
    vcard_content = "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Windows Contact\r\nEMAIL:test@example.com\r\nEND:VCARD\r\n"
    vcard_file = tmp_path / "windows.vcf"
    vcard_file.write_text(vcard_content, encoding="utf-8")

    result = VCardAdapter(path=str(vcard_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Windows Contact"
    assert unit.metadata["email"] == "test@example.com"
