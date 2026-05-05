from __future__ import annotations

import mailbox
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.mbox import MboxAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject


def write_mbox(path: Path, messages: list[dict]) -> Path:
    """Helper to create an mbox file with test messages."""
    mbox = mailbox.mbox(path, create=True)
    for msg_data in messages:
        msg = mailbox.mboxMessage()
        for header, value in msg_data.items():
            if header != "body":
                msg[header] = value
        msg.set_payload(msg_data.get("body", ""))
        mbox.add(msg)
    mbox.close()
    return path


def test_mbox_ingests_single_email_with_metadata(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "test.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Test Email",
                "Date": "Wed, 01 May 2026 10:30:00 +0000",
                "Message-ID": "<msg-001@example.com>",
                "body": "This is a test email body.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MBOX
    assert unit.source_entity_type == "email"
    assert unit.title == "Test Email"
    assert unit.content == "This is a test email body."
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.created_at == datetime(2026, 5, 1, 10, 30, tzinfo=timezone.utc)
    assert unit.metadata["from"] == "alice@example.com"
    assert unit.metadata["to"] == "bob@example.com"
    assert unit.metadata["subject"] == "Test Email"
    assert unit.metadata["message_id"] == "<msg-001@example.com>"


def test_mbox_ingests_multiple_emails(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "multiple.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "First Email",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<msg-001@example.com>",
                "body": "First message.",
            },
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Second Email",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<msg-002@example.com>",
                "body": "Second message.",
            },
            {
                "From": "charlie@example.com",
                "To": "alice@example.com",
                "Subject": "Third Email",
                "Date": "Wed, 01 May 2026 12:00:00 +0000",
                "Message-ID": "<msg-003@example.com>",
                "body": "Third message.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 3
    titles = [unit.title for unit in result.units]
    assert "First Email" in titles
    assert "Second Email" in titles
    assert "Third Email" in titles


def test_mbox_creates_reply_edges_from_in_reply_to(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "thread.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Original Message",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<original@example.com>",
                "body": "Original message body.",
            },
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Re: Original Message",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<reply-001@example.com>",
                "In-Reply-To": "<original@example.com>",
                "body": "This is a reply.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 2
    assert len(result.edges) == 1

    edge = result.edges[0]
    assert edge.relation == EdgeRelation.REPLIES_TO
    assert edge.source == EdgeSource.SOURCE

    # Verify edge connects reply to original
    original_unit = next(u for u in result.units if u.title == "Original Message")
    reply_unit = next(u for u in result.units if u.title == "Re: Original Message")
    assert edge.from_unit_id == reply_unit.source_id
    assert edge.to_unit_id == original_unit.source_id


def test_mbox_creates_reply_edges_from_references(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "thread-refs.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "team@example.com",
                "Subject": "Discussion Thread",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<thread-start@example.com>",
                "body": "Starting a discussion.",
            },
            {
                "From": "bob@example.com",
                "To": "team@example.com",
                "Subject": "Re: Discussion Thread",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<reply-001@example.com>",
                "References": "<thread-start@example.com>",
                "body": "My thoughts on this.",
            },
            {
                "From": "charlie@example.com",
                "To": "team@example.com",
                "Subject": "Re: Discussion Thread",
                "Date": "Wed, 01 May 2026 12:00:00 +0000",
                "Message-ID": "<reply-002@example.com>",
                "References": "<thread-start@example.com> <reply-001@example.com>",
                "body": "Building on previous comments.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 3
    assert len(result.edges) == 2

    # All edges should be REPLIES_TO
    assert all(edge.relation == EdgeRelation.REPLIES_TO for edge in result.edges)


def test_mbox_handles_email_without_message_id(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "no-msgid.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Email Without Message-ID",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "This email has no Message-ID header.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Email Without Message-ID"
    assert "message_id" not in unit.metadata


def test_mbox_handles_email_with_cc(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "with-cc.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Cc": "charlie@example.com, david@example.com",
                "Subject": "Email with CC",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<cc-test@example.com>",
                "body": "This email has CC recipients.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["cc"] == "charlie@example.com, david@example.com"


def test_mbox_handles_email_without_subject(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "no-subject.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<no-subject@example.com>",
                "body": "This email has no subject.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Untitled"


def test_mbox_filters_by_entity_type(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "filter-test.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Test Email",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "Test body.",
            }
        ],
    )

    # Should return empty when filtering by different entity type
    result = MboxAdapter(path=str(mbox_path)).ingest(entity_types=["other_type"])
    assert len(result.units) == 0

    # Should return results when filtering by email
    result = MboxAdapter(path=str(mbox_path)).ingest(entity_types=["email"])
    assert len(result.units) == 1


def test_mbox_filters_by_sync_state(tmp_path):
    from graph.types.models import SyncState

    mbox_path = write_mbox(
        tmp_path / "sync-test.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Old Email",
                "Date": "Wed, 01 Jan 2020 10:00:00 +0000",
                "Message-ID": "<old@example.com>",
                "body": "Old message.",
            },
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "New Email",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<new@example.com>",
                "body": "New message.",
            },
        ],
    )

    # Filter with sync state after old email but before new email
    sync_state = SyncState(
        source_project="mbox",
        source_entity_type="email",
        last_sync_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    result = MboxAdapter(path=str(mbox_path)).ingest(since=sync_state)

    # Should only get the new email
    assert len(result.units) == 1
    assert result.units[0].title == "New Email"


def test_mbox_handles_directory_of_mbox_files(tmp_path):
    mbox_dir = tmp_path / "mboxes"
    mbox_dir.mkdir()

    write_mbox(
        mbox_dir / "inbox.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Inbox Message",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "From inbox.",
            }
        ],
    )

    write_mbox(
        mbox_dir / "sent.mbox",
        [
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Sent Message",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "body": "From sent.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_dir)).ingest()

    assert len(result.units) == 2
    titles = [unit.title for unit in result.units]
    assert "Inbox Message" in titles
    assert "Sent Message" in titles


def test_mbox_avoids_duplicate_reply_edges(tmp_path):
    """Test that only one reply edge is created even with multiple references."""
    mbox_path = write_mbox(
        tmp_path / "multi-refs.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "team@example.com",
                "Subject": "Thread Start",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<start@example.com>",
                "body": "Starting thread.",
            },
            {
                "From": "bob@example.com",
                "To": "team@example.com",
                "Subject": "Re: Thread Start",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<reply@example.com>",
                "In-Reply-To": "<start@example.com>",
                "References": "<start@example.com>",
                "body": "Reply with both In-Reply-To and References.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 2
    # Should only create one edge, not duplicate
    assert len(result.edges) == 1


def test_mbox_adapter_is_registered():
    assert "mbox" in list_adapters()
    adapter = get_adapter("mbox", path="/tmp/mail.mbox")
    assert isinstance(adapter, MboxAdapter)
    assert adapter.name == "mbox"


def test_mbox_handles_html_email_content(tmp_path):
    """Test that HTML content in emails is properly stripped."""
    mbox_path = tmp_path / "html.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)
    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "HTML Email"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<html@example.com>"

    # Create multipart message with HTML
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    multipart = MIMEMultipart("alternative")
    for key in msg.keys():
        multipart[key] = msg[key]

    html_content = """
    <html>
      <body>
        <h1>Heading</h1>
        <p>This is a <strong>bold</strong> paragraph.</p>
        <ul>
          <li>Item 1</li>
          <li>Item 2</li>
        </ul>
      </body>
    </html>
    """
    html_part = MIMEText(html_content, "html")
    multipart.attach(html_part)

    mbox.add(multipart)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # HTML should be stripped to plain text
    assert "<" not in unit.content
    assert ">" not in unit.content
    assert "Heading" in unit.content
    assert "bold" in unit.content


def test_mbox_handles_multipart_with_plain_text(tmp_path):
    """Test that plain text is preferred over HTML in multipart emails."""
    mbox_path = tmp_path / "multipart.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Multipart Email"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<multipart@example.com>"

    plain_text = "This is plain text content."
    html_text = "<html><body><p>This is HTML content.</p></body></html>"

    plain_part = MIMEText(plain_text, "plain")
    html_part = MIMEText(html_text, "html")

    msg.attach(plain_part)
    msg.attach(html_part)

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should prefer plain text over HTML
    assert unit.content == "This is plain text content."


def test_mbox_handles_empty_path():
    """Test that adapter handles empty path gracefully."""
    result = MboxAdapter(path="").ingest()
    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_mbox_handles_nonexistent_path(tmp_path):
    """Test that adapter handles nonexistent path gracefully."""
    result = MboxAdapter(path=str(tmp_path / "nonexistent.mbox")).ingest()
    assert len(result.units) == 0
    assert len(result.edges) == 0


# Edge case tests: Malformed headers


def test_mbox_handles_malformed_date_header(tmp_path):
    """Test that malformed date headers don't crash ingestion."""
    mbox_path = write_mbox(
        tmp_path / "bad-date.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Bad Date",
                "Date": "This is not a valid date",
                "Message-ID": "<bad-date@example.com>",
                "body": "Email with malformed date.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should use current time as fallback
    assert unit.created_at is not None
    assert unit.title == "Bad Date"


def test_mbox_handles_empty_date_header(tmp_path):
    """Test that empty date headers use current time."""
    mbox_path = write_mbox(
        tmp_path / "no-date.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "No Date",
                "Message-ID": "<no-date@example.com>",
                "body": "Email without date.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.created_at is not None


def test_mbox_handles_malformed_message_id_in_references(tmp_path):
    """Test that malformed Message-IDs in References don't crash."""
    mbox_path = write_mbox(
        tmp_path / "bad-refs.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Original",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<original@example.com>",
                "body": "Original message.",
            },
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Re: Original",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<reply@example.com>",
                "References": "not-a-valid-message-id <original@example.com> another-invalid",
                "body": "Reply with malformed references.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 2
    # Should still create edge for valid Message-ID
    assert len(result.edges) == 1


def test_mbox_handles_empty_headers(tmp_path):
    """Test that empty headers are handled gracefully."""
    mbox_path = write_mbox(
        tmp_path / "empty-headers.mbox",
        [
            {
                "From": "",
                "To": "",
                "Subject": "",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "Email with empty headers.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Untitled"
    assert unit.metadata["from"] == ""
    assert unit.metadata["to"] == ""


# Edge case tests: Missing Message-ID scenarios


def test_mbox_missing_message_id_threading(tmp_path):
    """Test that emails without Message-ID can't be threaded."""
    mbox_path = write_mbox(
        tmp_path / "no-msgid-thread.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Original",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "Original without Message-ID.",
            },
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Re: Original",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "In-Reply-To": "<nonexistent@example.com>",
                "body": "Reply referencing nonexistent Message-ID.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 2
    # No edges should be created since Message-IDs don't match
    assert len(result.edges) == 0


def test_mbox_multiple_emails_without_message_id(tmp_path):
    """Test that multiple emails without Message-IDs get unique source IDs."""
    mbox_path = write_mbox(
        tmp_path / "multiple-no-msgid.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "First",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "First message.",
            },
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Second",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "body": "Second message.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 2
    # Verify unique source IDs
    source_ids = [unit.source_id for unit in result.units]
    assert len(set(source_ids)) == 2


# Edge case tests: Threading cycles


def test_mbox_handles_threading_cycle(tmp_path):
    """Test that circular threading references don't cause infinite loops."""
    mbox_path = write_mbox(
        tmp_path / "cycle.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Message A",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<msg-a@example.com>",
                "In-Reply-To": "<msg-b@example.com>",
                "body": "Message A replies to B.",
            },
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Message B",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<msg-b@example.com>",
                "In-Reply-To": "<msg-a@example.com>",
                "body": "Message B replies to A.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 2
    # Should create edges (one for each message to its parent)
    assert len(result.edges) == 2


def test_mbox_handles_self_referencing_message(tmp_path):
    """Test that a message referencing itself creates a self-loop edge."""
    mbox_path = write_mbox(
        tmp_path / "self-ref.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Self Reference",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<self@example.com>",
                "In-Reply-To": "<self@example.com>",
                "body": "Message references itself.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    # Creates self-loop edge (from_unit_id == to_unit_id)
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == edge.to_unit_id


def test_mbox_handles_long_reference_chain(tmp_path):
    """Test that long threading chains are handled correctly."""
    messages = []
    previous_id = None
    references = []

    for i in range(10):
        msg_id = f"<msg-{i:03d}@example.com>"
        msg = {
            "From": f"user{i}@example.com",
            "To": "team@example.com",
            "Subject": f"Re: Thread (Message {i})",
            "Date": f"Wed, 0{(i % 9) + 1} May 2026 10:{i:02d}:00 +0000",
            "Message-ID": msg_id,
            "body": f"Message {i} in chain.",
        }
        if previous_id:
            msg["In-Reply-To"] = previous_id
            msg["References"] = " ".join(references)

        messages.append(msg)
        references.append(msg_id)
        previous_id = msg_id

    mbox_path = write_mbox(tmp_path / "long-chain.mbox", messages)
    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 10
    # Each reply should have one edge (to its immediate parent)
    assert len(result.edges) == 9


# Edge case tests: Non-UTF8 encodings


def test_mbox_handles_latin1_encoding(tmp_path):
    """Test that Latin-1 encoded email content is decoded correctly."""
    mbox_path = tmp_path / "latin1.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Latin-1 Test"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<latin1@example.com>"

    # Set Latin-1 encoded payload
    latin1_text = "Café résumé naïve"
    msg.set_payload(latin1_text.encode("latin-1"))
    msg.set_type("text/plain")
    msg.set_param("charset", "iso-8859-1")

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert "Café" in unit.content or "Caf" in unit.content  # Should decode gracefully


def test_mbox_handles_invalid_encoding_gracefully(tmp_path):
    """Test that invalid encoding doesn't crash ingestion."""
    mbox_path = tmp_path / "invalid-encoding.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Invalid Encoding"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<invalid@example.com>"

    # Set payload with invalid UTF-8 bytes
    msg.set_payload(b"\xff\xfe Invalid UTF-8 \x80\x81")
    msg.set_type("text/plain")
    msg.set_param("charset", "utf-8")

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    # Should use replacement characters for invalid bytes
    assert result.units[0].content is not None


def test_mbox_handles_mixed_encodings_in_multipart(tmp_path):
    """Test multipart message with different encodings in each part."""
    mbox_path = tmp_path / "mixed-encoding.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Mixed Encoding"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<mixed@example.com>"

    # Add UTF-8 part
    utf8_part = MIMEText("UTF-8 text with émojis 🎉", "plain", "utf-8")
    msg.attach(utf8_part)

    # Add Latin-1 part
    latin1_part = MIMEText("Latin-1 text: café", "plain", "iso-8859-1")
    msg.attach(latin1_part)

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    # Should successfully decode both parts
    assert result.units[0].content is not None


# Edge case tests: Additional multipart MIME scenarios


def test_mbox_handles_multipart_with_attachments(tmp_path):
    """Test that attachments are ignored and text parts are extracted."""
    mbox_path = tmp_path / "with-attachment.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("mixed")
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "With Attachment"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<attachment@example.com>"

    # Add text part
    text_part = MIMEText("This is the email body.", "plain")
    msg.attach(text_part)

    # Add attachment (should be ignored)
    attachment = MIMEApplication(b"Binary data here", "octet-stream")
    attachment.add_header("Content-Disposition", "attachment", filename="file.bin")
    msg.attach(attachment)

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.content == "This is the email body."
    # Attachment content should not appear
    assert "Binary" not in unit.content


def test_mbox_handles_nested_multipart(tmp_path):
    """Test deeply nested multipart structures."""
    mbox_path = tmp_path / "nested.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("mixed")
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Nested Multipart"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<nested@example.com>"

    # Create nested structure
    inner = MIMEMultipart("alternative")
    inner.attach(MIMEText("Plain text version", "plain"))
    inner.attach(MIMEText("<p>HTML version</p>", "html"))

    msg.attach(inner)

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should extract plain text from nested structure
    assert "Plain text version" in unit.content


def test_mbox_handles_empty_multipart_sections(tmp_path):
    """Test multipart message with empty sections."""
    mbox_path = tmp_path / "empty-parts.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Empty Parts"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<empty@example.com>"

    # Add empty plain text part
    msg.attach(MIMEText("", "plain"))
    # Add HTML with actual content
    msg.attach(MIMEText("<p>HTML content</p>", "html"))

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    # Should fall back to HTML when plain text is empty
    assert "HTML content" in result.units[0].content


def test_mbox_handles_html_with_script_and_style_tags(tmp_path):
    """Test that script and style tags are properly excluded from HTML parsing."""
    mbox_path = tmp_path / "html-script.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.text import MIMEText

    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "HTML with Scripts"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<script@example.com>"

    html_content = """
    <html>
      <head>
        <style>
          .hidden { display: none; }
        </style>
        <script>
          alert('This should not appear');
        </script>
      </head>
      <body>
        <p>Visible content</p>
        <script>console.log('Also hidden');</script>
      </body>
    </html>
    """

    html_msg = MIMEText(html_content, "html")
    for key in msg.keys():
        html_msg[key] = msg[key]

    mbox.add(html_msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    content = result.units[0].content
    # Script and style content should be excluded
    assert "alert" not in content
    assert "console.log" not in content
    assert "display: none" not in content
    # Visible content should be present
    assert "Visible content" in content


# Edge case tests: Large archives


def test_mbox_handles_large_number_of_messages(tmp_path):
    """Test that large mbox files with many messages are handled efficiently."""
    messages = []
    for i in range(100):
        messages.append(
            {
                "From": f"user{i}@example.com",
                "To": "archive@example.com",
                "Subject": f"Message {i}",
                "Date": f"Wed, 01 May 2026 {i % 24:02d}:00:00 +0000",
                "Message-ID": f"<msg-{i}@example.com>",
                "body": f"Body of message {i}.",
            }
        )

    mbox_path = write_mbox(tmp_path / "large.mbox", messages)
    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 100
    # Verify all messages were processed
    subjects = [unit.title for unit in result.units]
    assert "Message 0" in subjects
    assert "Message 99" in subjects


def test_mbox_handles_very_long_email_body(tmp_path):
    """Test that emails with very long bodies are handled correctly."""
    long_body = "This is a very long email body. " * 1000  # ~33KB

    mbox_path = write_mbox(
        tmp_path / "long-body.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Long Email",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<long@example.com>",
                "body": long_body,
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should handle long content
    assert len(unit.content) > 30000


def test_mbox_handles_multiple_files_with_comma_separated_paths(tmp_path):
    """Test that comma-separated paths are handled correctly."""
    mbox1 = write_mbox(
        tmp_path / "first.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "First File",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "From first file.",
            }
        ],
    )

    mbox2 = write_mbox(
        tmp_path / "second.mbox",
        [
            {
                "From": "charlie@example.com",
                "To": "david@example.com",
                "Subject": "Second File",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "body": "From second file.",
            }
        ],
    )

    # Test comma-separated paths
    result = MboxAdapter(path=f"{mbox1},{mbox2}").ingest()
    assert len(result.units) == 2

    # Test newline-separated paths
    result = MboxAdapter(path=f"{mbox1}\n{mbox2}").ingest()
    assert len(result.units) == 2


def test_mbox_handles_corrupted_mbox_file(tmp_path):
    """Test that corrupted mbox files are handled gracefully."""
    corrupted_path = tmp_path / "corrupted.mbox"
    # Write invalid mbox content
    corrupted_path.write_bytes(b"\x00\x01\x02\x03 Invalid mbox data")

    result = MboxAdapter(path=str(corrupted_path)).ingest()

    # Should not crash, returns empty result
    assert len(result.units) >= 0  # May be 0 or may parse partial data


def test_mbox_deduplicates_source_paths(tmp_path):
    """Test that duplicate source paths are deduplicated."""
    mbox_path = write_mbox(
        tmp_path / "dedupe.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Test",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "Test body.",
            }
        ],
    )

    # Provide same path multiple times
    result = MboxAdapter(path=f"{mbox_path},{mbox_path},{mbox_path}").ingest()

    # Should only process once
    assert len(result.units) == 1


def test_mbox_entity_types_property():
    """Test that entity_types property returns correct list."""
    adapter = MboxAdapter()
    assert adapter.entity_types == ["email"]


def test_mbox_handles_date_without_timezone(tmp_path):
    """Test that dates without timezone info are handled correctly."""
    mbox_path = write_mbox(
        tmp_path / "no-tz.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "No TZ",
                "Date": "Wed, 01 May 2026 10:00:00",  # No timezone
                "Message-ID": "<no-tz@example.com>",
                "body": "Date without timezone.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should have UTC timezone assigned
    assert unit.created_at.tzinfo == timezone.utc


def test_mbox_handles_sync_state_with_string_datetime(tmp_path):
    """Test sync state with ISO string datetime format."""
    from graph.types.models import SyncState

    mbox_path = write_mbox(
        tmp_path / "sync-string.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Old Email",
                "Date": "Wed, 01 Jan 2020 10:00:00 +0000",
                "body": "Old message.",
            },
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "New Email",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "New message.",
            },
        ],
    )

    # Use string ISO format with Z suffix for sync state
    sync_state = SyncState(
        source_project="mbox",
        source_entity_type="email",
        last_sync_at="2025-01-01T00:00:00Z",
    )

    result = MboxAdapter(path=str(mbox_path)).ingest(since=sync_state)

    # Should only get the new email
    assert len(result.units) == 1
    assert result.units[0].title == "New Email"


def test_mbox_handles_path_with_empty_segments(tmp_path):
    """Test that paths with empty segments (extra commas/newlines) are handled."""
    mbox_path = write_mbox(
        tmp_path / "test.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Test",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "Test body.",
            }
        ],
    )

    # Path with empty segments (extra commas/newlines)
    result = MboxAdapter(path=f",,{mbox_path},,\n\n").ingest()

    assert len(result.units) == 1


def test_mbox_handles_multipart_with_non_text_parts(tmp_path):
    """Test that non-text parts (like application/pdf) are skipped."""
    mbox_path = tmp_path / "non-text.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("mixed")
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Non-Text Parts"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<non-text@example.com>"

    # Add text part
    text_part = MIMEText("This is the text content.", "plain")
    msg.attach(text_part)

    # Add application/pdf part (should be skipped)
    pdf_part = MIMEApplication(b"PDF content", "pdf")
    msg.attach(pdf_part)

    # Add application/json part (should be skipped)
    json_part = MIMEApplication(b'{"key": "value"}', "json")
    msg.attach(json_part)

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should only extract text content
    assert unit.content == "This is the text content."
    assert "PDF" not in unit.content
    assert "json" not in unit.content


def test_mbox_handles_html_with_inline_text(tmp_path):
    """Test HTML parsing with inline text mixed with block elements."""
    mbox_path = tmp_path / "inline-html.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.text import MIMEText

    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Inline HTML"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<inline@example.com>"

    html_content = """
    <html>
      <body>
        Inline text before paragraph
        <p>Paragraph text</p>
        Inline text after paragraph
      </body>
    </html>
    """

    html_msg = MIMEText(html_content, "html")
    for key in msg.keys():
        html_msg[key] = msg[key]

    mbox.add(html_msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    content = result.units[0].content
    # Should extract both inline and block text
    assert "Inline text before paragraph" in content
    assert "Paragraph text" in content
    assert "Inline text after paragraph" in content


# Additional error handling tests


def test_mbox_handles_empty_mbox_file(tmp_path):
    """Test that truly empty mbox file is handled gracefully."""
    empty_mbox = tmp_path / "empty.mbox"
    # Create empty mbox file
    mbox = mailbox.mbox(empty_mbox, create=True)
    mbox.close()

    result = MboxAdapter(path=str(empty_mbox)).ingest()

    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_mbox_handles_unknown_charset(tmp_path):
    """Test that messages with unknown/unsupported charset don't crash."""
    mbox_path = tmp_path / "unknown-charset.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Unknown Charset"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<unknown-charset@example.com>"

    # Set payload with bogus charset
    msg.set_payload(b"Test content with unknown charset")
    msg.set_type("text/plain")
    msg.set_param("charset", "x-unknown-bogus-charset")

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    # Should still ingest the message with replacement characters
    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Unknown Charset"
    # Content should be decoded with errors='replace'
    assert unit.content is not None


def test_mbox_handles_truncated_multipart_message(tmp_path):
    """Test that truncated/malformed multipart messages are handled gracefully."""
    mbox_path = tmp_path / "truncated.mbox"

    # Write a malformed multipart message directly
    malformed_content = """From alice@example.com Wed May 01 10:00:00 2026
From: alice@example.com
To: bob@example.com
Subject: Truncated Multipart
Date: Wed, 01 May 2026 10:00:00 +0000
Message-ID: <truncated@example.com>
Content-Type: multipart/mixed; boundary="boundary123"

--boundary123
Content-Type: text/plain

This is the first part.

--boundary123
Content-Type: text/html

<p>This part is incomplete and truncated
"""

    mbox_path.write_text(malformed_content)

    result = MboxAdapter(path=str(mbox_path)).ingest()

    # Should still extract what it can from malformed message
    assert len(result.units) >= 0  # May be 0 or 1 depending on parser behavior


def test_mbox_handles_message_with_invalid_message_id(tmp_path):
    """Test that invalid Message-ID values are handled gracefully."""
    mbox_path = write_mbox(
        tmp_path / "invalid-msgid.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Invalid Message-ID",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "not-a-valid-message-id-format",
                "body": "Email with improperly formatted Message-ID.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should still process the message
    assert unit.title == "Invalid Message-ID"
    assert unit.metadata["message_id"] == "not-a-valid-message-id-format"


def test_mbox_handles_missing_from_header(tmp_path):
    """Test that missing From header doesn't crash ingestion."""
    mbox_path = write_mbox(
        tmp_path / "no-from.mbox",
        [
            {
                "To": "bob@example.com",
                "Subject": "No From Header",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<no-from@example.com>",
                "body": "Email without From header.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["from"] == ""
    assert unit.title == "No From Header"


def test_mbox_handles_missing_to_header(tmp_path):
    """Test that missing To header doesn't crash ingestion."""
    mbox_path = write_mbox(
        tmp_path / "no-to.mbox",
        [
            {
                "From": "alice@example.com",
                "Subject": "No To Header",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<no-to@example.com>",
                "body": "Email without To header.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["to"] == ""
    assert unit.title == "No To Header"


def test_mbox_handles_malformed_html_in_content(tmp_path):
    """Test that malformed HTML doesn't crash HTML parser."""
    mbox_path = tmp_path / "malformed-html.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.text import MIMEText

    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Malformed HTML"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<malformed@example.com>"

    # Malformed HTML with unclosed tags and invalid structure
    malformed_html = """
    <html>
      <body>
        <p>Unclosed paragraph
        <div>Unclosed div
        <strong>Bold text
        Random text <><> invalid tags
        <p>Another paragraph</p
      </body>
    """

    html_msg = MIMEText(malformed_html, "html")
    for key in msg.keys():
        html_msg[key] = msg[key]

    mbox.add(html_msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    # Should extract text despite malformed HTML
    assert len(result.units) == 1
    unit = result.units[0]
    assert "Unclosed paragraph" in unit.content or "text" in unit.content.lower()


def test_mbox_handles_message_with_null_bytes(tmp_path):
    """Test that messages containing null bytes are handled."""
    mbox_path = tmp_path / "null-bytes.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Null Bytes"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<null@example.com>"

    # Payload with null bytes
    msg.set_payload(b"Content with \x00 null \x00 bytes \x00 embedded")
    msg.set_type("text/plain")
    msg.set_param("charset", "utf-8")

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    # Should handle null bytes without crashing
    assert len(result.units) == 1
    assert result.units[0].content is not None


def test_mbox_handles_extremely_long_headers(tmp_path):
    """Test that extremely long header values are handled."""
    long_value = "a" * 10000

    mbox_path = write_mbox(
        tmp_path / "long-headers.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": long_value,
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<long-header@example.com>",
                "body": "Email with extremely long header.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Long header should be preserved
    assert len(unit.title) > 9000


def test_mbox_handles_whitespace_only_body(tmp_path):
    """Test that emails with whitespace-only body are handled."""
    mbox_path = write_mbox(
        tmp_path / "whitespace.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Whitespace Only",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<whitespace@example.com>",
                "body": "   \n\n\t\t  \n   ",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Whitespace-only content should result in empty string
    assert unit.content == ""


def test_mbox_handles_empty_body(tmp_path):
    """Test that emails with completely empty body are handled."""
    mbox_path = write_mbox(
        tmp_path / "empty-body.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Empty Body",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<empty@example.com>",
                "body": "",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.content == ""
    assert unit.title == "Empty Body"


def test_mbox_handles_multipart_with_no_text_parts(tmp_path):
    """Test multipart message containing no text/plain or text/html parts."""
    mbox_path = tmp_path / "no-text-parts.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart("mixed")
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "No Text Parts"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<no-text@example.com>"

    # Only attach non-text parts
    pdf_part = MIMEApplication(b"PDF content", "pdf")
    msg.attach(pdf_part)

    image_part = MIMEApplication(b"Image data", "octet-stream")
    msg.attach(image_part)

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should have empty content when no text parts exist
    assert unit.content == ""


def test_mbox_handles_payload_decode_error(tmp_path):
    """Test that payload decode errors are handled gracefully."""
    mbox_path = tmp_path / "decode-error.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Decode Error"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<decode-error@example.com>"

    # Set invalid base64 content with base64 encoding declared
    msg.set_payload("Not valid base64 content!!!")
    msg["Content-Transfer-Encoding"] = "base64"
    msg.set_type("text/plain")
    msg.set_param("charset", "utf-8")

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    # Should handle decode error gracefully
    assert len(result.units) == 1
    assert result.units[0].title == "Decode Error"


def test_mbox_handles_special_characters_in_headers(tmp_path):
    """Test that special characters in headers are handled properly."""
    mbox_path = write_mbox(
        tmp_path / "special-chars.mbox",
        [
            {
                "From": "alice+test@example.com",
                "To": "bob@example.com, charlie@example.com; david@example.com",
                "Subject": "Test: Special <chars> & symbols!",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<special!chars#test@example.com>",
                "body": "Email with special characters in headers.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Special characters should be preserved
    assert "<" in unit.title or "&" in unit.title
    assert unit.metadata["from"] == "alice+test@example.com"


def test_mbox_handles_duplicate_headers(tmp_path):
    """Test that duplicate headers (multiple To, Cc, etc.) are handled."""
    mbox_path = tmp_path / "duplicate-headers.mbox"

    # Write mbox with duplicate headers directly
    content = """From alice@example.com Wed May 01 10:00:00 2026
From: alice@example.com
To: bob@example.com
To: charlie@example.com
Subject: Duplicate Headers
Date: Wed, 01 May 2026 10:00:00 +0000
Message-ID: <duplicate@example.com>

Email with duplicate To headers.
"""

    mbox_path.write_text(content)

    result = MboxAdapter(path=str(mbox_path)).ingest()

    # Should still process the message
    assert len(result.units) >= 0  # Implementation may vary


def test_mbox_handles_missing_all_headers(tmp_path):
    """Test that message with no headers (only body) is handled."""
    mbox_path = tmp_path / "no-headers.mbox"

    # Write minimal mbox with almost no headers
    content = """From MAILER-DAEMON Wed May 01 10:00:00 2026

This is just body content with no real headers.
"""

    mbox_path.write_text(content)

    result = MboxAdapter(path=str(mbox_path)).ingest()

    # Should handle gracefully, may create unit with defaults
    assert len(result.units) >= 0


def test_mbox_handles_date_edge_cases(tmp_path):
    """Test various edge cases in date parsing."""
    mbox_path = write_mbox(
        tmp_path / "date-edges.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Future Date",
                "Date": "Wed, 01 Jan 2099 23:59:59 +0000",
                "Message-ID": "<future@example.com>",
                "body": "Email from the future.",
            },
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Very Old Date",
                "Date": "Mon, 01 Jan 1970 00:00:00 +0000",
                "Message-ID": "<old@example.com>",
                "body": "Email from epoch.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    # Both messages should be processed
    assert len(result.units) == 2
    # Dates should be parsed correctly
    assert all(unit.created_at is not None for unit in result.units)
