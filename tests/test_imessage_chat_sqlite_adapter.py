from __future__ import annotations

import sqlite3

from graph.adapters.imessage_chat_sqlite import ImessageChatSqliteAdapter


def test_imessage_chat_sqlite_reads_messages_and_converts_apple_timestamps(tmp_path):
    db = tmp_path / "chat.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
            CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT, text TEXT, date INTEGER, date_read INTEGER, is_from_me INTEGER, is_read INTEGER, service TEXT, handle_id INTEGER);
            CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT);
            CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
            CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT);
            CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
            INSERT INTO handle VALUES (1, '+15551234567');
            INSERT INTO message VALUES (1, 'm1', 'hello', 631152000000000000, 631152060000000000, 0, 1, 'iMessage', 1);
            INSERT INTO chat VALUES (1, 'chat-1');
            INSERT INTO chat_message_join VALUES (1, 1);
            INSERT INTO attachment VALUES (1, '/tmp/photo.jpg');
            INSERT INTO message_attachment_join VALUES (1, 1);
            """
        )

    unit = ImessageChatSqliteAdapter(path=str(db)).ingest().units[0]

    assert unit.source_id == "imessage_chat_sqlite:message:m1"
    assert unit.metadata["handle"] == "+15551234567"
    assert unit.metadata["chat_ids"] == ["chat-1"]
    assert unit.metadata["text"] == "hello"
    assert unit.metadata["service"] == "iMessage"
    assert unit.metadata["is_read"] is True
    assert unit.metadata["attachments"] == ["/tmp/photo.jpg"]
    assert unit.metadata["sent_at"] == "2021-01-01T00:00:00+00:00"


def test_imessage_chat_sqlite_represents_attachment_only_messages(tmp_path):
    db = tmp_path / "chat.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT, text TEXT, date INTEGER, is_from_me INTEGER, is_read INTEGER, service TEXT);
            CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT);
            CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
            INSERT INTO message VALUES (1, 'm2', NULL, 631152000, 1, 0, 'SMS');
            INSERT INTO attachment VALUES (1, '/tmp/file.pdf');
            INSERT INTO message_attachment_join VALUES (1, 1);
            """
        )

    unit = ImessageChatSqliteAdapter(path=str(db)).ingest().units[0]

    assert unit.title == "Message attachment: /tmp/file.pdf"
    assert unit.metadata["attachments"] == ["/tmp/file.pdf"]
