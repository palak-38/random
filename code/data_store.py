from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from config import DATASET_DIR, SQLITE_DB
from schemas import (
    BusinessProfile,
    BusinessRelationship,
    GroupMembership,
    GroupProfile,
    IncomingMessage,
    MediaContent,
    UserContext,
)

TABLES = [
    "messages",
    "users",
    "groups",
    "group_members",
    "business_accounts",
    "user_business_history",
    "message_history",
    "message_events",
    "images",
    "voice_notes",
    "daily_notification_summary",
    "sample_messages",
]


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if any(c.strip() for c in r)]
    return header, rows


def build_db(db_path: Path = SQLITE_DB, dataset_dir: Path = DATASET_DIR) -> sqlite3.Connection:
    """Load every dataset CSV into a fresh SQLite database. All columns are TEXT;
    typing happens at the Pydantic boundary."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for table in TABLES:
        csv_path = dataset_dir / f"{table}.csv"
        if not csv_path.exists():
            continue
        header, rows = _read_csv(csv_path)
        cols = ", ".join(f'"{c}" TEXT' for c in header)
        conn.execute(f'CREATE TABLE "{table}" ({cols})')
        placeholders = ", ".join("?" for _ in header)
        conn.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', rows)

    conn.execute("CREATE INDEX idx_hist_user ON message_history(user_id)")
    conn.execute("CREATE INDEX idx_events_msg ON message_events(user_id, message_id)")
    conn.execute("CREATE INDEX idx_members ON group_members(group_id, user_id)")
    conn.commit()
    return conn


def connect(db_path: Path = SQLITE_DB) -> sqlite3.Connection:
    if not db_path.exists():
        return build_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes"}


def _blank_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class DataStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    # ---------- incoming messages ----------

    def incoming_messages(self, table: str = "messages") -> list[IncomingMessage]:
        # rowid preserves the order rows appear in the CSV, so output.csv lines up with the input.
        rows = self.query(f'SELECT * FROM "{table}" ORDER BY rowid')
        return [self._to_incoming(r) for r in rows]

    @staticmethod
    def _to_incoming(row: sqlite3.Row) -> IncomingMessage:
        media_type = _blank_to_none(row["media_type"])
        return IncomingMessage(
            message_id=row["message_id"],
            user_id=row["user_id"],
            conversation_type=row["conversation_type"],
            group_id=_blank_to_none(row["group_id"]),
            business_id=_blank_to_none(row["business_id"]),
            sender_user_id=_blank_to_none(row["sender_user_id"]),
            created_at=row["created_at"],
            message_text=row["message_text"] or "",
            media_type=media_type if media_type in {"image", "voice"} else None,
            media_id=_blank_to_none(row["media_id"]),
            forwarded_count=_to_int(row["forwarded_count"]),
        )

    # ---------- context lookups ----------

    def user_context(self, user_id: str) -> UserContext:
        row = self.one("SELECT * FROM users WHERE user_id = ?", (user_id,))
        summary = self.one(
            """SELECT AVG(CAST(notifications_sent AS FLOAT)) AS sent,
                      AVG(CAST(notifications_dismissed AS FLOAT)) AS dismissed
               FROM daily_notification_summary WHERE user_id = ?""",
            (user_id,),
        )
        if row is None:
            return UserContext(user_id=user_id)
        return UserContext(
            user_id=user_id,
            do_not_disturb_window=_blank_to_none(row["do_not_disturb_window"]),
            messages_opened_30d=_to_int(row["messages_opened_30d"]),
            messages_replied_30d=_to_int(row["messages_replied_30d"]),
            notifications_dismissed_30d=_to_int(row["notifications_dismissed_30d"]),
            messages_reported_30d=_to_int(row["messages_reported_30d"]),
            avg_daily_notifications=_to_float(summary["sent"] if summary else 0),
            avg_daily_dismissed=_to_float(summary["dismissed"] if summary else 0),
        )

    def group_profile(self, group_id: str) -> GroupProfile | None:
        row = self.one("SELECT * FROM groups WHERE group_id = ?", (group_id,))
        if row is None:
            return None
        return GroupProfile(
            group_id=group_id,
            group_name=row["group_name"] or "",
            group_type=row["group_type"] or "",
            member_count=_to_int(row["member_count"]),
            admin_count=_to_int(row["admin_count"]),
            created_at=_blank_to_none(row["created_at"]),
            messages_30d=_to_int(row["messages_30d"]),
        )

    def group_membership(self, group_id: str, user_id: str) -> GroupMembership | None:
        row = self.one(
            "SELECT * FROM group_members WHERE group_id = ? AND user_id = ?",
            (group_id, user_id),
        )
        if row is None:
            return None
        return GroupMembership(
            group_id=group_id,
            user_id=user_id,
            role=row["role"] or "member",
            joined_at=_blank_to_none(row["joined_at"]),
            messages_sent_30d=_to_int(row["messages_sent_30d"]),
            messages_read_30d=_to_int(row["messages_read_30d"]),
            replies_sent_30d=_to_int(row["replies_sent_30d"]),
            notifications_dismissed_30d=_to_int(row["notifications_dismissed_30d"]),
            group_muted_by_user=_to_bool(row["group_muted_by_user"]),
        )

    def business_profile(self, business_id: str) -> BusinessProfile | None:
        row = self.one("SELECT * FROM business_accounts WHERE business_id = ?", (business_id,))
        if row is None:
            return None
        return BusinessProfile(
            business_id=business_id,
            display_name=row["display_name"] or "",
            brand_name=row["brand_name"] or "",
            category=row["category"] or "",
            verified=_to_bool(row["verified"]),
            official_domain=row["official_domain"] or "",
            domain_used_by_sender=row["domain_used_by_sender"] or "",
            account_age_days=_to_int(row["account_age_days"]),
            messages_sent_30d=_to_int(row["messages_sent_30d"]),
            user_reports_30d=_to_int(row["user_reports_30d"]),
            domain_used_by_sender_age_days=_to_int(row["domain_used_by_sender_age_days"]),
        )

    def business_relationship(self, user_id: str, business_id: str) -> BusinessRelationship | None:
        row = self.one(
            "SELECT * FROM user_business_history WHERE user_id = ? AND business_id = ?",
            (user_id, business_id),
        )
        if row is None:
            return None
        return BusinessRelationship(
            user_id=user_id,
            business_id=business_id,
            why_user_knows_account=row["why_user_knows_account"] or "",
            last_activity_at=_blank_to_none(row["last_activity_at"]),
            allows_promotions=_to_bool(row["allows_promotions"]),
            promotions_opted_out_at=_blank_to_none(row["promotions_opted_out_at"]),
            activity_count_180d=_to_int(row["activity_count_180d"]),
            messages_opened_30d=_to_int(row["messages_opened_30d"]),
            messages_dismissed_30d=_to_int(row["messages_dismissed_30d"]),
            messages_replied_30d=_to_int(row["messages_replied_30d"]),
            last_reply_at=_blank_to_none(row["last_reply_at"]),
        )

    def media_content(self, media_id: str, media_type: str) -> MediaContent | None:
        if media_type == "image":
            row = self.one("SELECT file_path FROM images WHERE image_id = ?", (media_id,))
        else:
            row = self.one("SELECT file_path FROM voice_notes WHERE voice_note_id = ?", (media_id,))
        if row is None:
            return None
        return MediaContent(media_id=media_id, media_type=media_type, file_path=row["file_path"])

    def all_media(self) -> list[MediaContent]:
        items = [
            MediaContent(media_id=r["image_id"], media_type="image", file_path=r["file_path"])
            for r in self.query("SELECT * FROM images")
        ]
        items += [
            MediaContent(media_id=r["voice_note_id"], media_type="voice", file_path=r["file_path"])
            for r in self.query("SELECT * FROM voice_notes")
        ]
        return items

    # ---------- history ----------

    def user_history(self, user_id: str) -> list[sqlite3.Row]:
        """All historical messages for this user, joined with how they reacted."""
        return self.query(
            """
            SELECT h.*,
                   e.message_opened, e.message_replied, e.notification_dismissed,
                   e.muted_after_message, e.message_reported
            FROM message_history h
            LEFT JOIN message_events e
              ON e.message_id = h.message_id AND e.user_id = h.user_id
            WHERE h.user_id = ?
            ORDER BY h.created_at
            """,
            (user_id,),
        )
