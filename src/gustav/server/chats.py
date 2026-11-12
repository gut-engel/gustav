"""Chats, as the API and the web interface see them.

The store already had sessions and turns — the wake phase has been writing them since
stage 1. This is the view an account gets of them: your own chats, in order, with their
titles, and no way to read anyone else's.

The ownership rule is enforced in the query, not after it. ``for_user`` builds a WHERE
clause that cannot return another account's chat, so a missing check in a handler is a
404 rather than a leak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from gustav.store.db import Database

#: A chat's title is the first thing the user said, trimmed to something readable.
TITLE_LENGTH = 60


@dataclass
class Chat:
    id: int
    title: str
    source: str
    started_at: str
    last_activity_at: str
    messages: int = 0
    user_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "messages": self.messages,
        }


def title_from(message: str) -> str:
    text = " ".join((message or "").split())
    if len(text) <= TITLE_LENGTH:
        return text or "new chat"
    return text[: TITLE_LENGTH - 1].rstrip() + "…"


class Chats:
    def __init__(self, db: Database):
        self.db = db

    # --- ownership ------------------------------------------------------
    @staticmethod
    def _scope(user_id: int | None) -> tuple[str, list[Any]]:
        """The WHERE fragment that limits a query to one account's chats.

        ``None`` means the owner over the socket, whose chats are the ones with no user
        attached — including every chat that existed before accounts did.
        """
        if user_id is None:
            return "s.user_id IS NULL", []
        return "s.user_id = ?", [user_id]

    def create(self, *, user_id: int | None, title: str = "", source: str = "web") -> Chat:
        chat_id = self.db.insert(
            "sessions",
            {"source": source, "label": title or "new chat", "user_id": user_id},
        )
        # The wake phase finds a conversation by its external id, so give the chat one
        # that cannot collide with a CLI session or another account's.
        self.db.execute(
            "UPDATE sessions SET external_id = ? WHERE id = ?", (f"{source}-{chat_id}", chat_id)
        )
        return self.get(chat_id, user_id=user_id)

    def key(self, chat_id: int, *, user_id: int | None) -> str:
        """The session key the daemon's chat path expects, if this chat is yours."""
        self.get(chat_id, user_id=user_id)
        row = self.db.query_one("SELECT external_id FROM sessions WHERE id = ?", (chat_id,))
        return (row["external_id"] if row else None) or f"web-{chat_id}"

    def get(self, chat_id: int, *, user_id: int | None) -> Chat:
        scope, params = self._scope(user_id)
        row = self.db.query_one(
            f"SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id)"
            f" AS messages FROM sessions s WHERE s.id = ? AND {scope}",
            [chat_id, *params],
        )
        if row is None:
            raise KeyError(f"there is no chat {chat_id}")
        return self._to_chat(row)

    def all(self, *, user_id: int | None, limit: int = 50) -> list[Chat]:
        scope, params = self._scope(user_id)
        rows = self.db.query(
            f"SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id)"
            f" AS messages FROM sessions s WHERE {scope} AND s.archived = 0"
            f" ORDER BY s.last_activity_at DESC, s.id DESC LIMIT ?",
            [*params, limit],
        )
        return [self._to_chat(row) for row in rows]

    def messages(self, chat_id: int, *, user_id: int | None, limit: int = 200) -> list[dict]:
        self.get(chat_id, user_id=user_id)  # ownership, or KeyError
        rows = self.db.query(
            "SELECT id, role, content, ts, redacted, trainable, adapter_versions,"
            " tokens_per_second, completion_tokens FROM turns WHERE session_id = ?"
            " ORDER BY id LIMIT ?",
            (chat_id, limit),
        )
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["adapters"] = json.loads(item.pop("adapter_versions") or "{}")
            except (ValueError, TypeError):
                item["adapters"] = {}
            item["feedback"] = self.feedback_for(int(row["id"]))
            out.append(item)
        return out

    def feedback_for(self, turn_id: int) -> list[dict[str, Any]]:
        return [
            {"kind": row["kind"], "payload": row["payload"], "ts": row["ts"]}
            for row in self.db.query(
                "SELECT kind, payload, ts FROM feedback WHERE turn_id = ?"
                " AND consumed_run_id IS NOT -1 ORDER BY id",
                (turn_id,),
            )
        ]

    def rename(self, chat_id: int, title: str, *, user_id: int | None) -> Chat:
        self.get(chat_id, user_id=user_id)
        self.db.execute(
            "UPDATE sessions SET label = ? WHERE id = ?", (title[:200], chat_id)
        )
        return self.get(chat_id, user_id=user_id)

    def title_if_unset(self, chat_id: int, message: str) -> None:
        """Name a chat after its first message, once."""
        row = self.db.query_one("SELECT label FROM sessions WHERE id = ?", (chat_id,))
        if row is not None and (not row["label"] or row["label"] == "new chat"):
            self.db.execute(
                "UPDATE sessions SET label = ? WHERE id = ?",
                (title_from(message), chat_id),
            )

    def archive(self, chat_id: int, *, user_id: int | None) -> None:
        """Hide a chat. The turns stay: they are what the night learns from."""
        self.get(chat_id, user_id=user_id)
        self.db.execute("UPDATE sessions SET archived = 1 WHERE id = ?", (chat_id,))

    def touch(self, chat_id: int) -> None:
        """Mark a chat as the one you were just in.

        Sub-second, because the column's own default is only second-accurate and two
        chats made in the same second would otherwise never reorder. The formats still
        compare correctly as text: ``…:07.123`` sorts after ``…:07``.
        """
        self.db.execute(
            "UPDATE sessions SET last_activity_at = strftime('%Y-%m-%d %H:%M:%f', 'now')"
            " WHERE id = ?",
            (chat_id,),
        )

    @staticmethod
    def _to_chat(row: Any) -> Chat:
        return Chat(
            id=int(row["id"]),
            title=row["label"] or "new chat",
            source=row["source"],
            started_at=row["started_at"],
            last_activity_at=row["last_activity_at"],
            messages=int(row["messages"] if "messages" in row.keys() else 0),
            user_id=row["user_id"],
        )
