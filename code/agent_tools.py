"""Tools the routing agent may call to look for better evidence.

Both tools are bound to one incoming message when constructed and take no
user_id argument, so the "evidence never references another user's history"
invariant is structural rather than something the model has to respect.
"""

from __future__ import annotations

from config import MAX_EVIDENCE, MIN_VECTOR_SIMILARITY
from data_store import DataStore, _to_bool
from retrieval import EvidenceRetriever
from schemas import EvidenceCandidate, IncomingMessage

MAX_TOOL_RESULTS = 5


def _summarise(candidates: list[EvidenceCandidate]) -> str:
    if not candidates:
        return "No matching messages in this user's history."
    lines = []
    for c in candidates:
        flags = [
            name
            for name, on in [
                ("opened", c.was_opened),
                ("replied", c.was_replied),
                ("dismissed", c.was_dismissed),
                ("reported", c.was_reported),
                ("muted after", c.muted_after),
            ]
            if on
        ]
        text = " ".join((c.message_text or "").split())[:160]
        lines.append(
            f"[{c.message_id}] {c.created_at} | user: {', '.join(flags) or 'no reaction'}\n    \"{text}\""
        )
    return "\n".join(lines)


class EvidenceTools:
    """Tool surface for one message. Construct per message being routed."""

    def __init__(self, retriever: EvidenceRetriever, store: DataStore, msg: IncomingMessage):
        self._retriever = retriever
        self._store = store
        self._msg = msg
        self.calls: list[dict[str, str]] = []

    def _rows(self) -> dict:
        return {r["message_id"]: r for r in self._retriever._history(self._msg.user_id)}

    def _to_candidate(self, row, score: float, source: str, note: str) -> EvidenceCandidate:
        return EvidenceCandidate(
            message_id=row["message_id"],
            message_text=(row["message_text"] or "")[:300],
            created_at=row["created_at"],
            source=source,
            score=round(score, 4),
            was_opened=_to_bool(row["message_opened"]),
            was_replied=_to_bool(row["message_replied"]),
            was_dismissed=_to_bool(row["notification_dismissed"]),
            was_reported=_to_bool(row["message_reported"]),
            muted_after=_to_bool(row["muted_after_message"]),
            match_note=note,
        )

    def search_history(self, query: str) -> tuple[str, list[EvidenceCandidate]]:
        """Semantic search over this user's own history, using the agent's own wording."""
        self.calls.append({"tool": "search_history", "query": query})
        rows = self._rows()
        if not rows or not query.strip():
            return "No matching messages in this user's history.", []

        found: list[EvidenceCandidate] = []
        index = self._retriever._index(self._msg.user_id)
        for mid, score in index.search(query, allowed_ids=set(rows), top_k=MAX_TOOL_RESULTS):
            if score < MIN_VECTOR_SIMILARITY or mid == self._msg.message_id:
                continue
            found.append(self._to_candidate(rows[mid], score, "vector_semantic", f"matched search: {query[:60]}"))
        return _summarise(found), found

    def find_messages_from_sender(self) -> tuple[str, list[EvidenceCandidate]]:
        """Every earlier message this user received from the same sender, with their reactions."""
        self.calls.append({"tool": "find_messages_from_sender", "query": ""})
        msg = self._msg
        sql = """
            SELECT h.*, e.message_opened, e.message_replied, e.notification_dismissed,
                   e.muted_after_message, e.message_reported
            FROM message_history h
            LEFT JOIN message_events e
              ON e.message_id = h.message_id AND e.user_id = h.user_id
            WHERE h.user_id = ?
        """
        params: list[str] = [msg.user_id]
        if msg.business_id:
            sql += " AND h.business_id = ?"
            params.append(msg.business_id)
        elif msg.sender_user_id:
            sql += " AND h.sender_user_id = ?"
            params.append(msg.sender_user_id)
        elif msg.group_id:
            sql += " AND h.group_id = ?"
            params.append(msg.group_id)
        else:
            return "This message has no identifiable sender to look up.", []

        sql += " ORDER BY h.created_at DESC LIMIT ?"
        params.append(str(MAX_TOOL_RESULTS))

        found = [
            self._to_candidate(row, 0.7, "sql_exact", "earlier message from the same sender")
            for row in self._store.query(sql, tuple(params))
            if row["message_id"] != msg.message_id
        ]
        return _summarise(found), found


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "Search this user's own message history for messages similar to a phrase you choose. "
                "Use it when the evidence you were given looks unrelated to the incoming message, "
                "wording your query around the meaning you actually want to match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Words describing the kind of past message to find.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_messages_from_sender",
            "description": (
                "List this user's earlier messages from the same sender, group, or business, "
                "with how the user reacted. Use it to check for repetition, prior reports, or "
                "an established relationship."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
