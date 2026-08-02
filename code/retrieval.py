from __future__ import annotations

import re
import sqlite3

from config import MAX_EVIDENCE, MIN_VECTOR_SIMILARITY, SQL_MATCH_SCORE
from data_store import DataStore, _to_bool
from embeddings import SimilarityIndex
from schemas import EvidenceCandidate, HistoricalEvidence, IncomingMessage

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class EvidenceRetriever:
    """Fuses two independent retrieval branches over the same user-scoped pool:

    - SQL branch: structured matches (same sender/business, near-duplicate text,
      past messages this user reported or muted).
    - Vector branch: semantic similarity via local MiniLM embeddings.

    Both branches are hard-scoped to the receiving user, so evidence can never
    point at another user's history.
    """

    def __init__(self, store: DataStore):
        self.store = store
        self._history_cache: dict[str, list[sqlite3.Row]] = {}
        self._index_cache: dict[str, SimilarityIndex] = {}

    def _history(self, user_id: str) -> list[sqlite3.Row]:
        if user_id not in self._history_cache:
            self._history_cache[user_id] = self.store.user_history(user_id)
        return self._history_cache[user_id]

    def _index(self, user_id: str) -> SimilarityIndex:
        if user_id not in self._index_cache:
            rows = self._history(user_id)
            self._index_cache[user_id] = SimilarityIndex.build(
                [(r["message_id"], r["message_text"] or "") for r in rows]
            )
        return self._index_cache[user_id]

    def _sql_branch(self, msg: IncomingMessage, query_text: str) -> dict[str, tuple[float, str]]:
        """Structured matches. Returns {message_id: (score, note)}."""
        hits: dict[str, tuple[float, str]] = {}
        query_tokens = _tokens(query_text)

        same_sender_sql = """
            SELECT h.*, e.message_reported, e.muted_after_message, e.notification_dismissed
            FROM message_history h
            LEFT JOIN message_events e
              ON e.message_id = h.message_id AND e.user_id = h.user_id
            WHERE h.user_id = ?
        """
        params: list[str] = [msg.user_id]
        if msg.business_id:
            same_sender_sql += " AND h.business_id = ?"
            params.append(msg.business_id)
        elif msg.sender_user_id:
            same_sender_sql += " AND h.sender_user_id = ?"
            params.append(msg.sender_user_id)
        elif msg.group_id:
            same_sender_sql += " AND h.group_id = ?"
            params.append(msg.group_id)
        else:
            return hits

        for row in self.store.query(same_sender_sql, tuple(params)):
            mid = row["message_id"]
            overlap = _jaccard(query_tokens, _tokens(row["message_text"] or ""))
            notes = []
            score = 0.0

            if overlap >= 0.6:
                score = SQL_MATCH_SCORE
                notes.append("near-duplicate of this message from the same sender")
            elif overlap >= 0.35:
                score = SQL_MATCH_SCORE - 0.15
                notes.append("similar earlier message from the same sender")

            if _to_bool(row["message_reported"]):
                score = max(score, SQL_MATCH_SCORE + 0.05)
                notes.append("user reported this sender before")
            elif _to_bool(row["muted_after_message"]):
                score = max(score, SQL_MATCH_SCORE)
                notes.append("user muted this sender after an earlier message")
            elif _to_bool(row["notification_dismissed"]) and overlap >= 0.2:
                score = max(score, SQL_MATCH_SCORE - 0.25)
                notes.append("user dismissed a similar earlier message")

            if score > 0:
                hits[mid] = (score, "; ".join(notes))
        return hits

    def retrieve(self, msg: IncomingMessage, media_text: str = "") -> HistoricalEvidence:
        query_text = " ".join(p for p in [msg.message_text, media_text] if p).strip()
        rows = {r["message_id"]: r for r in self._history(msg.user_id)}
        if not rows:
            return HistoricalEvidence()

        sql_hits = self._sql_branch(msg, query_text)

        vector_hits: dict[str, float] = {}
        if query_text:
            for mid, score in self._index(msg.user_id).search(query_text, allowed_ids=set(rows), top_k=MAX_EVIDENCE * 3):
                if score >= MIN_VECTOR_SIMILARITY:
                    vector_hits[mid] = score

        candidates: list[EvidenceCandidate] = []
        for mid in set(sql_hits) | set(vector_hits):
            row = rows.get(mid)
            if row is None or mid == msg.message_id:
                continue
            sql_score, note = sql_hits.get(mid, (0.0, ""))
            vec_score = vector_hits.get(mid, 0.0)

            if mid in sql_hits and mid in vector_hits:
                source = "both"
                score = max(sql_score, vec_score) + 0.05
            elif mid in sql_hits:
                source = "sql_exact"
                score = sql_score
            else:
                source = "vector_semantic"
                score = vec_score
                note = "semantically similar earlier message"

            candidates.append(
                EvidenceCandidate(
                    message_id=mid,
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
            )

        candidates.sort(key=lambda c: (-c.score, c.message_id))
        return HistoricalEvidence(candidates=candidates[:MAX_EVIDENCE])
