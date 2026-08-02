from __future__ import annotations

import csv
from pathlib import Path

from config import FALLBACK_CONFIDENCE, OUTPUT_CSV, PROMPT_VERSION, ROUTING_CACHE
from data_store import DataStore
from pipeline import load_cache
from schemas import Action, MessageType, RoutingDecision

COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def _fallback(message_id: str) -> RoutingDecision:
    """digest is the safe default: it neither suppresses something urgent nor
    interrupts the user on no information."""
    return RoutingDecision(
        message_id=message_id,
        action=Action.digest,
        message_type=MessageType.unknown,
        reason="Routing could not be completed for this message; held for later review.",
        confidence=FALLBACK_CONFIDENCE,
        evidence_message_ids="none",
        decided_by="fallback",
        prompt_version=PROMPT_VERSION,
    )


def finalize(
    store: DataStore,
    table: str = "messages",
    cache_path: Path = ROUTING_CACHE,
    output_path: Path = OUTPUT_CSV,
) -> tuple[int, int]:
    """Writes output.csv with exactly one row per input message. Any message still
    missing from the cache is backfilled with a safe default."""
    messages = store.incoming_messages(table)
    decisions = load_cache(cache_path)

    rows = []
    missing = 0
    for msg in messages:
        decision = decisions.get(msg.message_id)
        if decision is None:
            decision = _fallback(msg.message_id)
            missing += 1
        rows.append(decision.as_output_row())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), missing
