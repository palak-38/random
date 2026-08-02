from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from config import (
    AGENT_ENABLED,
    CACHE_DIR,
    CALL_DELAY_SECONDS,
    GATED_MUTE_CONFIDENCE,
    MAX_EVIDENCE,
    PROMPT_VERSION,
    ROUTING_CACHE,
)
from context_builder import ContextBuilder
from data_store import DataStore
from safety_gate import should_hard_mute
from schemas import Action, IncomingMessage, MessageType, RoutingDecision


def load_cache(path: Path = ROUTING_CACHE) -> dict[str, RoutingDecision]:
    """Decisions made under an older prompt version are ignored, not reused."""
    if not path.exists():
        return {}
    out: dict[str, RoutingDecision] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("prompt_version") != PROMPT_VERSION:
            continue
        decision = RoutingDecision.model_validate(record)
        out[decision.message_id] = decision
    return out


def append_cache(decision: RoutingDecision, path: Path = ROUTING_CACHE) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False) + "\n")


def _merge_evidence(ctx, extra) -> str:
    """Seeded fused evidence first, then anything the agent turned up, capped as before.

    The deterministic retrieval stays the floor: the agent can add to the citation
    list but never displaces a seeded candidate.
    """
    seen = {c.message_id for c in ctx.evidence.candidates}
    merged = list(ctx.evidence.candidates)
    for cand in sorted(extra, key=lambda c: -c.score):
        if cand.message_id not in seen and cand.message_id != ctx.message.message_id:
            merged.append(cand)
            seen.add(cand.message_id)
    ids = [c.message_id for c in merged[:MAX_EVIDENCE]]
    return ";".join(ids) if ids else "none"


def decide(msg: IncomingMessage, builder: ContextBuilder, llm_router, tools_factory=None) -> RoutingDecision:
    ctx = builder.build(msg)
    evidence_ids = ctx.evidence.as_csv_field()

    gated, gate_reason = should_hard_mute(ctx.features.safety)
    if gated:
        # The LLM still classifies, but cannot change the action.
        try:
            llm = llm_router.route(ctx)
            message_type = llm.message_type
            if message_type not in {MessageType.scam, MessageType.spam}:
                message_type = MessageType.scam
        except Exception:  # noqa: BLE001 - safety decision must not depend on the LLM
            message_type = MessageType.scam
        return RoutingDecision(
            message_id=msg.message_id,
            action=Action.mute,
            message_type=message_type,
            reason=gate_reason,
            confidence=GATED_MUTE_CONFIDENCE,
            evidence_message_ids=evidence_ids,
            decided_by="safety_gate",
            prompt_version=PROMPT_VERSION,
            provider=llm_router.provider,
            model=llm_router.model,
        )

    rounds = 0
    if AGENT_ENABLED and tools_factory is not None:
        llm, extra, rounds = llm_router.route_agentic(ctx, tools_factory(msg))
        if extra:
            evidence_ids = _merge_evidence(ctx, extra)
    else:
        llm = llm_router.route(ctx)

    return RoutingDecision(
        message_id=msg.message_id,
        action=llm.action,
        message_type=llm.message_type,
        reason=llm.reason,
        confidence=llm.confidence,
        evidence_message_ids=evidence_ids,
        decided_by="llm",
        prompt_version=PROMPT_VERSION,
        provider=llm_router.provider,
        model=llm_router.model,
        agent_rounds=rounds,
    )


def run(
    store: DataStore,
    builder: ContextBuilder,
    llm_router,
    tools_factory=None,
    table: str = "messages",
    limit: int | None = None,
    cache_path: Path = ROUTING_CACHE,
) -> dict[str, RoutingDecision]:
    """Sequential by design: rate limits are real, and a paused run must be able to
    resume from exactly where it stopped."""
    messages = store.incoming_messages(table)
    if limit:
        messages = messages[:limit]

    done = load_cache(cache_path)
    pending = [m for m in messages if m.message_id not in done]
    print(f"{len(messages)} messages, {len(done)} already cached, {len(pending)} to process")

    for i, msg in enumerate(pending, 1):
        try:
            decision = decide(msg, builder, llm_router, tools_factory)
        except Exception as exc:  # noqa: BLE001 - stop, do not fabricate a decision
            print(f"\nSTOPPED at {msg.message_id} ({i}/{len(pending)}): {type(exc).__name__}: {exc}")
            print(f"{len(done)} decisions are safe in {cache_path}. Re-run to resume from here.")
            sys.exit(1)

        append_cache(decision, cache_path)
        done[msg.message_id] = decision
        print(f"[{i}/{len(pending)}] {msg.message_id} -> {decision.action.value}/{decision.message_type.value} ({decision.decided_by})")
        if i < len(pending) and CALL_DELAY_SECONDS:
            time.sleep(CALL_DELAY_SECONDS)

    return done
