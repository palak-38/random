from __future__ import annotations

from datetime import datetime

import safety_gate
from data_store import DataStore
from retrieval import EvidenceRetriever
from schemas import (
    BusinessContext,
    GroupContext,
    IncomingMessage,
    MediaAnalysis,
    MessageRoutingContext,
    RoutingFeatures,
)


def _in_quiet_hours(window: str | None, created_at: str) -> bool:
    if not window or "-" not in window:
        return False
    try:
        start_s, end_s = window.split("-")
        start = datetime.strptime(start_s.strip(), "%H:%M").time()
        end = datetime.strptime(end_s.strip(), "%H:%M").time()
        stamp = datetime.strptime(created_at.strip(), "%Y-%m-%d %H:%M").time()
    except ValueError:
        return False
    if start <= end:
        return start <= stamp <= end
    return stamp >= start or stamp <= end  # window wraps midnight


class ContextBuilder:
    def __init__(self, store: DataStore, media_analyses: dict[str, MediaAnalysis]):
        self.store = store
        self.media_analyses = media_analyses
        self.retriever = EvidenceRetriever(store)

    def build(self, msg: IncomingMessage) -> MessageRoutingContext:
        user = self.store.user_context(msg.user_id)
        user.in_quiet_hours = _in_quiet_hours(user.do_not_disturb_window, msg.created_at)

        media = None
        analysis = None
        if msg.media_id and msg.media_type:
            media = self.store.media_content(msg.media_id, msg.media_type)
            analysis = self.media_analyses.get(msg.media_id)

        group = None
        if msg.group_id:
            profile = self.store.group_profile(msg.group_id)
            if profile:
                group = GroupContext(
                    profile=profile,
                    membership=self.store.group_membership(msg.group_id, msg.user_id),
                    sender_membership=(
                        self.store.group_membership(msg.group_id, msg.sender_user_id)
                        if msg.sender_user_id
                        else None
                    ),
                )

        business = None
        if msg.business_id:
            profile = self.store.business_profile(msg.business_id)
            if profile:
                business = BusinessContext(
                    profile=profile,
                    relationship=self.store.business_relationship(msg.user_id, msg.business_id),
                )

        media_text = analysis.as_context() if analysis else ""
        evidence = self.retriever.retrieve(msg, media_text=media_text)

        ctx = MessageRoutingContext(
            message=msg,
            media=media,
            media_analysis=analysis,
            user=user,
            group=group,
            business=business,
            sender_display=self._sender_display(msg, group, business),
            evidence=evidence,
        )
        ctx.features = self._features(ctx)
        return ctx

    @staticmethod
    def _sender_display(msg: IncomingMessage, group: GroupContext | None, business: BusinessContext | None) -> str:
        if business:
            return f"{business.profile.display_name} (business, {business.profile.category})"
        if group and msg.sender_user_id:
            role = group.sender_membership.role if group.sender_membership else "member"
            return f"{msg.sender_user_id} ({role} of {group.profile.group_name})"
        return msg.sender_user_id or "unknown sender"

    def _features(self, ctx: MessageRoutingContext) -> RoutingFeatures:
        msg, user = ctx.message, ctx.user
        f = RoutingFeatures(
            is_quiet_hours=user.in_quiet_hours,
            forwarded_count=msg.forwarded_count,
            notification_load=user.avg_daily_notifications,
        )

        total_seen = user.messages_opened_30d + user.notifications_dismissed_30d
        f.user_open_rate = user.messages_opened_30d / total_seen if total_seen else 0.0
        f.user_dismiss_rate = user.notifications_dismissed_30d / total_seen if total_seen else 0.0

        if ctx.group and ctx.group.membership:
            f.group_muted_by_user = ctx.group.membership.group_muted_by_user
        if ctx.group and ctx.group.sender_membership:
            f.sender_is_admin = ctx.group.sender_membership.role == "admin"

        if ctx.business:
            rel = ctx.business.relationship
            f.has_business_relationship = rel is not None and rel.activity_count_180d > 0
            f.opted_out_of_promotions = rel is not None and (
                bool(rel.promotions_opted_out_at) or not rel.allows_promotions
            )

        f.repetition_count = sum(
            1 for c in ctx.evidence.candidates if c.source in {"sql_exact", "both"} and c.score >= 0.75
        )
        f.safety = safety_gate.evaluate(ctx)
        return f
