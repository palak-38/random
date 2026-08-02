from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Action(str, Enum):
    notify = "notify"
    digest = "digest"
    mute = "mute"


class MessageType(str, Enum):
    personal = "personal"
    urgent = "urgent"
    event = "event"
    payment = "payment"
    business_update = "business_update"
    promotion = "promotion"
    greeting = "greeting"
    forward = "forward"
    spam = "spam"
    scam = "scam"
    unknown = "unknown"


class IncomingMessage(BaseModel):
    message_id: str
    user_id: str
    conversation_type: Literal["personal", "group", "business"]
    group_id: str | None = None
    business_id: str | None = None
    sender_user_id: str | None = None
    created_at: str
    message_text: str = ""
    media_type: Literal["image", "voice"] | None = None
    media_id: str | None = None
    forwarded_count: int = 0


class MediaContent(BaseModel):
    """Raw pointer from images.csv / voice_notes.csv."""

    media_id: str
    media_type: Literal["image", "voice"]
    file_path: str


class MediaAnalysis(BaseModel):
    """Derived by a one-time Gemini vision/ASR pass; cached to disk."""

    media_id: str
    media_type: Literal["image", "voice"]
    transcript: str = ""
    description: str = ""
    detected_urls: list[str] = Field(default_factory=list)
    detected_phone_numbers: list[str] = Field(default_factory=list)
    mentions_payment: bool = False
    analysis_error: str | None = None

    def as_context(self) -> str:
        parts = []
        if self.description:
            parts.append(f"Media description: {self.description}")
        if self.transcript:
            label = "Voice transcript" if self.media_type == "voice" else "Text in image"
            parts.append(f"{label}: {self.transcript}")
        if self.detected_urls:
            parts.append(f"URLs in media: {', '.join(self.detected_urls)}")
        if self.detected_phone_numbers:
            parts.append(f"Phone numbers in media: {', '.join(self.detected_phone_numbers)}")
        if self.mentions_payment:
            parts.append("Media references payment/money.")
        if self.analysis_error:
            parts.append(f"(media analysis unavailable: {self.analysis_error})")
        return "\n".join(parts)


class UserContext(BaseModel):
    user_id: str
    do_not_disturb_window: str | None = None
    messages_opened_30d: int = 0
    messages_replied_30d: int = 0
    notifications_dismissed_30d: int = 0
    messages_reported_30d: int = 0
    # from daily_notification_summary.csv
    avg_daily_notifications: float = 0.0
    avg_daily_dismissed: float = 0.0
    in_quiet_hours: bool = False


class GroupProfile(BaseModel):
    """Entity-level facts from groups.csv."""

    group_id: str
    group_name: str = ""
    group_type: str = ""
    member_count: int = 0
    admin_count: int = 0
    created_at: str | None = None
    messages_30d: int = 0


class GroupMembership(BaseModel):
    """This user's relationship to the group, from group_members.csv."""

    group_id: str
    user_id: str
    role: str = "member"
    joined_at: str | None = None
    messages_sent_30d: int = 0
    messages_read_30d: int = 0
    replies_sent_30d: int = 0
    notifications_dismissed_30d: int = 0
    group_muted_by_user: bool = False


class GroupContext(BaseModel):
    profile: GroupProfile
    membership: GroupMembership | None = None
    sender_membership: GroupMembership | None = None


class BusinessProfile(BaseModel):
    """Entity-level facts from business_accounts.csv."""

    business_id: str
    display_name: str = ""
    brand_name: str = ""
    category: str = ""
    verified: bool = False
    official_domain: str = ""
    domain_used_by_sender: str = ""
    account_age_days: int = 0
    messages_sent_30d: int = 0
    user_reports_30d: int = 0
    domain_used_by_sender_age_days: int = 0


class BusinessRelationship(BaseModel):
    """This user's history with the business, from user_business_history.csv."""

    user_id: str
    business_id: str
    why_user_knows_account: str = ""
    last_activity_at: str | None = None
    allows_promotions: bool = False
    promotions_opted_out_at: str | None = None
    activity_count_180d: int = 0
    messages_opened_30d: int = 0
    messages_dismissed_30d: int = 0
    messages_replied_30d: int = 0
    last_reply_at: str | None = None


class BusinessContext(BaseModel):
    profile: BusinessProfile
    relationship: BusinessRelationship | None = None


class EvidenceCandidate(BaseModel):
    message_id: str
    message_text: str = ""
    created_at: str | None = None
    source: Literal["sql_exact", "vector_semantic", "both"]
    score: float
    # from message_events.csv, when available
    was_opened: bool | None = None
    was_replied: bool | None = None
    was_dismissed: bool | None = None
    was_reported: bool | None = None
    muted_after: bool | None = None
    match_note: str = ""


class HistoricalEvidence(BaseModel):
    candidates: list[EvidenceCandidate] = Field(default_factory=list)

    def message_ids(self) -> list[str]:
        return [c.message_id for c in self.candidates]

    def as_csv_field(self) -> str:
        ids = self.message_ids()
        return ";".join(ids) if ids else "none"


class SafetySignals(BaseModel):
    domain_mismatch: bool = False
    sender_domain_is_new: bool = False
    business_unverified: bool = False
    high_report_count: bool = False
    scam_text_pattern: bool = False
    matched_patterns: list[str] = Field(default_factory=list)
    heavily_forwarded: bool = False
    prior_reported_by_user: bool = False

    def summary(self) -> str:
        flags = [k for k, v in self.model_dump().items() if v is True]
        if self.matched_patterns:
            flags.append(f"patterns={self.matched_patterns}")
        return ", ".join(flags) if flags else "none"


class RoutingFeatures(BaseModel):
    is_quiet_hours: bool = False
    group_muted_by_user: bool = False
    sender_is_admin: bool = False
    user_mentioned: bool = False
    forwarded_count: int = 0
    repetition_count: int = 0
    user_dismiss_rate: float = 0.0
    user_open_rate: float = 0.0
    notification_load: float = 0.0
    opted_out_of_promotions: bool = False
    has_business_relationship: bool = False
    safety: SafetySignals = Field(default_factory=SafetySignals)


class MessageRoutingContext(BaseModel):
    """Everything the decision layer needs for one message."""

    message: IncomingMessage
    media: MediaContent | None = None
    media_analysis: MediaAnalysis | None = None
    user: UserContext
    group: GroupContext | None = None
    business: BusinessContext | None = None
    sender_display: str = ""
    evidence: HistoricalEvidence = Field(default_factory=HistoricalEvidence)
    features: RoutingFeatures = Field(default_factory=RoutingFeatures)


class LLMRoutingDecision(BaseModel):
    """Structured output requested from the LLM."""

    action: Action = Field(description="notify to interrupt now, digest to show later, mute to suppress")
    message_type: MessageType = Field(description="best-fit category for this message")
    reason: str = Field(
        description="One short sentence (max 25 words) explaining the decision in terms of this user's context.",
        max_length=300,
    )
    confidence: float = Field(ge=0.0, le=1.0, description="How confident you are in this routing decision")


class RoutingDecision(BaseModel):
    """Final decision written to output.csv."""

    message_id: str
    action: Action
    message_type: MessageType
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_message_ids: str = "none"
    decided_by: Literal["safety_gate", "llm", "fallback"] = "llm"
    prompt_version: str = "v1"

    def as_output_row(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "action": self.action.value,
            "message_type": self.message_type.value,
            "reason": self.reason.replace("\n", " ").strip(),
            "confidence": f"{self.confidence:.2f}",
            "evidence_message_ids": self.evidence_message_ids or "none",
        }
