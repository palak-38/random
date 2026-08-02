from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Anchored to the repo root so the audit runs from any working directory.
# output.csv is generated from messages.csv, so it must be audited against that
# file - sample_messages.csv uses a separate sample_msg_* id namespace.
messages = pd.read_csv(ROOT / "dataset" / "messages.csv").fillna("")
history = pd.read_csv(ROOT / "dataset" / "message_history.csv").fillna("")
output = pd.read_csv(ROOT / "output.csv").fillna("")

required_columns = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

allowed_actions = {"notify", "digest", "mute"}
allowed_types = {
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
}

errors: list[str] = []

if list(output.columns) != required_columns:
    errors.append(f"Wrong columns/order: {list(output.columns)}")

if len(output) != len(messages):
    errors.append(
        f"Row mismatch: messages={len(messages)}, output={len(output)}"
    )

if output["message_id"].duplicated().any():
    duplicates = output.loc[
        output["message_id"].duplicated(), "message_id"
    ].tolist()
    errors.append(f"Duplicate output IDs: {duplicates}")

input_ids = set(messages["message_id"])
output_ids = set(output["message_id"])

missing_ids = input_ids - output_ids
extra_ids = output_ids - input_ids

if missing_ids:
    errors.append(f"Missing IDs: {sorted(missing_ids)}")

if extra_ids:
    errors.append(f"Unexpected IDs: {sorted(extra_ids)}")

bad_actions = output.loc[
    ~output["action"].isin(allowed_actions),
    ["message_id", "action"],
]
if not bad_actions.empty:
    errors.append(f"Invalid actions:\n{bad_actions}")

bad_types = output.loc[
    ~output["message_type"].isin(allowed_types),
    ["message_id", "message_type"],
]
if not bad_types.empty:
    errors.append(f"Invalid message types:\n{bad_types}")

confidence = pd.to_numeric(output["confidence"], errors="coerce")
bad_confidence = output.loc[
    confidence.isna() | (confidence < 0) | (confidence > 1),
    ["message_id", "confidence"],
]
if not bad_confidence.empty:
    errors.append(f"Invalid confidence:\n{bad_confidence}")

empty_reasons = output.loc[
    output["reason"].str.strip().eq(""),
    "message_id",
].tolist()
if empty_reasons:
    errors.append(f"Empty reasons: {empty_reasons}")

history_user = history.set_index("message_id")["user_id"].to_dict()
message_user = messages.set_index("message_id")["user_id"].to_dict()

invalid_evidence: list[tuple[str, str]] = []
cross_user_evidence: list[tuple[str, str, str, str]] = []

for row in output.itertuples(index=False):
    raw_ids = str(row.evidence_message_ids).strip()

    if not raw_ids or raw_ids == "none":
        continue

    for evidence_id in raw_ids.split(";"):
        evidence_id = evidence_id.strip()

        if evidence_id not in history_user:
            invalid_evidence.append((row.message_id, evidence_id))
            continue

        # An unexpected output id is already recorded above; skip it here so the
        # audit reports every problem instead of dying on the first one.
        if row.message_id not in message_user:
            continue

        current_user = message_user[row.message_id]
        evidence_user = history_user[evidence_id]

        if current_user != evidence_user:
            cross_user_evidence.append(
                (
                    row.message_id,
                    evidence_id,
                    current_user,
                    evidence_user,
                )
            )

if invalid_evidence:
    errors.append(f"Unknown evidence IDs: {invalid_evidence}")

if cross_user_evidence:
    errors.append(f"Cross-user evidence: {cross_user_evidence}")

if errors:
    print("AUDIT FAILED\n")
    for error in errors:
        print(error)
        print()
    raise SystemExit(1)

print("AUDIT PASSED")
print(f"Rows: {len(output)}")
print(f"Notify: {(output['action'] == 'notify').sum()}")
print(f"Digest: {(output['action'] == 'digest').sum()}")
print(f"Mute: {(output['action'] == 'mute').sum()}")
print(f"Fallback/unknown: {(output['message_type'] == 'unknown').sum()}")