import os
from pathlib import Path

from dotenv import load_dotenv

# override=True so a key in .env beats a stale one already exported in the shell.
load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
CACHE_DIR = REPO_ROOT / "cache"

MEDIA_ANALYSIS_CACHE = CACHE_DIR / "media_analysis.json"
ROUTING_CACHE = CACHE_DIR / "routing_results.jsonl"
SQLITE_DB = CACHE_DIR / "router.db"
EMBEDDING_CACHE = CACHE_DIR / "history_embeddings.npz"

OUTPUT_CSV = REPO_ROOT / "output.csv"

# Which provider answers the per-message routing call. Groq's free tier has a
# 100k tokens/day cap that a full run plus iteration exceeds, so Gemini is the default.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.0
ROUTING_GEMINI_MODEL = "gemini-flash-lite-latest"

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OrcaRouter speaks the Anthropic Messages API, so it is reached with the Anthropic
# SDK pointed at its base URL rather than the OpenAI-compatible path.
ORCAROUTER_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.orcarouter.ai")
ORCAROUTER_MODEL = os.environ.get("ANTHROPIC_MODEL", "orcarouter/auto")

# Tried in order when the provider ahead runs out of quota. A provider whose key
# is missing is skipped, so failover simply does not engage until one is set.
PROVIDER_FAILOVER = ["orcarouter", "openrouter"]

# Agentic evidence loop: extra tool-calling rounds allowed before the model must
# decide. 0 disables the loop and uses the deterministic fused evidence as-is.
AGENT_MAX_ROUNDS = int(os.environ.get("AGENT_MAX_ROUNDS", "2"))
AGENT_ENABLED = os.environ.get("AGENT_ENABLED", "0") not in {"0", "false", "False"}

# Seconds to pause between routing calls, to stay under free-tier requests-per-minute.
CALL_DELAY_SECONDS = float(os.environ.get("CALL_DELAY_SECONDS", "1.0"))
# gemini-2.0-flash is quota-exhausted on the current key; the -latest aliases have headroom.
GEMINI_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Bumping this invalidates cached routing decisions made under an older prompt/schema.
PROMPT_VERSION = "v7-orca"

MAX_EVIDENCE = 3
MIN_VECTOR_SIMILARITY = 0.5
SQL_MATCH_SCORE = 0.9

GATED_MUTE_CONFIDENCE = 0.95
FALLBACK_CONFIDENCE = 0.1


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key


def orcarouter_auth_token() -> str | None:
    """Optional: absent means failover to OrcaRouter is simply not available."""
    return os.environ.get("ANTHROPIC_AUTH_TOKEN") or None


def openrouter_api_key() -> str | None:
    """Optional: absent means failover to OpenRouter is simply not available."""
    return os.environ.get("OPENROUTER_API_KEY") or None


def gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key
