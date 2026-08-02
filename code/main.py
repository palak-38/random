from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_CSV, ROUTING_CACHE, SQLITE_DB  # noqa: E402
from context_builder import ContextBuilder  # noqa: E402
from data_store import DataStore, build_db, connect  # noqa: E402
from finalize import finalize  # noqa: E402
from media_analysis import analyse_all, load_cached  # noqa: E402
from pipeline import run  # noqa: E402


def cmd_media(args: argparse.Namespace) -> None:
    store = DataStore(build_db())
    items = store.all_media()
    print(f"analysing {len(items)} media files (cached results are skipped)")
    results = analyse_all(items, force=args.force)
    failed = [k for k, v in results.items() if v.analysis_error]
    print(f"done: {len(results)} cached, {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))


def cmd_route(args: argparse.Namespace) -> None:
    store = DataStore(build_db() if args.rebuild_db else connect())
    media = load_cached()
    if not media:
        print("warning: no media analysis cache found; media messages will be routed on text alone.")
        print("         run `python code/main.py media` first for best results.")
    builder = ContextBuilder(store, media)

    from agent_tools import EvidenceTools
    from router import LLMRouter

    def tools_factory(msg):
        return EvidenceTools(builder.retriever, store, msg)

    decisions = run(store, builder, LLMRouter(), tools_factory, table=args.table, limit=args.limit)
    print(f"\n{len(decisions)} decisions cached in {ROUTING_CACHE}")

    if args.table == "messages":
        total, missing = finalize(store)
        print(f"wrote {OUTPUT_CSV} with {total} rows ({missing} backfilled)")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Score the labelled samples, routing them first if nothing is cached."""
    from evaluation.main import main as score

    from pipeline import load_cache

    if not any(mid.startswith("sample_") for mid in load_cache()):
        print("no cached sample predictions; routing them first\n")
        cmd_route(argparse.Namespace(table="sample_messages", limit=None, rebuild_db=True))
        print()
    score()


def cmd_finalize(args: argparse.Namespace) -> None:
    store = DataStore(connect())
    total, missing = finalize(store)
    print(f"wrote {OUTPUT_CSV} with {total} rows ({missing} backfilled with safe defaults)")


def cmd_build(args: argparse.Namespace) -> None:
    build_db()
    print(f"built {SQLITE_DB}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="main.py", description="WhatsApp message notification router")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-db", help="load dataset CSVs into SQLite")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("media", help="one-time Gemini pass over images and voice notes")
    p.add_argument("--force", action="store_true", help="re-analyse everything, ignoring the cache")
    p.set_defaults(func=cmd_media)

    p = sub.add_parser("route", help="route messages and write output.csv")
    p.add_argument("--table", default="messages", choices=["messages", "sample_messages"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rebuild-db", action="store_true")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("evaluate", help="score the labelled samples (routes them first if needed)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("finalize", help="write output.csv from cached decisions")
    p.set_defaults(func=cmd_finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
