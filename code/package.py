"""Builds the submission zip.

    python code/package.py

Layout inside the zip, per the upload form's "source code + README" shape:

    README.md                     the solution README, at the root where it is looked for
    requirements.txt
    .env.example
    code/...                      the router
    derived/media_analysis.json   committed OCR/ASR result, so it runs without a media key

output/ is deliberately excluded: predictions are uploaded separately, and shipping
them would also let a grader mistake stale output for a fresh run.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "code.zip"

# Anything matching these never enters the zip, whatever else the rules say.
EXCLUDE = re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.venv|venv)(/|$)|\.pyc$|\.env$")

SECRETS = re.compile(r"gsk_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}")

# (source on disk, path inside the zip)
CONTENTS: list[tuple[Path, str]] = [
    (ROOT / "code" / "README.md", "README.md"),
    (ROOT / "requirements.txt", "requirements.txt"),
    (ROOT / ".env.example", ".env.example"),
    (ROOT / "derived" / "media_analysis.json", "derived/media_analysis.json"),
]


def _collect() -> list[tuple[Path, str]]:
    items = list(CONTENTS)
    for path in sorted((ROOT / "code").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if not EXCLUDE.search(rel):
            items.append((path, rel))
    return items


def main() -> None:
    items = _collect()

    missing = [str(src.relative_to(ROOT)) for src, _ in items if not src.exists()]
    if missing:
        raise SystemExit(f"cannot package, missing: {', '.join(missing)}")

    leaked = [
        arc
        for src, arc in items
        if src.suffix in {".py", ".md", ".txt", ".json", ".example"}
        and SECRETS.search(src.read_text(encoding="utf-8", errors="ignore"))
    ]
    if leaked:
        raise SystemExit(f"refusing to package, possible secrets in: {', '.join(leaked)}")

    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in items:
            zf.write(src, arc)

    size_kb = round(ZIP_PATH.stat().st_size / 1024)
    print(f"wrote {ZIP_PATH.name} ({len(items)} files, {size_kb}K)")
    for _, arc in sorted(items, key=lambda pair: pair[1]):
        print(f"  {arc}")


if __name__ == "__main__":
    main()
