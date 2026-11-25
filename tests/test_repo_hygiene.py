"""Repository hygiene: every text file must be valid UTF-8 (cp1252 bytes have slipped in twice)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".md", ".py", ".slang", ".slangh", ".json", ".toml", ".txt", ".yml", ".yaml"}


def tracked_text_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / f for f in out.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in ROOT.rglob("*") if ".git" not in p.parts]
    return [f for f in files if f.suffix.lower() in TEXT and f.is_file()]


def test_text_files_are_utf8():
    bad = []
    for f in tracked_text_files():
        try:
            f.read_bytes().decode("utf-8")
        except UnicodeDecodeError as e:
            bad.append(f"{f.relative_to(ROOT)} @ byte {e.start}")
    assert not bad, "not valid UTF-8: " + ", ".join(bad)
