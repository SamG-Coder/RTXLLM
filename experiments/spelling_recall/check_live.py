"""Check byte-for-byte copy, eviction and fresh-process generation."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from recall import Memory, Reader, cpu_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("spell", type=Path)
    args = parser.parse_args()
    reader = Reader(args.model)
    mem = Memory(reader)
    words = [b"quazz", b"quaz", b"quazzx", b"quaz", b"x" * 512, b"\x00", b"\xff"]
    mem.build(words)
    assert [mem.token(i) for i in range(len(words))] == words
    np.testing.assert_array_equal(mem.counts(words, 0), cpu_counts(words, 0))
    prompt = [f"qzz{i}".encode() for i in range(100)]
    known = set(reader.words[1:])
    assert not known.intersection(prompt)
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve().with_name("recall.py")),
        "generate",
        str(args.model.resolve()),
        str(args.spell.resolve()),
        "--prompt",
        b" ".join(prompt).decode(),
        "--count",
        "512",
        "--seed",
        "71",
    ]
    with tempfile.TemporaryDirectory() as tmp:

        def run():
            return json.loads(
                subprocess.check_output(command, cwd=tmp).splitlines()[-1]
            )

        output = run()
        assert len(output) == 512 and output == run()
    history = prompt[-64:]
    copied = []
    for row in output:
        word = bytes.fromhex(row["hex"])
        if row["copied"]:
            assert word in history and word not in known
            copied.append(word)
        if word not in known:
            history = (history + [word])[-64:]
    assert set(copied).intersection(prompt[-64:])
    print("Exact live copies:", copied)


if __name__ == "__main__":
    main()
