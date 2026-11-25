"""Verify E9 against the loaded scene, including the live distribution and CLI resume."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from model import BOOKS, ROOT, EpisodicRing, KneserNey, Predictor, book, empty_ring

sys.path.insert(0, str(ROOT / "experiments" / "copy_confidence"))
from experiment import WAR_SHA, read
from run import reference


def generate(path, output, *args):
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).with_name("model.py")),
        "generate",
        str(path),
        str(output),
        *map(str, args),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return bytes.fromhex(json.loads(result.stdout.splitlines()[-1])["sample_hex"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("gate", type=Path)
    parser.add_argument("books", type=Path)
    args = parser.parse_args()
    predictor = Predictor(args.model)
    docs = [np.frombuffer(book(args.books, n), np.uint8) for n in list(BOOKS)[:4]]
    kn = KneserNey(5).fit(docs)
    raw = book(args.books, "war-36", WAR_SHA)
    data = np.frombuffer(raw, np.uint8)
    matches = reference(raw)
    with np.load(args.gate, allow_pickle=False) as saved:
        gate = {k: saved[k] for k in saved.files if k != "global_weight"}
    weights = read(
        predictor.device,
        predictor.kernel("copy_confidence.slang", "trace"),
        gate,
        np.arange(9),
    )
    w = weights[matches[:, 2]]
    expected = (1 - w) * kn.prob(data, start=8) + w * (matches[:, 1] == data[8:])
    actual = predictor.score(raw)
    np.testing.assert_allclose(actual, expected, rtol=8e-7, atol=1e-10)
    np.testing.assert_allclose(
        -np.log2(actual).mean(), 2.2010866081353164, atol=1e-5, rtol=0
    )
    ring = EpisodicRing(predictor.device, empty_ring(), 0, 0)
    fixture = bytes(range(256)) + b" language is language " * 7
    ref = reference(fixture + b"x")
    for t, byte in enumerate(fixture, 1):
        ring.observe(byte)
        if t not in (1, 2, 3, 4, 8, 255, 256, 257, 258, len(fixture)):
            continue
        p = predictor.distribution(ring)
        history = fixture[max(0, t - 4) : t]
        base = np.array(
            [
                kn.prob(
                    np.frombuffer(history + bytes([b]), np.uint8), start=len(history)
                )[0]
                for b in range(256)
            ]
        )
        if t >= 8:
            _, pred, n = ref[t - 8]
            base *= 1 - weights[n]
            if pred >= 0:
                base[pred] += weights[n]
        np.testing.assert_allclose(p, base, rtol=8e-7, atol=1e-10)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prompt = folder / "prompt.bin"
        prompt.write_bytes(b"The Martians ")
        whole = generate(
            args.model,
            folder / "whole.npz",
            "--prompt",
            prompt,
            "--seed",
            53,
            "--count",
            128,
        )
        first = generate(
            args.model,
            folder / "first.npz",
            "--prompt",
            prompt,
            "--seed",
            53,
            "--count",
            53,
        )
        rest = generate(
            args.model,
            folder / "last.npz",
            "--resume",
            folder / "first.npz",
            "--count",
            75,
        )
        assert whole == first + rest
        print(
            json.dumps(
                {
                    "scored": len(actual),
                    "bpb": float(-np.log2(actual).mean()),
                    "max_abs_probability_error": float(
                        np.max(np.abs(actual - expected))
                    ),
                    "archive_bytes": os.path.getsize(args.model),
                    "long_term_as_bytes": predictor.layered.metrics["as_bytes"],
                    "session_bytes": (folder / "last.npz").stat().st_size,
                    "resume_bytes": len(whole),
                    "generated_sample": repr(whole),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
