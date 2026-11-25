"""Check the paired ring at eviction boundaries and the E10 full-book score."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/integrated"))
from check import generate
from model import EpisodicRing, Predictor, book, empty_ring
from run_index import TREASURE, oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("books", type=Path)
    args = parser.parse_args()
    pred = Predictor(args.model)
    raw = book(args.books, "treasure-120", TREASURE)
    p = pred.score(raw, paired=True)
    bpb = float(-np.log2(p).mean())
    np.testing.assert_allclose(bpb, 2.113709757601711, rtol=0, atol=1e-6)
    rng = np.random.default_rng(103)
    motif = b"\xff\x00the raven croaked.\n"
    fixture = (
        bytes(range(256)) * 3
        + motif * 49
        + rng.integers(0, 32, 2419, dtype=np.uint8).tobytes()
        + motif * 12
    )
    ref = oracle(fixture + b"\x00", 2048, True)
    ring = EpisodicRing(pred.device, empty_ring(True), 0, 0, paired=True)
    start = perf_counter()
    for t, byte in enumerate(fixture, 1):
        ring.observe(byte)
        hit = ring.query()
        assert hit[3] == byte
        np.testing.assert_array_equal(hit[:3], ref[t - 8] if t >= 8 else [-1, -1, 0])
        if t in (1, 4, 8, 2047, 2048, 2049, 4095, 4096, 4097, len(fixture)):
            data = np.frombuffer(fixture[:t], np.uint8)
            k = pred.layered.k
            n = min(t, k)
            padded = np.zeros(k, np.uint8)
            padded[-n:] = data[-n:]
            dishes, _ = pred.layered.trace(
                pred.suffix_kernel,
                pred.layered.points(padded, np.array([k])),
                budget=n + 1,
            )
            expected = pred.decode(dishes, hit[None, :], np.arange(256), full=True)
            np.testing.assert_allclose(
                pred.distribution(ring), expected / expected.sum(), rtol=1e-7
            )
    elapsed = perf_counter() - start
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        prompt = d / "prompt.bin"
        prompt.write_bytes(fixture[:2038])
        whole = generate(
            args.model,
            d / "whole.npz",
            "--layout",
            "pair",
            "--prompt",
            prompt,
            "--seed",
            3,
            "--count",
            48,
        )
        a = generate(
            args.model,
            d / "a.npz",
            "--layout",
            "pair",
            "--prompt",
            prompt,
            "--seed",
            3,
            "--count",
            10,
        )
        b = generate(args.model, d / "b.npz", "--resume", d / "a.npz", "--count", 38)
        assert whole == a + b
        with np.load(d / "a.npz") as state:
            assert int(state["cursor"]) == 0 and state["layout"] == "pair"
        print(
            json.dumps(
                {
                    "book_targets": len(p),
                    "bpb": bpb,
                    "verified_live_positions": len(fixture),
                    "ring_as_bytes": ring.scene.metrics["as_bytes"],
                    "resumed_identical_bytes": len(b),
                    "observe_and_query_per_second": len(fixture) / elapsed,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
