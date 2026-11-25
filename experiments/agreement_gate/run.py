"""E13: route recalled successors by their support in the long-term scene."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "experiments" / d) for d in ("pair_address", "copy_confidence")
]
from run_index import Predictor, book, oracle, place, search
from experiment import fit, read
from run import BOOKS, geometry
from rtxllm.baselines.ngram import KneserNey

MOBY = "baf9b04fcf6e0008f30eaaa6697277751448c1070b20175e0082f28cbb161fd3"


def evidence(model, kernel, raw):
    data = np.frombuffer(raw, np.uint8)
    idx = np.arange(8, len(raw))
    hits = search(model, kernel, place(model, raw, True), len(idx), 2048, True)
    dishes, _ = model.layered.trace(
        model.suffix_kernel, model.layered.points(data, idx)
    )
    empty = np.zeros_like(hits)
    candidate = np.maximum(hits[:, 1], 0).astype(np.uint8)
    support = model.decode(dishes, empty, candidate)
    bins = np.searchsorted([1 / 16, 1 / 2], support, side="right")
    return hits, support, hits[:, 2] + 9 * bins, model.decode(dishes, empty, data[8:])


def check_priors(kn, raw, hits, base, support):
    data = np.frombuffer(raw, np.uint8)
    np.testing.assert_allclose(base, kn.prob(data, start=8), atol=2e-7, rtol=2e-5)
    # Each six-byte row has its own complete five-byte history at its final byte.
    contexts = np.lib.stride_tricks.sliding_window_view(data, 6)[3:]
    for start in range(0, len(hits), 16384):
        sl = slice(start, start + 16384)
        rows = contexts[sl].copy()
        rows[:, -1] = np.maximum(hits[sl, 1], 0)
        expected = kn.prob(rows.ravel(), start=5)[::6]
        np.testing.assert_allclose(support[sl], expected, atol=2e-7, rtol=2e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    for name in ("model", "books", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    train = [np.frombuffer(book(args.books, n), np.uint8) for n in list(BOOKS)[:4]]
    kn = KneserNey(6).fit(train)
    raw = (
        book(args.books, "jekyll-43")
        if args.phase == "fit"
        else book(args.books, "moby-2701", MOBY)
    )
    model = Predictor(args.model)
    if model.layered.k != 5:
        raise ValueError("Requires the four-book order-six archive")
    kernel = model.kernel("episodic.slang", "indexedTrace")
    hits, support, rows, base = evidence(model, kernel, raw)
    np.testing.assert_array_equal(hits[:, :3], oracle(raw, 2048, True))
    check_priors(kn, raw, hits, base, support)
    for cut in (8, 257, 4096):
        altered = raw[:cut] + bytes(b ^ 255 for b in raw[cut : cut + 512])
        alt_hits, alt_support, alt_rows, _ = evidence(model, kernel, altered)
        for old, new in ((hits, alt_hits), (support, alt_support), (rows, alt_rows)):
            np.testing.assert_array_equal(old[: cut - 7], new[: cut - 7])
    correct = np.frombuffer(raw, np.uint8)[8:] == hits[:, 1]
    report = {
        "model_sha256": model.identity,
        "targets": len(rows),
        "prior_bpb": float(-np.log2(base).mean()),
    }
    gate_kernel = model.kernel("copy_confidence.slang", "trace")
    for name, address, count in (("length", hits[:, 2], 9), ("agreement", rows, 27)):
        archive = args.out / f"{name}.npz"
        if args.phase == "fit":
            heights = [
                fit(base, correct, address == i) if i % 9 >= 2 else 0.0
                for i in range(count)
            ]
            centers = np.column_stack((np.arange(count), np.zeros(count), heights))
            np.savez_compressed(
                archive,
                **geometry(centers, [(-0.4, -0.4, 0), (0.4, -0.4, 0), (0, 0.4, 0)]),
            )
        with np.load(archive, allow_pickle=False) as saved:
            geo = dict(saved)
        w = read(model.device, gate_kernel, geo, address)
        h = geo["tri_vertices"][:, 0, 2]
        np.testing.assert_allclose(w, h[address], atol=1e-7)
        p = (1 - w) * base + w * correct
        report[name] = {
            "bpb": float(-np.log2(p).mean()),
            "support": np.bincount(address, minlength=count).tolist(),
            "heights": h.tolist(),
        }
    report["gain_bpb"] = report["length"]["bpb"] - report["agreement"]["bpb"]
    report["meets_threshold"] = report["gain_bpb"] >= 0.005
    (args.out / f"{args.phase}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(report["targets"], report["gain_bpb"], report["meets_threshold"])


if __name__ == "__main__":
    main()
