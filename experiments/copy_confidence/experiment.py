"""Calibrate physical copy gates on Jekyll; test unchanged gates on War of the Worlds."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "episodic"))
from run import (
    BOOKS,
    TriangleScene,
    book,
    dispatch,
    geometry,
    gpu_buffer,
    open_kernel,
    reference,
    trace,
)

from rtxllm.baselines.ngram import KneserNey

GRID = (0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)
WAR_SHA = "0c9bea907452abd6e9a31d9152188ef5007f0e2d3e4a52d7a6a627a37616249f"


def loss(base, correct, w):
    return float(-np.log2((1 - w) * base + w * correct).mean())


def fit(base, correct, mask):
    if not mask.any():
        return 0.0
    return min(GRID, key=lambda w: loss(base[mask], correct[mask], w))


def gate(weights):
    centers = np.column_stack((np.arange(9), np.zeros(9), weights))
    return geometry(centers, [(-0.4, -0.4, 0), (0.4, -0.4, 0), (0, 0.4, 0)])


def read(device, kernel, geo, lengths):
    weights = dispatch(
        device,
        kernel,
        TriangleScene(device, geo),
        np.empty(len(lengths), np.float32),
        "weights",
        queries=len(lengths),
        lengths=gpu_buffer(device, lengths.astype(np.uint32)),
    )
    if not np.all((weights >= 0) & (weights <= 1)):
        raise RuntimeError("Invalid gate intersection")
    return weights.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "test"])
    parser.add_argument("books", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train_names = list(BOOKS)[:4]
    kn = KneserNey(5).fit(
        [np.frombuffer(book(args.books, n), np.uint8) for n in train_names]
    )
    if args.phase == "calibrate":
        name = "jekyll-43"
        raw = book(args.books, name)
    else:
        name = "war-36"
        raw = book(args.books, name, WAR_SHA)
    data = np.frombuffer(raw, np.uint8)
    device, search = open_kernel("episodic.slang")
    kernel = device.create_compute_kernel(
        device.load_program("copy_confidence.slang", ["trace"])
    )
    hits, _ = trace(device, search, raw)
    np.testing.assert_array_equal(hits[:, :3], reference(raw))
    length = hits[:, 2]
    correct = hits[:, 1] == data[8:]
    base = kn.prob(data, start=8)
    checkpoint = args.out / "gate.npz"
    if args.phase == "calibrate":
        heights = np.array(
            [0.0, 0.0] + [fit(base, correct, length == n) for n in range(2, 9)]
        )
        global_weight = fit(base, correct, length >= 2)
        np.savez(checkpoint, **gate(heights), global_weight=global_weight)
    with np.load(checkpoint, allow_pickle=False) as saved:
        global_weight = float(saved["global_weight"])
        geo = {k: saved[k] for k in saved.files if k != "global_weight"}
    weights = read(device, kernel, geo, length)
    heights = read(device, kernel, geo, np.arange(9))
    np.testing.assert_array_equal(weights, heights[length])
    # Move only the learned height at each active length into reverse order.
    reverse = {**geo, "tri_vertices": geo["tri_vertices"].copy()}
    reverse["tri_vertices"][2:, :, 2] = geo["tri_vertices"][:1:-1, :, 2]
    wrong = read(device, kernel, reverse, length)
    np.testing.assert_array_equal(wrong, np.r_[heights[:2], heights[:1:-1]][length])
    scores = {
        "kn5": loss(base, correct, 0.0),
        "fixed": loss(base, correct, 0.25 * (length >= 2)),
        "global": loss(base, correct, global_weight * (length >= 2)),
        "learned": loss(base, correct, weights),
        "reversed": loss(base, correct, wrong),
    }
    report = {
        "book": name,
        "scored": len(base),
        "device": str(device.info.adapter_name),
        "global_weight": global_weight,
        "ray_read_weights": heights.tolist(),
        "bits_per_byte": scores,
        "changed_probabilities_under_reversal": int(
            np.count_nonzero((wrong - weights) * (correct - base))
        ),
        "positions_by_length": np.bincount(length, minlength=9).tolist(),
        "passes": scores["kn5"] - scores["learned"] >= 0.01
        and scores["learned"] < scores["global"],
    }
    print(json.dumps(report, indent=2))
    (args.out / (args.phase + ".json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
