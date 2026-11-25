"""Trace the earlier prefix, keeping its last byte as a separate detector coordinate."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optical_cavity"))
from run import (
    ESCAPE,
    TriangleScene,
    cell_ids,
    land,
    mass,
    open_kernel,
    pairs,
    read,
    read_strips,
    reflector,
    save,
    write_strips,
)

DIGEST = "c802711470ddb2aaca81432fc101ed29c154e7c3d32df4172478695c33f68eae"
GRID = 128
BACKOFF = 256 * GRID**2


def splits(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Unexpected corpus edition")
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1]
    data = np.frombuffer(body, np.uint8)
    return np.split(data, [len(data) * 70 // 100, len(data) * 85 // 100])


def addresses(device, kernel, scene, history, width):
    cell = cell_ids(land(device, kernel, scene, history[:, -width - 1 : -1]), GRID)
    return np.where(cell >= 0, history[:, -1].astype(np.int64) * GRID**2 + cell, -1)


def fit(device, kernel, history, target, path, width):
    mirror = reflector(0.25)
    scene = TriangleScene(device, mirror)
    locations = addresses(device, kernel, scene, history, width)
    valid = locations >= 0
    keys, rows = np.unique(locations[valid], return_inverse=True)
    n = len(keys)
    counts = np.zeros((n + 257, 256), np.int32)
    np.add.at(counts, (rows, target[valid]), 1)
    np.add.at(counts, (n + history[:, -1].astype(int), target), 1)
    counts[-1] = np.bincount(target, minlength=256)
    strips = write_strips(counts, np.r_[keys, BACKOFF + np.arange(257)])
    save(path, mirror, strips, width, GRID, 0.25)
    return {
        "training_detector_rows": n,
        "readout_triangles": len(strips["tri_kind"]),
        "checkpoint_bytes": path.stat().st_size,
    }


def evaluate(device, kernel, reader, history, y, path):
    mirror, strips, width, grid, _ = read(path)
    if grid != GRID:
        raise ValueError("Unexpected detector dimensions")
    scene = TriangleScene(device, mirror)
    locations = addresses(device, kernel, scene, history, width)
    # Batch identical requests, including addresses that were never written.
    # This list comes solely from query landings; hit/miss comes from the AS.
    unique, revision = np.unique(locations, return_inverse=True)
    ids = np.r_[np.where(unique >= 0, unique, BACKOFF + 257), BACKOFF + np.arange(257)]
    votes = read_strips(device, reader, strips, len(ids), ids)
    back = mass(
        votes,
        len(unique) + history[:, -1].astype(int),
        y,
        votes[-1, y] / votes[-1].sum(),
    )
    high = mass(votes, revision, y, back)
    return {
        "bpb": float(-np.log2(0.99 * (0.8 * high + 0.2 * back) + ESCAPE).mean()),
        "valid": float(np.mean(locations >= 0)),
        "coverage": float(np.mean(votes.sum(1)[revision] > 0)),
        "backoff_bpb": float(-np.log2(0.99 * back + ESCAPE).mean()),
    }


def frequency(training, query):
    keys, counts = np.unique(training, return_counts=True)
    at = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    return np.where(keys[at] == query, counts[at], 0)


def ngrams(train_history, train_y, history, y):
    # Pack at most seven context bytes plus target in a uint64 exactly.
    p = np.bincount(train_y, minlength=256)[y] / len(train_y)
    code = np.zeros(len(train_y), np.uint64)
    q = np.zeros(len(y), np.uint64)
    scores = {}
    for width in range(1, 8):
        code |= train_history[:, -width].astype(np.uint64) << (8 * (width - 1))
        q |= history[:, -width].astype(np.uint64) << (8 * (width - 1))
        total = frequency(code, q)
        joint = frequency((code << 8) | train_y, (q << 8) | y)
        mle = np.divide(joint, total, out=p.copy(), where=total > 0)
        p = 0.8 * mle + 0.2 * p
        scores[f"order_{width + 1}"] = float(-np.log2(0.99 * p + ESCAPE).mean())
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "test"))
    parser.add_argument("book", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    train, val, test = splits(args.book)
    th, ty = pairs(train)
    h, y = pairs(val if args.phase == "validate" else test)
    device, kernel = open_kernel("cavity.slang")
    reader = device.create_compute_kernel(
        device.load_program("cavity_readout.slang", ["trace"])
    )
    report = {
        "training_bytes": len(train),
        "scored_bytes": len(y),
        "ngrams": ngrams(th, ty, h, y),
    }
    for width in (1, 7):
        path = args.out / f"conditioned-{width}.npz"
        if args.phase == "validate":
            report[f"geometry_{width}"] = fit(device, kernel, th, ty, path, width)
        report[f"returns_{width}"] = evaluate(device, kernel, reader, h, y, path)
    shuffled = h.copy()
    shuffled[:, :-2] = h[np.random.default_rng(20261125).permutation(len(h)), :-2]
    report["shuffled"] = evaluate(
        device, kernel, reader, shuffled, y, args.out / "conditioned-7.npz"
    )
    best = report["returns_7"]
    report["passes"] = bool(
        best["valid"] == 1
        and all(
            report[k]["bpb"] - best["bpb"] >= 0.05 for k in ("shuffled", "returns_1")
        )
    )
    print(json.dumps(report, indent=2))
    (args.out / f"{args.phase}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
