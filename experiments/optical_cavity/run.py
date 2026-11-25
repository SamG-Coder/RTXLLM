"""Select optical cavity geometry on validation data and score the saved scene."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from optics import (
    TriangleScene,
    cell_ids,
    check_optics,
    land,
    open_kernel,
    read_strips,
    reflector,
    write_strips,
)

DIGEST = "c29a4f56dd91aa894a3f17d5f96b356ccb499e3e2c5a27964de911845e39efab"
ESCAPE = 0.01 / 256


def corpus(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Corpus SHA256 mismatch")
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1]
    data = np.frombuffer(body, np.uint8)
    return np.split(data, [len(data) * 70 // 100, len(data) * 85 // 100])


def pairs(data):
    w = np.lib.stride_tricks.sliding_window_view(data, 9)
    return w[:, :8], w[:, 8]


def observations(cell, history, y, grid):
    # Reserve 256 backoff detectors after the grid, followed by the unigram.
    base = grid**2
    counts = np.zeros((base + 257, 256), np.int32)
    valid = cell >= 0
    np.add.at(counts, (cell[valid], y[valid]), 1)
    np.add.at(counts, (base + history[:, -1].astype(np.int32), y), 1)
    counts[-1] = np.bincount(y, minlength=256)
    return counts


def mass(rows, keys, targets, default):
    totals = rows.sum(1)
    denom = totals[keys]
    # Invalid or unwritten rows back off to the separately traced lower order model.
    return np.divide(
        rows[keys, targets], denom, out=default.copy(), where=(keys >= 0) & (denom > 0)
    )


def measure(rows, cell, history, y, grid):
    unigram = rows[-1, y] / rows[-1].sum()
    backoff = mass(rows, grid**2 + history[:, -1].astype(np.int32), y, unigram)
    higher = mass(rows, cell, y, backoff)
    prob = 0.99 * (0.8 * higher + 0.2 * backoff) + ESCAPE
    return {
        "bpb": float(-np.log2(prob).mean()),
        "valid": float(np.mean(cell >= 0)),
        "occupied": float(np.mean((cell >= 0) & (rows.sum(1)[cell] > 0))),
        "backoff_bpb": float(-np.log2(0.99 * backoff + ESCAPE).mean()),
    }


def save(path, mirror, strips, width, grid, retention):
    arrays = {
        prefix + k: v
        for prefix, d in (("mirror_", mirror), ("strip_", strips))
        for k, v in d.items()
    }
    np.savez_compressed(path, **arrays, width=width, grid=grid, retention=retention)


def read(path):
    with np.load(path, allow_pickle=False) as f:

        def mesh(prefix):
            return {k[len(prefix) :]: f[k] for k in f.files if k.startswith(prefix)}

        return (
            mesh("mirror_"),
            mesh("strip_"),
            int(f["width"]),
            int(f["grid"]),
            float(f["retention"]),
        )


def validate(device, kernel, reader, train, val, out):
    history, labels = pairs(train)
    vh, vy = pairs(val)
    best = {1: np.inf, 8: np.inf}
    results = []
    for retention in (0.25, 0.5, 0.75):
        mirror = reflector(retention)
        scene = TriangleScene(device, mirror)
        for width in (1, 8):
            tr = land(device, kernel, scene, history[:, -width:])
            vr = land(device, kernel, scene, vh[:, -width:])
            for grid in (32, 64, 128):
                counts = observations(cell_ids(tr, grid), history, labels, grid)
                strips = write_strips(counts)
                rows = read_strips(device, reader, strips, len(counts))
                score = measure(rows, cell_ids(vr, grid), vh, vy, grid)
                result = {"width": width, "grid": grid, "retention": retention, **score}
                results.append(result)
                if score["bpb"] < best[width]:
                    save(
                        out / f"width-{width}.npz",
                        mirror,
                        strips,
                        width,
                        grid,
                        retention,
                    )
                    best[width] = score["bpb"]
    print("Best validation losses:", best)
    return results


def statistical_controls(training, history, y):
    th, ty = pairs(training)

    def key(h):
        return h[:, -2].astype(np.int32) * 256 + h[:, -1]

    # Sparse pair indexing avoids a dense 256^3 validation table.
    ids, idx = np.unique(key(th), return_inverse=True)
    tri = np.zeros((len(ids) + 1, 256), np.int32)
    np.add.at(tri, (idx, ty), 1)
    query = np.searchsorted(ids, key(history))
    valid = ids[np.minimum(query, len(ids) - 1)] == key(history)
    query = np.where(valid, query, len(ids))
    bi = np.zeros((256, 256), np.int32)
    np.add.at(bi, (th[:, -1], ty), 1)
    counts = np.bincount(ty, minlength=256)
    uni = counts[y] / counts.sum()
    big = 0.8 * mass(bi, history[:, -1], y, uni) + 0.2 * uni
    trig = 0.8 * mass(tri, query, y, big) + 0.2 * big
    return {
        k: float(-np.log2(0.99 * p + ESCAPE).mean())
        for k, p in (("exact_bigram", big), ("exact_trigram", trig))
    }


def test_models(device, kernel, reader, train, test, out):
    histories, y = pairs(test)
    report = {
        "scored_bytes": len(y),
        "training_bytes": len(train),
        **statistical_controls(train, histories, y),
    }
    for saved_width in (1, 8):
        path = out / f"width-{saved_width}.npz"
        mirror, strips, width, grid, retention = read(path)
        if width != saved_width:
            raise ValueError("Inconsistent checkpoint width")
        scene = TriangleScene(device, mirror)
        rows = read_strips(device, reader, strips, grid**2 + 257)
        for shuffled in (False, True) if width == 8 else (False,):
            h = histories[:, -width:].copy()
            if shuffled:
                h[:, :-1] = h[np.random.default_rng(20261125).permutation(len(h)), :-1]
            positions = land(device, kernel, scene, h)
            score = measure(rows, cell_ids(positions, grid), h, y, grid)
            report["shuffled" if shuffled else f"width_{width}"] = {
                **score,
                "grid": grid,
                "retention": retention,
                "checkpoint_bytes": path.stat().st_size,
                "mirror_triangles": len(mirror["tri_kind"]),
                "readout_triangles": len(strips["tri_kind"]),
            }
    full = report["width_8"]
    report["passes"] = bool(
        full["valid"] >= 0.999
        and all(report[k]["bpb"] - full["bpb"] >= 0.1 for k in ("shuffled", "width_1"))
    )
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "test"))
    parser.add_argument("book", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train, val, test = corpus(args.book)
    device, kernel = open_kernel("cavity.slang")
    reader = device.create_compute_kernel(
        device.load_program("cavity_readout.slang", ["trace"])
    )
    strip_probe = np.zeros((3, 256), np.int32)
    strip_probe[0, 65] = 256
    strip_probe[1, [0, 255]] = [192, 64]
    np.testing.assert_array_equal(
        read_strips(device, reader, write_strips(strip_probe), 3), strip_probe
    )
    check_optics(device, kernel)
    result = (
        validate(device, kernel, reader, train, val, args.out)
        if args.phase == "validate"
        else test_models(device, kernel, reader, train, test, args.out)
    )
    (args.out / f"{args.phase}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
