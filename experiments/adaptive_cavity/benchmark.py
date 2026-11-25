"""Cross-author validation and test of learned geometric readout under a common budget."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conditioned_cavity"))
from check_budget import BUDGET, budgeted_strips
from experiment import (
    ESCAPE,
    GRID,
    TriangleScene,
    cell_ids,
    evaluate,
    land,
    mass,
    ngrams,
    open_kernel,
    pairs,
    read_strips,
    reflector,
    save,
)
from partition import detect, lattice, make_partition

BOOKS = {
    "alice-11": "49a0b2726606e1290ac03a63978fa1dd1bd38a8d805704d98265f393533ea094",
    "pride-1342": "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163",
    "frankenstein-84": "c29a4f56dd91aa894a3f17d5f96b356ccb499e3e2c5a27964de911845e39efab",
    "dracula-345": "c802711470ddb2aaca81432fc101ed29c154e7c3d32df4172478695c33f68eae",
    "jekyll-43": "31df0a9af1bce24ccbb5ab3ccb659603353accde5b7e40f7c5b887fb88173d48",
    "time-35": "0eb70818a4c7e2e04f91be807bad37ca5c1475cd51cec2a7b620c4db2b595ce4",
}


def windows(folder, name):
    raw = (folder / f"rtxllm-{name}.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != BOOKS[name]:
        raise ValueError(f"Digest mismatch for {name}")
    body = re.search(
        rb"(?ms)^\*\*\* START OF.*?\*\*\*[^\S\n]*\n(.*?)^\*\*\* END OF", raw
    )[1]
    return pairs(np.frombuffer(body, np.uint8))


def fit(device, kernel, h, y, width, out):
    mirror = reflector(0.25)
    cavity = TriangleScene(device, mirror)
    pos = land(device, kernel, cavity, h[:, -width - 1 : -1])
    byte = h[:, -1]
    regions, strips = make_partition(lattice(pos), byte, y, BUDGET)
    adaptive = out / f"adaptive-{width}.npz"
    np.savez_compressed(
        adaptive,
        width=width,
        **{
            label + k: v
            for label, geo in (
                ("mirror_", mirror),
                ("partition_", regions),
                ("strip_", strips),
            )
            for k, v in geo.items()
        },
    )
    at = byte.astype(np.int64) * GRID**2 + cell_ids(pos, GRID)
    pruned, kept = budgeted_strips(at, h, y)
    fixed = out / f"fixed-{width}.npz"
    save(fixed, mirror, pruned, width, GRID, 0.25)
    return {
        "leaves": len(regions["tri_kind"]) // 2,
        "adaptive_triangles": len(regions["tri_kind"]) + len(strips["tri_kind"]),
        "adaptive_bytes": adaptive.stat().st_size,
        "fixed_rows": kept,
        "fixed_triangles": len(pruned["tri_kind"]),
        "fixed_bytes": fixed.stat().st_size,
    }


def score(device, kernel, partition_kernel, reader, h, y, path):
    with np.load(path, allow_pickle=False) as data:

        def geo(prefix):
            return {
                k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)
            }

        width = int(data["width"])
        cavity = TriangleScene(device, geo("mirror_"))
        regions, strips = geo("partition_"), geo("strip_")
    positions = land(device, kernel, cavity, h[:, -width - 1 : -1])
    rows = detect(device, partition_kernel, regions, lattice(positions), h[:, -1])
    counts = read_strips(device, reader, strips, len(regions["station_pos"]))
    back = mass(counts, h[:, -1], y, counts[256, y] / counts[256].sum())
    high = mass(counts, rows, y, back)
    return {
        "bpb": float(-np.log2(0.99 * (0.8 * high + 0.2 * back) + ESCAPE).mean()),
        "cavity_and_region_completion": float(np.mean(rows >= 0)),
        "nonempty_row": float(np.mean(counts.sum(1)[rows] > 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "test"))
    parser.add_argument(
        "books", type=Path, help="Directory containing the pinned rtxllm-*.txt files"
    )
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    train_names = tuple(BOOKS)[:4]
    examples = [windows(args.books, name) for name in train_names]
    th, ty = (np.concatenate(columns) for columns in zip(*examples))
    held = "jekyll-43" if args.phase == "validate" else "time-35"
    h, y = windows(args.books, held)
    device, kernel = open_kernel("cavity.slang")
    region = device.create_compute_kernel(
        device.load_program("cavity_region.slang", ["trace"])
    )
    reader = device.create_compute_kernel(
        device.load_program("cavity_readout.slang", ["trace"])
    )
    report = {
        "train_targets": len(ty),
        "book": held,
        "scored_bytes": len(y),
        "counts": ngrams(th, ty, h, y),
    }
    for w in (1, 7):
        if args.phase == "validate":
            report[f"geometry_{w}"] = fit(device, kernel, th, ty, w, args.out)
        report[f"adaptive_{w}"] = score(
            device, kernel, region, reader, h, y, args.out / f"adaptive-{w}.npz"
        )
        report[f"fixed_{w}"] = evaluate(
            device, kernel, reader, h, y, args.out / f"fixed-{w}.npz"
        )
    shuffled = h.copy()
    shuffled[:, :-2] = h[np.random.default_rng(20261125).permutation(len(h)), :-2]
    report["shuffled"] = score(
        device, kernel, region, reader, shuffled, y, args.out / "adaptive-7.npz"
    )
    full = report["adaptive_7"]
    report["passes"] = bool(
        full["cavity_and_region_completion"] >= 0.999
        and all(
            report[k]["bpb"] - full["bpb"] >= 0.05 for k in ("adaptive_1", "fixed_7")
        )
    )
    print(json.dumps(report, indent=2))
    (args.out / f"{args.phase}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
