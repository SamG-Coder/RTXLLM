"""Test pooling over all equally long ray-retrieved candidates."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import slangpy as sp

ROOT = Path(__file__).resolve().parents[2]
for path in ("integrated", "copy_confidence", "pair_address"):
    sys.path.insert(0, str(ROOT / "experiments" / path))
from experiment import fit, gate, loss, read
from model import Predictor, book, gpu_buffer
from run_index import place

HOLMES = "2a0ce5568625a304f54fbb6bfdeeadf527a1437254b5f0e06b5a9e233d4784a2"


def oracle(raw, t):
    low = max(0, t - 2048)
    key = raw[t - 2 : t]
    longest = 0
    successors = []
    cursor = t - 1
    for _ in range(16):
        start = raw.rfind(key, low, cursor)
        if start < 0:
            break
        end = start + 1
        length = 2
        while (
            length < 8
            and end - length >= low
            and raw[end - length] == raw[t - 1 - length]
        ):
            length += 1
        if length > longest:
            longest, successors = length, []
        if length == longest:
            successors.append(raw[end + 1])
        cursor = end
    return [longest, len(successors), *successors, *([-1] * (16 - len(successors)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "test"])
    parser.add_argument("model", type=Path)
    parser.add_argument("books", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    test = args.phase == "test"
    name = "holmes-1661" if test else "jekyll-43"
    raw = book(args.books, name, HOLMES if test else None)
    model = Predictor(args.model)
    if model.layered.k != 5:
        raise ValueError("Requires order-six scene")
    dev = model.device
    data = np.frombuffer(raw, np.uint8)
    idx = np.arange(8, len(data))
    y = data[8:]
    dishes, _ = model.layered.trace(
        model.suffix_kernel, model.layered.points(data, idx)
    )
    base = model.decode(dishes, np.zeros((len(y), 4), np.int32), y)
    scene = place(model, raw, True)
    buf = gpu_buffer(
        dev, np.empty((len(y), 18), np.int32), sp.BufferUsage.unordered_access
    )
    model.kernel("pooled_recall.slang", "collect").dispatch(
        thread_count=[len(y), 1, 1],
        vars={"targets": len(y), "scene": scene.tlas.structure, "paths": buf},
    )
    copied = gpu_buffer(
        dev, np.empty(len(y), np.uint32), sp.BufferUsage.unordered_access
    )
    model.kernel("pooled_recall.slang", "collectMass").dispatch(
        thread_count=[len(y), 1, 1],
        vars={
            "targets": len(y),
            "paths": buf,
            "observed": gpu_buffer(dev, y.astype(np.uint32)),
            "copied": copied,
        },
    )
    h = buf.to_numpy().view(np.int32).reshape(-1, 18)
    mass = copied.to_numpy().view(np.uint32) / np.maximum(h[:, 1], 1)
    for i in range(0, len(y), 4096):
        ref = np.asarray(
            [oracle(raw, t) for t in range(i + 8, min(i + 4096, len(y)) + 8)]
        )
        np.testing.assert_array_equal(h[i : i + len(ref)], ref)
        expected = (ref[:, 2:] == y[i : i + len(ref), None]).sum(1) / np.maximum(
            ref[:, 1], 1
        )
        np.testing.assert_array_equal(mass[i : i + len(ref)], expected)
    length = h[:, 0]
    nearest = h[:, 2] == y
    weights = {}
    kernel = model.kernel("copy_confidence.slang", "trace")
    for kind, signal in (("nearest", nearest), ("pool", mass)):
        ck = args.out / f"{kind}.npz"
        if not test:
            heights = np.array(
                [0.0, 0.0] + [fit(base, signal, length == n) for n in range(2, 9)]
            )
            np.savez(ck, **gate(heights))
        with np.load(ck, allow_pickle=False) as saved:
            weights[kind] = read(dev, kernel, dict(saved), np.arange(9))
    near = weights["nearest"][length]
    pool = weights["pool"][length]
    scores = {
        "prior": loss(base, nearest, 0.0),
        "nearest": loss(base, nearest, near),
        "pool_at_nearest_weights": loss(base, mass, near),
        "pool": loss(base, mass, pool),
    }
    tie = h[:, 1] > 1
    disagreement = tie & ((h[:, 2:] >= 0) & (h[:, 2:] != h[:, 2:3])).any(1)
    result = {
        "book": name,
        "targets": len(y),
        "tie_positions": int(tie.sum()),
        "different_successors": int(disagreement.sum()),
        "weights": {k: v.tolist() for k, v in weights.items()},
        "bits_per_byte": scores,
        "model_sha256": model.identity,
        "passes": scores["nearest"] - scores["pool"] >= 0.005
        and scores["pool"] < scores["prior"],
    }
    print(json.dumps(result, indent=2))
    (args.out / (args.phase + ".json")).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
