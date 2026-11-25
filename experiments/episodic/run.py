"""Measure a ray-addressed 256-byte episodic memory on held-out books."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import slangpy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optical_cavity"))
from optics import TriangleScene, mesh, open_kernel
from run_specular import gpu_buffer

BOOKS = {
    "alice-11": "49a0b2726606e1290ac03a63978fa1dd1bd38a8d805704d98265f393533ea094",
    "pride-1342": "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163",
    "frankenstein-84": "c29a4f56dd91aa894a3f17d5f96b356ccb499e3e2c5a27964de911845e39efab",
    "dracula-345": "c802711470ddb2aaca81432fc101ed29c154e7c3d32df4172478695c33f68eae",
    "jekyll-43": "31df0a9af1bce24ccbb5ab3ccb659603353accde5b7e40f7c5b887fb88173d48",
    "time-35": "0eb70818a4c7e2e04f91be807bad37ca5c1475cd51cec2a7b620c4db2b595ce4",
}


def book(folder, name, digest=None):
    raw = (folder / f"rtxllm-{name}.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (digest or BOOKS[name]):
        raise ValueError(f"Digest mismatch for {name}")
    body = re.search(
        rb"(?ms)^\*\*\* START OF.*?\*\*\*[^\S\n]*\n(.*?)^\*\*\* END OF", raw
    )[1]
    return body


def geometry(centers, offsets):
    vertices = (centers[:, None] + np.asarray(offsets, np.float32)).astype(np.float32)
    zeros = np.zeros(len(vertices), np.int32)
    return mesh(vertices, zeros, zeros, np.array([[1, 0, 0]]))


def dispatch(device, kernel, scene, output, field, **params):
    buf = gpu_buffer(device, output, sp.BufferUsage.unordered_access)
    kernel.dispatch(
        thread_count=[len(output), 1, 1],
        vars={"scene": scene.tlas.structure, field: buf, **params},
    )
    return buf.to_numpy().view(output.dtype).reshape(output.shape).copy()


def fit_prior(documents):
    counts = np.zeros((256, 256), np.int64)
    uni = np.zeros(256, np.int64)
    for raw in documents:
        x = np.frombuffer(raw, np.uint8).astype(np.int32)
        counts += np.bincount(256 * x[:-1] + x[1:], minlength=65536).reshape(256, 256)
        uni += np.bincount(x, minlength=256)
    if not uni.sum():
        raise ValueError("Prior training requires at least one observed byte")
    unigram = uni / uni.sum()
    base = np.divide(
        counts,
        counts.sum(1)[:, None],
        out=np.tile(unigram, (256, 1)),
        where=counts.sum(1)[:, None] > 0,
    )
    probs = 0.99 * base + 0.01 / 256
    idx = np.arange(65536)
    centers = np.column_stack((idx % 256, idx // 256, probs.ravel()))
    data = geometry(centers, [(-0.4, -0.4, 0), (0.4, -0.4, 0), (0, 0.4, 0)])
    return data, probs, 0.99 * unigram + 0.01 / 256


def read_prior(device, data):
    scene = TriangleScene(device, data)
    kernel = device.create_compute_kernel(
        device.load_program("episodic.slang", ["readMass"])
    )
    values = dispatch(
        device, kernel, scene, np.empty(65536, np.float32), "masses"
    ).reshape(256, 256)
    return values.astype(np.float64)


def prior(device, folder):
    data, probs, uni = fit_prior(book(folder, name) for name in list(BOOKS)[:4])
    values = read_prior(device, data)
    np.testing.assert_allclose(values, probs, rtol=2e-6, atol=1e-10)
    return values, uni


def trace(device, kernel, raw):
    idx = np.arange(len(raw))
    center = np.column_stack((np.frombuffer(raw, np.uint8), idx, np.zeros(len(idx))))
    scene = TriangleScene(
        device, geometry(center, [(-0.3, 0.3, -0.3), (0.3, -0.3, -0.3), (0, 0, 0.3)])
    )
    start = perf_counter()
    hits = dispatch(
        device,
        kernel,
        scene,
        np.empty((len(raw) - 8, 4), np.int32),
        "outcomes",
        targets=len(raw) - 8,
    )
    return hits, {**scene.metrics, "dispatch_and_readback_s": perf_counter() - start}


def reference(raw):
    result = np.empty((len(raw) - 8, 3), np.int32)
    for t in range(8, len(raw)):
        start = max(0, t - 256)
        key = raw[t - 1 : t]
        cursor = t - 1
        recent = pred = -1
        best = 0
        for _ in range(16):
            row = raw.rfind(key, start, cursor)
            if row < 0:
                break
            n = 1
            while n < 8 and row - n >= start and raw[row - n] == raw[t - 1 - n]:
                n += 1
            if n > best:
                best, pred = n, raw[row + 1]
                if recent < 0:
                    recent = pred
            if n == 8:
                break
            cursor = row
        result[t - 8] = recent, pred, best
    return result


def evaluate(raw, hits, probs, uni):
    data = np.frombuffer(raw, np.uint8)
    y = data[8:]
    base = probs[data[7:-1], y]
    result = {
        "scored_bytes": len(y),
        "rays": int(hits[:, 3].sum()),
        "unigram_bpb": float(-np.log2(uni[y]).mean()),
        "bigram_bpb": float(-np.log2(base).mean()),
    }
    for key, pred, used in (
        ("one", hits[:, 0], hits[:, 0] >= 0),
        ("multi", hits[:, 1], hits[:, 2] >= 2),
    ):
        p = np.where(used, 0.75 * base + 0.25 * (y == pred), base)
        result[key] = {
            "bpb": float(-np.log2(p).mean()),
            "coverage": float(used.mean()),
            "successor_accuracy_on_matches": float(np.mean(pred[used] == y[used])),
        }
    result["passes"] = all(
        v - result["multi"]["bpb"] >= 0.05
        for v in (result["one"]["bpb"], result["bigram_bpb"])
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("name", choices=list(BOOKS)[4:])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    raw = book(args.folder, args.name)
    device, kernel = open_kernel("episodic.slang")
    probs, uni = prior(device, args.folder)
    hits, metrics = trace(device, kernel, raw)
    np.testing.assert_array_equal(reference(raw), hits[:, :3])
    # Rebuild two radically different futures at several cutoffs. Include
    # the first prediction whose target itself is changed in the comparison.
    for cut in (8, 257, 4096):
        altered = raw[:cut] + bytes(b ^ 255 for b in raw[cut : cut + 512])
        changed, _ = trace(device, kernel, altered)
        np.testing.assert_array_equal(changed[: cut - 7], hits[: cut - 7])
    result = {
        "book": args.name,
        "device": str(device.info.adapter_name),
        **evaluate(raw, hits, probs, uni),
        **metrics,
    }
    print(json.dumps(result, indent=2))
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
