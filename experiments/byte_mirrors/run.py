"""Write byte transitions as mirrors and decode predictions from Vulkan ray votes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "rtx"))
from run_specular import SpecularScene, open_kernel

REV = "f40d4af56598f728a14708619ce849b2734e7686"
FILES = ("README.md", "AGENTS.md", "GOAL.md")
VOTES = 32
GLOBAL_VOTES = 256
TOTAL = 256 * VOTES + GLOBAL_VOTES


def splits():
    parts = [[], [], []]
    for name in FILES:
        raw = subprocess.check_output(["git", "show", f"{REV}:{name}"], cwd=ROOT)
        a, b = len(raw) * 70 // 100, len(raw) * 85 // 100
        for group, chunk in zip(parts, (raw[:a], raw[a:b], raw[b:])):
            group.append(np.frombuffer(chunk, np.uint8).astype(np.int32))
    return parts


def counts(documents):
    conditional = np.zeros((256, 256), dtype=np.int64)
    unigram = np.zeros(256, dtype=np.int64)
    for doc in documents:
        np.add.at(conditional, (doc[:-1], doc[1:]), 1)
        np.add.at(unigram, doc, 1)
    return conditional, unigram


def quantiles(row, n):
    if row.sum() == 0:
        return np.full(n, -1, np.int32)
    ranks = (2 * np.arange(n) + 1) * row.sum() // (2 * n)
    return np.searchsorted(row.cumsum(), ranks, side="right").astype(np.int32)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def write_scene(dest):
    # Receivers occupy a distinct plane. A byte's 32 fixed launch positions
    # select samples, while each mirror's learned normal encodes a destination.
    i = np.arange(TOTAL)
    j = np.arange(256)
    origins = np.column_stack((i % 128, i // 128, np.zeros_like(i))).astype(np.float32)
    targets = np.column_stack((160 + j % 16, j // 16, np.full_like(j, 64))).astype(
        np.float32
    )
    slots = np.flatnonzero(dest >= 0)
    mirrors = origins[slots] + [0, 0, 2]
    outgoing = unit(targets[dest[slots]] - mirrors)
    normals = unit(outgoing - [0, 0, 1])
    u = unit(np.cross(normals, [0, 1, 0])) * 0.18
    v = unit(np.cross(normals, u)) * 0.18
    facts = np.stack([mirrors - u - v, mirrors + u - v, mirrors + v], axis=1)
    recv = targets[:, None, :] + np.array(
        [[-0.35, -0.35, 0], [0.35, -0.35, 0], [0, 0.35, 0]]
    )
    return {
        "station_pos": np.vstack([targets, origins]),
        "relation_dir": np.array([[0, 0, 1]], np.float32),
        "tri_vertices": np.concatenate([recv, facts]).astype(np.float32),
        "tri_kind": np.repeat(np.array([0, 1], np.uint32), [256, len(facts)]),
        "tri_station": np.concatenate([j, np.full(len(facts), -1)]).astype(np.int32),
        "tri_reflectance": np.ones(256 + len(facts), np.float32),
        "t_min": np.float32(0.001),
        "emit_offset": np.float32(0),
    }


def load_scene(device, path):
    # The reader gets a saved geometric scene, with no training counts or destination array.
    with np.load(path, allow_pickle=False) as data:
        return SpecularScene(device, {k: data[k] for k in data.files})


def trace_slots(gpu, kernel, start, count):
    queries = {
        "start": 256 + np.arange(start, start + count, dtype=np.int32),
        "program": np.zeros((count, 1), np.int32),
    }
    receivers, metrics = gpu.trace(kernel, queries)
    return receivers[:, 0], metrics


def blend(conditional, global_counts):
    if global_counts.sum() == 0:
        raise ValueError("No unigram receivers reached")
    prior = global_counts / global_counts.sum()
    totals = conditional.sum(axis=1, keepdims=True)
    rows = np.divide(
        conditional,
        totals,
        out=np.broadcast_to(prior, conditional.shape).copy(),
        where=totals > 0,
    )
    return 0.99 * (0.8 * rows + 0.2 * prior) + 0.01 / 256, 0.99 * prior + 0.01 / 256


def decode(ids):
    votes = ids[: 256 * VOTES].reshape(256, VOTES)
    context, slot = np.nonzero(votes >= 0)
    histogram = np.zeros((256, 256), np.int64)
    np.add.at(histogram, (context, votes[context, slot]), 1)
    global_hits = ids[256 * VOTES :]
    return blend(histogram, np.bincount(global_hits[global_hits >= 0], minlength=256))


def score(prob, docs):
    bits = np.concatenate([-np.log2(prob[doc[:-1], doc[1:]]) for doc in docs])
    return float(bits.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out", type=Path, help="Directory for scene checkpoints and result JSON"
    )
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    train, validation, test = splits()
    conditional, unigram = counts(train)
    expected = np.concatenate(
        [
            *(quantiles(row, VOTES) for row in conditional),
            quantiles(unigram, GLOBAL_VOTES),
        ]
    )
    shuffled = expected.copy()
    live = np.flatnonzero(expected[: 256 * VOTES] >= 0)
    shuffled[live] = np.random.default_rng(20261125).permutation(expected[live])
    device, kernel = open_kernel()
    exact, exact_prior = blend(conditional, unigram)
    oracle, _ = decode(expected)
    probabilities = {
        "exact_bigram": exact,
        "exact_unigram": np.tile(exact_prior, (256, 1)),
        "finite_vote_oracle": oracle,
    }
    result = {
        "corpus_revision": REV,
        "files": FILES,
        "training_bytes": sum(map(len, train)),
        "hardware": {},
    }
    for name, destinations in (("rtx", expected), ("shuffled", shuffled)):
        path = args.out / f"{name}.npz"
        np.savez(path, **write_scene(destinations))
        hits, metrics = trace_slots(load_scene(device, path), kernel, 0, TOTAL)
        probabilities[name], prior = decode(hits)
        metrics.update(
            vote_mismatches=int(np.count_nonzero(hits != destinations)),
            serialized_bytes=path.stat().st_size,
            receiver_hits=int(np.count_nonzero(hits >= 0)),
            launched_rays=TOTAL,
        )
        result["hardware"][name] = metrics
        if name == "rtx":
            probabilities["traced_unigram"] = np.tile(prior, (256, 1))
    for name, docs in (("validation", validation), ("test", test)):
        result[name] = {
            "predicted_bytes": sum(len(doc) - 1 for doc in docs),
            "seen_context_fraction": float(
                np.mean(
                    conditional.sum(axis=1)[np.concatenate([doc[:-1] for doc in docs])]
                    > 0
                )
            ),
            "bits_per_byte": {
                model: score(prob, docs) for model, prob in probabilities.items()
            },
        }
    report = json.dumps(result, indent=2)
    (args.out / "result.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
