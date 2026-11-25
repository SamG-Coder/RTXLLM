"""Fit a byte-pair router and read next-byte votes after two RTX reflections."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "byte_mirrors"))
import run as bigram
from e1 import E1Scene

SHA256 = "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163"
N = bigram.VOTES


def corpus(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Corpus digest does not match the preregistered edition")
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1]
    data = np.frombuffer(body, np.uint8).astype(np.int32)
    return np.split(data, [len(data) * 70 // 100, len(data) * 85 // 100])


def locations(i, columns, pitch, offset):
    return (
        np.column_stack([i % columns, i // columns, np.zeros_like(i)]) * pitch + offset
    )


def mirrors(origins, incoming, targets):
    center = origins + 2 * incoming
    normal = bigram.unit(bigram.unit(targets - center) - incoming)
    u = bigram.unit(np.cross(normal, [0, 1, 0])) * 0.06
    v = bigram.unit(np.cross(normal, u)) * 0.06
    return np.stack((center - u - v, center + u - v, center + v), axis=1)


def scene_arrays(keys, votes, routing):
    byte = np.arange(256)
    outputs = locations(byte, 16, 3, [280, 0, 90])
    starts = locations(byte, 16, 4, [0, 0, 0])
    pair = locations(np.arange(len(keys)), 48, 4, [80, 0, 45])
    dirs = bigram.unit(
        np.column_stack(
            ((byte % 16 - 7.5) * 0.1, (byte // 16 - 7.5) * 0.1, np.ones(256))
        )
    )
    first = mirrors(starts[keys // 256], dirs[keys % 256], pair[routing])
    second = mirrors(
        np.repeat(pair, N, axis=0),
        np.tile(dirs[:N], (len(pair), 1)),
        outputs[votes.ravel()],
    )
    receiver_centers = np.vstack((outputs, pair))
    receivers = receiver_centers[:, None, :] + [
        [-0.8, -0.8, 0],
        [0.8, -0.8, 0],
        [0, 0.8, 0],
    ]
    triangles = np.concatenate((receivers, first, second)).astype(np.float32)
    owners = np.concatenate(
        (byte, 512 + np.arange(len(pair)), np.full(len(first) + len(second), -1))
    )
    return {
        "params": json.dumps(
            {
                "hop_mode": "specular",
                "max_events": 4,
                "t_min": 0.001,
                "emit_offset": 0.1,
                "emit_range": 2.1,
                "arrival_range": 1.0,
                "transport_range": 512.0,
                "energy_floor": 0.001,
                "transport_hits_facts": False,
            }
        ),
        "tri_vertices": triangles,
        "tri_station": owners.astype(np.int32),
        "tri_reflectance": np.ones(len(owners), np.float32),
        "tri_kind": (owners < 0).astype(np.uint32),
        "tri_displacement": np.zeros((len(owners), 3), np.float32),
        "relation_dir": dirs.astype(np.float32),
        "station_pos": np.vstack((outputs, starts, pair)).astype(np.float32),
    }


def gpu_votes(device, kernel, checkpoint, contexts):
    with np.load(checkpoint, allow_pickle=False) as saved:
        scene = E1Scene(device, {k: saved[k] for k in saved.files})
    queries = np.repeat(contexts, N)
    starts = (256 + queries // 256).astype(np.int32)
    programs = np.column_stack(
        (queries % 256, np.tile(np.arange(N), len(contexts)))
    ).astype(np.int32)
    receivers = []
    gpu_ms = 0
    begin = perf_counter()
    for i in range(0, len(queries), 512):
        result, m = scene.trace(kernel, starts[i : i + 512], programs[i : i + 512])
        receivers.append(result["paths"][:, 1:])
        gpu_ms += m["dispatch_gpu_ms"]
    paths = np.concatenate(receivers).reshape(len(contexts), N, 2)
    return paths, dict(
        scene.metrics,
        dispatch_gpu_ms=gpu_ms,
        trace_wall_ms=(perf_counter() - begin) * 1000,
    )


def pairs(data):
    return data[:-2] * 256 + data[1:-1]


def distribution(ids, backoff):
    hist = np.zeros((len(ids), 256), np.float64)
    row, slot = np.nonzero((ids >= 0) & (ids < 256))
    np.add.at(hist, (row, ids[row, slot]), 1)
    return interpolate(hist, backoff)


def interpolate(hist, backoff):
    totals = hist.sum(1, keepdims=True)
    # Match the pilot's uniform escape. The lower order distribution already
    # includes its escape, so mix the two smoothed components directly.
    raw = np.divide(
        hist, totals, out=np.zeros_like(hist, dtype=float), where=totals > 0
    )
    mixture = 0.8 * (0.99 * raw + 0.01 / 256) + 0.2 * backoff
    return np.where(totals > 0, mixture, backoff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    train, val, test = corpus(args.corpus)
    held = val if args.split == "validation" else test
    keys, rows = np.unique(pairs(train), return_inverse=True)
    hist = np.zeros((len(keys), 256), np.int64)
    np.add.at(hist, (rows, train[2:]), 1)
    votes = np.stack([bigram.quantiles(row, N) for row in hist])
    conditional, unigram = bigram.counts([train])
    low_votes = np.concatenate(
        [
            *(bigram.quantiles(row, N) for row in conditional),
            bigram.quantiles(unigram, bigram.GLOBAL_VOTES),
        ]
    )
    device, kernel = bigram.open_kernel()
    low_path = args.out / "one_hop.npz"
    np.savez(low_path, **bigram.write_scene(low_votes))
    launched, _ = bigram.trace_slots(
        bigram.load_scene(device, low_path), kernel, 0, bigram.TOTAL
    )
    low, _ = bigram.decode(launched)
    exact_low, _ = bigram.blend(conditional, unigram)
    e1_kernel = device.create_compute_kernel(device.load_program("e1.slang", ["trace"]))
    contexts, inverse = np.unique(pairs(held), return_inverse=True)
    # Include every written pair in the parity check, and query held-out novel
    # pairs via their byte input rather than looking them up for RTX inference.
    all_keys = np.union1d(contexts, keys)
    index = np.searchsorted(all_keys, keys)
    selected = np.searchsorted(all_keys, contexts)[inverse]
    backoff = low[all_keys % 256]
    oracle_votes = np.full((len(all_keys), N), -1, np.int32)
    oracle_votes[index] = votes
    exact_hist = np.zeros((len(all_keys), 256), np.int64)
    exact_hist[index] = hist
    models = {
        "one_hop": backoff,
        "finite_vote_oracle": distribution(oracle_votes, backoff),
        "exact_trigram": interpolate(exact_hist, exact_low[all_keys % 256]),
    }
    report = {
        "split": args.split,
        "training_bytes": len(train),
        "scored_bytes": len(held) - 2,
        "learned_pairs": len(keys),
        "seen_pair_fraction": float(np.isin(pairs(held), keys).mean()),
        "bigram_vote_mismatches": int(np.count_nonzero(launched != low_votes)),
        "hardware": {},
    }
    for name, routing in (
        ("two_hop", np.arange(len(keys))),
        ("shuffled", np.random.default_rng(20261125).permutation(len(keys))),
    ):
        checkpoint = args.out / f"{name}.npz"
        np.savez(checkpoint, **scene_arrays(keys, votes, routing))
        paths, metrics = gpu_votes(device, e1_kernel, checkpoint, all_keys)
        wanted = np.full((len(all_keys), N, 2), -2, np.int32)
        wanted[:, :, 0] = -1
        wanted[index, :, 0] = (512 + routing)[:, None]
        wanted[index, :, 1] = votes[routing]
        metrics.update(
            path_mismatches=int(np.count_nonzero(np.any(paths != wanted, axis=2))),
            tested_paths=len(all_keys) * N,
            checkpoint_bytes=checkpoint.stat().st_size,
        )
        report["hardware"][name] = metrics
        models[name] = distribution(paths[:, :, 1], backoff)
    target = held[2:]
    report["bits_per_byte"] = {
        key: float(-np.log2(p[selected, target]).mean()) for key, p in models.items()
    }
    report["success"] = bool(
        report["hardware"]["two_hop"]["path_mismatches"] == 0
        and all(
            report["bits_per_byte"][k] - report["bits_per_byte"]["two_hop"] >= 0.1
            for k in ("one_hop", "shuffled")
        )
    )
    output = json.dumps(report, indent=2)
    (args.out / f"{args.split}.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
