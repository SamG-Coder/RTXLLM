"""Fit shared routing mirrors using next-byte errors from two-reflection RTX paths."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_reflection_bytes"))
from experiment import N, bigram, locations, mirrors

BANKS = 8
ROUNDS = 24
DIGEST = "49a0b2726606e1290ac03a63978fa1dd1bd38a8d805704d98265f393533ea094"


def dataset(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Corpus digest mismatch")
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1]
    data = np.frombuffer(body, dtype=np.uint8).astype(np.int32)
    return np.split(data, [len(data) * 70 // 100, len(data) * 85 // 100])


def make_scene(aim, votes):
    byte = np.arange(256)
    starts = locations(byte, 16, 3, [0, 0, 0])
    banks = locations(np.arange(BANKS), BANKS, 4, [72, 0, 35])
    outputs = locations(byte, 16, 2, [150, 0, 75])
    # Aim is a continuous horizontal coordinate on the intermediate receiver plane.
    intercepts = np.column_stack((72 + 4 * aim, np.zeros(256), np.full(256, 35)))
    incoming = np.array([0, 0, 1.0])
    angles = 2 * np.pi * np.arange(N) / N
    fan = bigram.unit(
        np.column_stack((0.65 * np.cos(angles), 0.65 * np.sin(angles), np.ones(N)))
    )
    first = mirrors(starts, np.tile(incoming, (256, 1)), intercepts)
    second = mirrors(
        np.repeat(banks, N, axis=0), np.tile(fan, (BANKS, 1)), outputs[votes.ravel()]
    )
    out_tri = outputs[:, None, :] + [[-0.65, -0.65, 0], [0.65, -0.65, 0], [0, 0.65, 0]]
    # Adjacent strips share edges, so a small rotation can move a ray to another bank.
    strip = np.array(
        [[[-2, -1, 0], [2, -1, 0], [2, 1, 0]], [[-2, -1, 0], [2, 1, 0], [-2, 1, 0]]]
    )
    bank_tri = (banks[:, None, None, :] + strip).reshape(-1, 3, 3)
    tri = np.concatenate([out_tri, bank_tri, first, second]).astype(np.float32)
    owners = np.concatenate(
        (
            byte,
            np.repeat(512 + np.arange(BANKS), 2),
            np.full(len(first) + len(second), -1),
        )
    )
    return {
        "tri_vertices": tri,
        "tri_station": owners.astype(np.int32),
        "tri_kind": (owners < 0).astype(np.uint32),
        "tri_reflectance": np.ones(len(tri), np.float32),
        "station_pos": np.vstack([outputs, starts, banks]).astype(np.float32),
        "relation_dir": np.vstack([fan, incoming]).astype(np.float32),
        "emit_offset": np.float32(0.01),
        "t_min": np.float32(0.001),
    }


def vote_distribution(final):
    if np.any((final < 0) | (final >= 256)):
        raise ValueError("Output ray failed to reach a byte receiver")
    counts = np.zeros((len(final), 256), np.int32)
    np.add.at(counts, (np.repeat(np.arange(len(final)), N), final.ravel()), 1)
    return 0.99 * counts / N + 0.01 / 256


class OpticalTrainer:
    def __init__(self, device, kernel, scene):
        self.kernel = kernel
        self.gpu = bigram.SpecularScene(device, scene)
        self.queries = {
            "start": np.repeat(256 + np.arange(256, dtype=np.int32), N),
            "program": np.column_stack(
                (
                    np.full(256 * N, N, np.int32),
                    np.tile(np.arange(N, dtype=np.int32), 256),
                )
            ),
        }
        self.dispatches = 0
        self.dispatch_ms = 0.0

    def read(self):
        paths, metrics = self.gpu.trace(self.kernel, self.queries)
        self.dispatches += 1
        self.dispatch_ms += metrics["dispatch_gpu_ms"]
        paths = paths.reshape(256, N, 2)
        route = paths[:, 0, 0] - 512
        if np.any((route < 0) | (route >= BANKS)) or np.any(
            paths[:, :, 0] != paths[:, :1, 0]
        ):
            raise ValueError(
                "A launch mirror failed to route its rays to a shared bank"
            )
        return route, vote_distribution(paths[:, :, 1])

    def trial(self, aim, votes):
        self.gpu.refit_vertices(make_scene(aim, votes)["tri_vertices"])
        return self.read()


def bank_emissions(train, route, default):
    pooled = np.zeros((BANKS, 256), np.int64)
    np.add.at(pooled, route, train)
    return np.stack(
        [bigram.quantiles(row, N) if row.sum() else default for row in pooled]
    )


def nll(train, distributions):
    return -(train * np.log2(distributions)).sum(1)


def cpu_assignment(train, route, votes, default):
    # Exhaustive reassignment to every bank, with the same 32-vote emissions.
    for _ in range(ROUNDS):
        cost = -(train @ np.log2(vote_distribution(votes)).T)
        candidate = cost.argmin(1)
        route = np.where(
            cost[np.arange(256), candidate] < cost[np.arange(256), route],
            candidate,
            route,
        )
        votes = bank_emissions(train, route, default)
    return vote_distribution(votes)[route]


def optimize_routes(
    optical, train, default, rng, aim, votes, route, p, rounds, scramble=None
):
    active = np.flatnonzero(train.sum(1))
    changed = 0
    for _ in range(rounds):
        proposal = rng.uniform(-0.49, BANKS - 0.51, 256)
        trial_route, trial_p = optical.trial(proposal, votes)
        accept = nll(train, trial_p) < nll(train, p)
        if scramble is not None:
            accept[active] = scramble.permutation(accept[active])
        aim = np.where(accept, proposal, aim)
        changed += int(np.count_nonzero(accept & (trial_route != route)))
        # Re-trace the accepted geometry before crediting its receivers with observations.
        route, _ = optical.trial(aim, votes)
        votes = bank_emissions(train, route, default)
        route, p = optical.trial(aim, votes)
    return aim, votes, route, p, changed


def fit(device, kernel, train, default, seed, control, out):
    rng = np.random.default_rng(seed)
    scramble = np.random.default_rng(seed + 700)
    active = np.flatnonzero(train.sum(1))
    aim = rng.uniform(-0.49, BANKS - 0.51, 256)
    votes = np.tile(default, (BANKS, 1))
    start = perf_counter()
    optical = OpticalTrainer(device, kernel, make_scene(aim, votes))
    route, _ = optical.read()
    votes = bank_emissions(train, route, default)
    route, p = optical.trial(aim, votes)
    frozen = p.copy()
    cpu = (
        cpu_assignment(train, route.copy(), votes.copy(), default)
        if control == "feedback"
        else None
    )
    initial_loss = float(nll(train, p).sum() / train.sum())
    aim, votes, route, p, changed = optimize_routes(
        optical,
        train,
        default,
        rng,
        aim,
        votes,
        route,
        p,
        ROUNDS,
        scramble if control == "shuffled_credit" else None,
    )
    ckpt = out / f"{control}-{seed}.npz"
    np.savez(ckpt, **make_scene(aim, votes))
    with np.load(ckpt, allow_pickle=False) as data:
        persisted = OpticalTrainer(device, kernel, {k: data[k] for k in data.files})
    written_route, readout = persisted.read()
    np.testing.assert_array_equal(written_route, route)
    np.testing.assert_array_equal(readout, p)
    metrics = {
        "train_initial_bpb": initial_loss,
        "train_final_bpb": float(nll(train, p).sum() / train.sum()),
        "route_changes": changed,
        "occupied_banks": len(np.unique(route[active])),
        "checkpoint_bytes": ckpt.stat().st_size,
        "training_dispatches": optical.dispatches,
        "training_dispatch_ms": optical.dispatch_ms,
        "fit_and_verify_wall_s": perf_counter() - start,
    }
    return readout, frozen, cpu, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    training, validation, test = dataset(args.corpus)
    held = validation if args.split == "validation" else test
    train, uni = bigram.counts([training])
    default = bigram.quantiles(uni, N)
    base = np.divide(
        train,
        train.sum(1, keepdims=True),
        out=np.tile(uni / uni.sum(), (256, 1)),
        where=train.sum(1, keepdims=True) > 0,
    )
    exact = 0.99 * base + 0.01 / 256
    device, kernel = bigram.open_kernel()
    report = {
        "split": args.split,
        "train_bytes": len(training),
        "scored_bytes": len(held) - 1,
        "seeds": [],
    }
    for seed in range(3):
        models = {"exact_bigram": exact}
        work = {}
        for control in ("feedback", "shuffled_credit"):
            p, frozen, cpu, metrics = fit(
                device, kernel, train, default, seed, control, args.out
            )
            models[control] = p
            work[control] = metrics
            if cpu is not None:
                models.update(frozen=frozen, cpu_assignment=cpu)
        scores = {name: bigram.score(p, [held]) for name, p in models.items()}
        result = {
            "seed": seed,
            "bits_per_byte": scores,
            "training": work,
            "passes": all(
                scores[c] - scores["feedback"] >= 0.1
                for c in ("frozen", "shuffled_credit")
            ),
        }
        print(json.dumps(result), flush=True)
        report["seeds"].append(result)
    (args.out / f"{args.split}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
