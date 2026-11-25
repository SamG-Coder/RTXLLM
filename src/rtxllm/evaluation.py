"""E1 baselines and metrics.

Methods (all scored by :func:`score` with the same abstention rule):

* ``graph``      symbolic traversal over the fact table (correctness ceiling;
                 allowed to read facts by index, nothing else is).
* ``lookup``     whole-query direct lookup: a table of *training* queries
                 (atomic facts as length-1 queries + training paths) -> answer.
* ``trace``      full multi-bounce trace of the scene.
* ``trace_1hop`` same scene, program capped at its first relation (one hop).
* ``trace_shuffled`` scene whose stations were moved onto each other's
                 locations after facts were written (randomized geometry).
"""

from __future__ import annotations

import time

import numpy as np

from rtxllm.core.scene import shuffle_destinations
from rtxllm.core.world import AMBIGUOUS, MISS, UNIQUE, traverse
from rtxllm.cpu.tracer import trace

ABSTAIN_THRESHOLD = 0.5


def dense(dist, S):
    v = np.zeros(S, np.float32)
    for k, p in dist.items():
        v[k] = p
    return v


def method_graph(world, paths):
    return np.stack([dense(traverse(world.facts, p.start, p.program)[0], world.n_stations) for p in paths]), {}


def method_lookup(world, paths):
    table = {(s, (r,)): {o: 1.0 / len(objs) for o in objs} for (s, r), objs in world.facts.items()}
    for p in world.split("train"):
        table[p.key] = p.answer
    out = np.stack([dense(table.get(p.key, {}), world.n_stations) for p in paths])
    return out, {"table_size": len(table)}


def method_trace(scene, paths, cap=None):
    t0 = time.perf_counter()
    progs = [p.program[:cap] if cap else p.program for p in paths]
    r = trace(scene, [p.start for p in paths], progs)
    dt = time.perf_counter() - t0
    extra = {
        "cpu_seconds": dt,
        "cpu_ms_per_query": 1e3 * dt / max(1, len(paths)),
        "rays_per_query": r.rays_traced / max(1, len(paths)),
        "crosstalk_rate": float((r.fact_hits_off_emit > 0).mean()),
        "miss_energy": float(r.miss.mean()),
        "floor_energy": float(r.floor.mean()),
        "overflow_energy": float(r.overflow.mean()),
        "max_conservation_error": float(np.abs(r.answer.sum(1) + r.miss + r.floor + r.overflow - 1).max()),
    }
    return r.answer, extra, r


def per_hop_agreement(result, paths):
    """Fraction of hops on UNIQUE paths where the dominant lineage's station
    matches the symbolic station sequence."""
    agree = total = 0
    for i, p in enumerate(paths):
        if p.category != UNIQUE or not result.receiver_paths[i]:
            continue
        best = max(result.receiver_paths[i], key=lambda x: x[1])[0]
        for h in range(1, len(p.stations)):
            total += 1
            agree += int(h < len(best) and best[h] == p.stations[h])
    return agree / max(1, total)


def score(answers, paths, threshold=ABSTAIN_THRESHOLD):
    """Metrics per (length, category)."""
    tot = answers.sum(1)
    pred = np.where(tot >= threshold, answers.argmax(1), -1)
    out = {}
    for L in sorted({len(p.program) for p in paths}):
        for cat in (UNIQUE, AMBIGUOUS, MISS):
            idx = [i for i, p in enumerate(paths) if len(p.program) == L and p.category == cat]
            if not idx:
                continue
            m = {"n": len(idx), "abstain_rate": float(np.mean(pred[idx] == -1))}
            if cat == UNIQUE:
                gold = np.array([paths[i].stations[-1] for i in idx])
                m["accuracy"] = float(np.mean(pred[idx] == gold))
                cyc = [j for j, i in enumerate(idx) if paths[i].has_cycle]
                m["n_cycle"] = len(cyc)
                if cyc:
                    m["cycle_accuracy"] = float(np.mean(pred[idx][cyc] == gold[cyc]))
            elif cat == MISS:
                m["abstain_accuracy"] = m["abstain_rate"]
            else:
                mass, tv = [], []
                for i in idx:
                    g = dense(paths[i].answer, answers.shape[1])
                    mass.append(float(answers[i][g > 0].sum()))
                    tv.append(0.5 * float(np.abs(answers[i] - g).sum()))
                m["mass_on_answer_set"] = float(np.mean(mass))
                m["total_variation"] = float(np.mean(tv))
            out[f"L{L}/{cat}"] = m
    return out


def evaluate(world, scene, split, shuffle_seed=0):
    paths = world.split(split)
    res = {}
    a, extra = method_graph(world, paths)
    res["graph"] = {"metrics": score(a, paths), **extra}
    a, extra = method_lookup(world, paths)
    res["lookup"] = {"metrics": score(a, paths), **extra}
    a, extra, r = method_trace(scene, paths)
    res["trace"] = {"metrics": score(a, paths), "per_hop_agreement": per_hop_agreement(r, paths), **extra}
    a, extra, _ = method_trace(scene, paths, cap=1)
    res["trace_1hop"] = {"metrics": score(a, paths), **extra}
    a, extra, _ = method_trace(shuffle_destinations(scene, shuffle_seed), paths)
    res["trace_shuffled"] = {"metrics": score(a, paths), **extra}
    return res


def check_acceptance(results_by_seed, lengths=(2, 3), min_acc=0.95, min_lift=0.50, min_n=256):
    """Pre-registered E1 criteria (issue #1 / #4). Returns list of (check, ok, detail)."""
    checks = []
    for seed, res in results_by_seed.items():
        for L in lengths:
            k = f"L{L}/{UNIQUE}"
            m = res["trace"]["metrics"][k]
            checks.append((f"seed{seed} {k} n>={min_n}", m["n"] >= min_n, m["n"]))
            checks.append((f"seed{seed} {k} trace acc>={min_acc}", m["accuracy"] >= min_acc, round(m["accuracy"], 4)))
            for b in ("trace_1hop", "lookup", "trace_shuffled"):
                lift = m["accuracy"] - res[b]["metrics"][k]["accuracy"]
                checks.append((f"seed{seed} {k} lift over {b}>={min_lift}", lift >= min_lift, round(lift, 4)))
    return checks
