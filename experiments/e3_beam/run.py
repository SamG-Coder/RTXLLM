"""E3: answer never-written facts with beams over a learned layout (issue #11).

    python experiments/e3_beam/run.py sweep            # val: geometry/beam selection
    python experiments/e3_beam/run.py test --dish 0.3 --width 1.5 --knn 5   # sealed test, once

Selection rule (fixed before the sweep): maximise mean val held-out single-fact
accuracy on the force layout subject to written-fact accuracy >= 0.90. Two-hop
accuracy is not used for selection and is only computed on the sealed test.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rtxllm.core.cluster_world import ClusterConfig, generate_clustered  # noqa: E402
from rtxllm.core.scene import GeometryConfig, build_scene  # noqa: E402
from rtxllm.cpu.beam import BeamParams, BeamTracer  # noqa: E402
from rtxllm.learning.placement import (force_layout, layout_quality, oracle_layout,  # noqa: E402
                                       random_lattice)

SEEDS = (0, 1, 2)
OUT = Path(__file__).resolve().parent / "results"
FACET = 0.05
RECEIVER = 0.075
RAYS = 256
CORE = 0.5


def layouts(world, seed):
    S = world.n_stations
    return {"random": random_lattice(S, seed),
            "force": force_layout(world.written, S, seed),
            "oracle": oracle_layout(world.cluster, seed)}


def written_sample(world, n=300, seed=0):
    keys = sorted(world.written)
    rng = np.random.default_rng(seed)
    return [keys[i] for i in sorted(rng.choice(len(keys), size=n, replace=False))]


_RTX = {}


def rtx_context():
    """Lazily open the RTX device (only for --backend rtx)."""
    if not _RTX:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "rtx"))
        from beam import BeamRTX, open_kernel
        _RTX["device"], _RTX["kernel"] = open_kernel("beam.slang")
        _RTX["cls"] = BeamRTX
    return _RTX


class _Checked:
    """RTX tracer that re-traces a subset on the CPU reference and records parity."""

    def __init__(self, cpu, parity_limit):
        ctx = rtx_context()
        self.cpu, self.gpu, self.limit = cpu, ctx["cls"](ctx["device"], ctx["kernel"], cpu), parity_limit
        self.parity = []

    def trace(self, starts, programs):
        g = self.gpu.trace(starts, programs)
        n = len(starts) if self.limit is None else min(self.limit, len(starts))
        c = self.cpu.trace(starts[:n], programs[:n])
        self.parity.append({"queries_checked": n, "of": len(starts),
                            "max_answer_diff": float(np.abs(c.answer - g.answer[:n]).max()),
                            "max_lost_diff": float(np.abs(c.lost - g.lost[:n]).max()),
                            "pred_agreement": float(np.mean(c.predict() == g.predict()[:n]))})
        return g


def beam_eval(world, pos, dish, width, split, two_hop=True, backend="cpu"):
    geom = GeometryConfig(receiver_radius=RECEIVER, fact_radius=FACET, dish_radius=dish)
    scene = build_scene(world.with_positions(pos), geom)
    bt = BeamTracer(scene, BeamParams(width=width, core=CORE if width > 0 else 1.0, rays=RAYS,
                                      fact_distance=geom.fact_distance))
    out = {"triangles": scene.n_triangles, "backend": backend}
    if backend == "rtx":
        bt_one = _Checked(bt, None)       # single-hop sets: full CPU parity
        bt_two = _Checked(bt, 50)         # two-hop: CPU parity on the first 50 queries
    else:
        bt_one = bt_two = bt
    held = world.heldout[split]
    t0 = time.perf_counter()
    r = bt_one.trace([s for s, _ in held], [(x,) for _, x in held])
    out["seconds_heldout"] = time.perf_counter() - t0
    gold = np.array([world.truth[k] for k in held])
    pred = r.predict()
    out["heldout_acc"] = float(np.mean(pred == gold))
    out["heldout_abstain"] = float(np.mean(pred < 0))
    out["heldout_lost_energy"] = float(r.lost.mean())
    out["rays_per_query"] = r.rays / len(held)
    ws = written_sample(world, seed=world.config.seed)
    rw = bt_one.trace([s for s, _ in ws], [(x,) for _, x in ws])
    wgold = np.array([world.written[k][0] for k in ws])
    wp = rw.predict()
    exc = np.array([world.truth[k] != world.prototype[world.cluster[k[0]], k[1]] for k in ws])
    out["written_acc"] = float(np.mean(wp == wgold))
    out["written_exception_acc"] = float(np.mean((wp == wgold)[exc])) if exc.any() else None
    if two_hop:
        th = world.two_hop[split]
        t0 = time.perf_counter()
        r2 = bt_two.trace([s for s, *_ in th], [(r1, r2_) for _, r1, r2_, _ in th])
        out["seconds_two_hop"] = time.perf_counter() - t0
        out["two_hop_acc"] = float(np.mean(r2.predict() == np.array([g for *_, g in th])))
    if backend == "rtx":
        out["rtx_cpu_parity"] = bt_one.parity + bt_two.parity
    return out


def knn_predict(world, pos, s, r, k):
    if (s, r) in world.written:
        return world.written[(s, r)][0]
    cands = np.array([x for x in range(world.n_stations) if x != s and (x, r) in world.written])
    d = np.linalg.norm(pos[cands] - pos[s], axis=1)
    nn = cands[np.argsort(d, kind="stable")[:k]]
    votes = Counter(world.written[(x, r)][0] for x in nn)
    best = max(votes.values())
    for x in nn:                         # tie -> nearest neighbour's answer
        if votes[world.written[(x, r)][0]] == best:
            return world.written[(x, r)][0]


def knn_eval(world, pos, k, split):
    held = world.heldout[split]
    acc = np.mean([knn_predict(world, pos, s, r, k) == world.truth[(s, r)] for s, r in held])
    th = world.two_hop[split]
    acc2 = np.mean([knn_predict(world, pos, knn_predict(world, pos, s, r1, k), r2, k) == g for s, r1, r2, g in th])
    return {"heldout_acc": float(acc), "two_hop_acc": float(acc2), "written_acc": 1.0}


def cluster_majority(world, split):
    def pred(s, r):
        mem = np.flatnonzero(world.cluster == world.cluster[s])
        v = Counter(world.written[(x, r)][0] for x in mem if x != s and (x, r) in world.written)
        return v.most_common(1)[0][0] if v else -1
    held = world.heldout[split]
    return {"heldout_acc": float(np.mean([pred(s, r) == world.truth[(s, r)] for s, r in held]))}


def env():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}


def sweep():
    res = {}
    for seed in SEEDS:
        world = generate_clustered(ClusterConfig(seed=seed))
        lay = layouts(world, seed)
        res[seed] = {"layout_quality": {k: layout_quality(v, world.cluster) for k, v in lay.items()},
                     "beam": {}, "knn": {}}
        for dish in (0.0, 0.15, 0.3, 0.5):
            for width in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                if width == 0.0 and dish > 0:
                    continue
                key = f"dish{dish}_w{width}"
                m = beam_eval(world, lay["force"], dish, width, "val", two_hop=False)
                res[seed]["beam"][key] = m
                print(f"seed{seed} force {key}: held={m['heldout_acc']:.3f} abst={m['heldout_abstain']:.3f} "
                      f"written={m['written_acc']:.3f} tris={m['triangles']} "
                      f"{m['seconds_heldout']:.1f}s", flush=True)
        for k in (1, 3, 5, 10, 20):
            res[seed]["knn"][k] = knn_eval(world, lay["force"], k, "val")
            print(f"seed{seed} knn k={k}: {res[seed]['knn'][k]}", flush=True)
    keys = res[SEEDS[0]]["beam"].keys()
    table = {k: (np.mean([res[s]["beam"][k]["heldout_acc"] for s in SEEDS]),
                 np.mean([res[s]["beam"][k]["written_acc"] for s in SEEDS])) for k in keys}
    ok = {k: v for k, v in table.items() if v[1] >= 0.90}
    best = max(ok, key=lambda k: ok[k][0]) if ok else None
    kbest = max((1, 3, 5, 10, 20), key=lambda k: np.mean([res[s]["knn"][k]["heldout_acc"] for s in SEEDS]))
    print("selected beam:", best, table.get(best), "selected knn k:", kbest)
    OUT.mkdir(exist_ok=True)
    (OUT / "val_sweep.json").write_text(json.dumps({"env": env(), "results": res, "selected": best,
                                                    "selected_k": kbest}, indent=1))


def test(dish, width, k, backend="cpu"):
    res = {}
    for seed in SEEDS:
        world = generate_clustered(ClusterConfig(seed=seed))
        lay = layouts(world, seed)
        row = {"layout_quality": {n: layout_quality(v, world.cluster) for n, v in lay.items()}}
        for name, pos in lay.items():
            row[f"beam_{name}"] = beam_eval(world, pos, dish, width, "test", backend=backend)
            row[f"pencil_{name}"] = beam_eval(world, pos, dish, 0.0, "test", backend=backend)
            row[f"knn_{name}"] = knn_eval(world, pos, k, "test")
        row["cluster_majority"] = cluster_majority(world, "test")
        row["graph_written_only"] = {"heldout_acc": 0.0, "note": "held-out facts are not written; abstains"}
        res[seed] = row
        print(f"seed{seed}: " + " | ".join(
            f"{m}: held={v['heldout_acc']:.3f}" + (f" 2hop={v['two_hop_acc']:.3f}" if "two_hop_acc" in v else "")
            + (f" written={v['written_acc']:.3f}" if "written_acc" in v else "")
            for m, v in row.items() if isinstance(v, dict) and "heldout_acc" in v), flush=True)
    checks = []
    for seed, row in res.items():
        b = row["beam_force"]
        checks.append((f"seed{seed} beam_force held >= 0.5", b["heldout_acc"] >= 0.5, b["heldout_acc"]))
        for ctrl in ("beam_random", "pencil_force"):
            lift = b["heldout_acc"] - row[ctrl]["heldout_acc"]
            checks.append((f"seed{seed} lift over {ctrl} >= 0.4", lift >= 0.4, round(lift, 4)))
        checks.append((f"seed{seed} beam_force 2hop >= 0.4", b["two_hop_acc"] >= 0.4, b["two_hop_acc"]))
    for c in checks:
        print(("PASS " if c[1] else "FAIL ") + c[0], c[2])
    OUT.mkdir(exist_ok=True)
    parity = [p for row in res.values() for v in row.values() if isinstance(v, dict)
              for p in v.get("rtx_cpu_parity", [])]
    if parity:
        worst = max(p["max_answer_diff"] for p in parity)
        agree = min(p["pred_agreement"] for p in parity)
        print(f"RTX vs CPU parity over {len(parity)} checked sets: max answer diff {worst:.2e}, "
              f"min prediction agreement {agree:.4f}")
    (OUT / f"test_dish{dish}_w{width}_k{k}_{backend}.json").write_text(json.dumps(
        {"env": env(), "results": res, "acceptance": [(c, bool(o), v) for c, o, v in checks],
         "config": {"dish": dish, "width": width, "knn_k": k, "facet": FACET, "receiver": RECEIVER,
                    "rays": RAYS, "core": CORE}}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["sweep", "test"])
    ap.add_argument("--dish", type=float)
    ap.add_argument("--width", type=float)
    ap.add_argument("--knn", type=int)
    ap.add_argument("--backend", choices=("cpu", "rtx"), default="cpu",
                    help="rtx: hardware traces, re-checked against the CPU reference")
    a = ap.parse_args()
    sweep() if a.split == "sweep" else test(a.dish, a.width, a.knn, a.backend)


if __name__ == "__main__":
    main()
