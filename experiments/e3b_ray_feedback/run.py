"""E3b (#40): learn the station layout from ray-traced prediction error on RTX.

    python experiments/e3b_ray_feedback/run.py val            # learning curves on validation
    python experiments/e3b_ray_feedback/run.py test --rounds N  # sealed test, once

Each round: rebuild the scene at the current layout (every dish re-aimed at its
object's current position), fire an annular beam for every written fact (its own
dish is inside the hole: physical leave-one-out), and credit the stations whose
dishes reflected rays into the *correct* receiver with an attracting spring.
Springs + inverse-square repulsion move the stations. The only learning signal
is which traced rays landed on the right receiver. "shuffled" permutes the
correct/incorrect flags among rays that hit a dish (same rays, no error signal).
Evaluation uses the frozen E3 inference configuration (dish 0.3, width 3.0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "rtx"))
sys.path.insert(0, str(ROOT / "src"))
from beam import BeamRTX, open_kernel  # noqa: E402

from rtxllm.core.cluster_world import ClusterConfig, generate_clustered  # noqa: E402
from rtxllm.core.scene import KIND_FACT, GeometryConfig, build_scene  # noqa: E402
from rtxllm.cpu.beam import BeamParams, BeamTracer  # noqa: E402
from rtxllm.learning.placement import layout_quality, normalise, random_lattice  # noqa: E402

SEEDS = (0, 1, 2)
GEOM = GeometryConfig(receiver_radius=0.075, fact_radius=0.05, dish_radius=0.3)
EVAL = BeamParams(width=3.0, core=0.5, rays=256, fact_distance=GEOM.fact_distance)
INNER = 0.4                      # > dish radius 0.3: the fact's own dish is never hit
HP = {"learn_rays": 128, "w_start": 6.0, "w_end": 3.0, "steps": 10, "decay": 0.9}   # defaults; CLI overrides
OUT = Path(__file__).resolve().parent / "results"


class Learner:
    def __init__(self, device, kernel, world, seed, mode):
        self.device, self.kernel, self.world, self.mode = device, kernel, world, mode
        self.S = world.n_stations
        self.P = random_lattice(self.S, seed).astype(np.float64)
        self.W = np.zeros((self.S, self.S))
        self.rng = np.random.default_rng(10_000 + seed)
        self.keys = sorted(world.written)
        self.src = np.array([s for s, _ in self.keys])
        self.rel = np.array([r for _, r in self.keys])
        self.obj = np.array([world.written[k][0] for k in self.keys])
        self.stats = []

    def scene(self):
        return build_scene(self.world.with_positions(self.P.astype(np.float32)), GEOM)

    def round(self, width):
        t0 = time.perf_counter()
        sc = self.scene()
        n_rec = int((sc.tri_kind != KIND_FACT).sum())
        F = (sc.n_triangles - n_rec) // len(self.keys)
        assert n_rec + F * len(self.keys) == sc.n_triangles
        cpu = BeamTracer(sc, BeamParams(width=width, core=0.0, inner=INNER, rays=HP["learn_rays"],
                                        fact_distance=GEOM.fact_distance))
        gpu = BeamRTX(self.device, self.kernel, cpu)
        land, ms, hit = gpu.dispatch(self.src, self.rel, with_hits=True)
        land, hit = land[:, 1:], hit[:, 1:]                     # drop the zero-weight centre ray
        met = hit >= 0
        neigh = np.where(met, self.src[np.maximum((hit - n_rec) // F, 0)], -1)
        correct = met & (land == self.obj[:, None])
        if self.mode == "shuffled":                           # same rays, scrambled error signal
            flat = correct[met]
            correct = np.zeros_like(correct)
            correct[met] = self.rng.permutation(flat)
        bi, ki = np.nonzero(correct)
        A = np.zeros((self.S, self.S))
        np.add.at(A, (self.src[bi], neigh[bi, ki]), 1.0)
        A = A + A.T
        np.fill_diagonal(A, 0)
        self.W = HP["decay"] * self.W + A
        Wn = self.W / max(1.0, self.W.sum(1).mean())
        P = self.P
        for _ in range(HP["steps"]):
            D = P[:, None, :] - P[None, :, :]
            dist2 = (D ** 2).sum(-1) + 1e-2
            f = (D / dist2[..., None] ** 1.5).sum(1) * 2.0 - (Wn[..., None] * D).sum(1)
            n = np.linalg.norm(f, axis=1, keepdims=True)
            P = P + 0.5 * f / np.maximum(1.0, n)
        self.P = normalise(P).astype(np.float64)
        self.stats.append({"width": width, "rays": int(land.size), "rays_met_dish": int(met.sum()),
                           "rays_correct": int(correct.sum()), "gpu_ms": ms,
                           "round_wall_s": time.perf_counter() - t0, "triangles": sc.n_triangles})

    def evaluate(self, split):
        sc = self.scene()
        gpu = BeamRTX(self.device, self.kernel, BeamTracer(sc, EVAL))
        w = self.world
        held = w.heldout[split]
        r = gpu.trace([s for s, _ in held], [(x,) for _, x in held])
        gold = np.array([w.truth[k] for k in held])
        th = w.two_hop[split]
        r2 = gpu.trace([q[0] for q in th], [(q[1], q[2]) for q in th])
        keys = sorted(w.written)
        ws = [keys[i] for i in sorted(np.random.default_rng(w.config.seed).choice(len(keys), 300, replace=False))]
        rw = gpu.trace([s for s, _ in ws], [(x,) for _, x in ws])
        return {"heldout_acc": float(np.mean(r.predict() == gold)),
                "two_hop_acc": float(np.mean(r2.predict() == np.array([q[3] for q in th]))),
                "written_acc": float(np.mean(rw.predict() == np.array([w.written[k][0] for k in ws]))),
                "layout_quality": layout_quality(self.P.astype(np.float32), w.cluster)}


def run(split, rounds, checkpoints, stop=None):
    """``rounds`` sets the annealing schedule; training stops after ``stop`` rounds (default all)."""
    device, kernel = open_kernel("beam.slang")
    out = {}
    for seed in SEEDS:
        world = generate_clustered(ClusterConfig(seed=seed))
        out[seed] = {}
        for mode in ("feedback", "shuffled"):
            L = Learner(device, kernel, world, seed, mode)
            curve = {0: L.evaluate(split)}
            for t in range(1, (stop or rounds) + 1):
                width = HP["w_start"] + (HP["w_end"] - HP["w_start"]) * (t - 1) / max(1, rounds - 1)
                L.round(width)
                if t in checkpoints:
                    curve[t] = L.evaluate(split)
                    print(f"seed{seed} {mode} round {t}: {curve[t]}", flush=True)
            out[seed][mode] = {"curve": curve, "rounds": L.stats}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("val", "test"))
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--decay", type=float, default=HP["decay"])
    ap.add_argument("--w-start", type=float, default=HP["w_start"])
    ap.add_argument("--w-end", type=float, default=HP["w_end"])
    ap.add_argument("--learn-rays", type=int, default=HP["learn_rays"])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--eval-round", type=int, default=None,
                    help="test: evaluate at this round of a --rounds-long annealing schedule")
    a = ap.parse_args()
    HP.update(decay=a.decay, w_start=a.w_start, w_end=a.w_end, learn_rays=a.learn_rays)
    OUT.mkdir(exist_ok=True)
    if a.split == "val":
        cps = {r for r in (5, 10, 20, 30, 40, 60, 80, 100) if r <= a.rounds} | {a.rounds}
        res = run("val", a.rounds, cps)
        (OUT / f"val_curves_{a.tag}.json").write_text(json.dumps({"hp": HP, "rounds": a.rounds, "results": res},
                                                                 indent=1, default=float))
    else:
        er = a.eval_round or a.rounds
        res = run("test", a.rounds, {er}, stop=er)
        checks = []
        for seed, r in res.items():
            fb, sh = r["feedback"]["curve"][er], r["shuffled"]["curve"][er]
            start = r["feedback"]["curve"][0]
            checks += [(f"seed{seed} held >= 0.6", fb["heldout_acc"] >= 0.6, fb["heldout_acc"]),
                       (f"seed{seed} lift over random start >= 0.4", fb["heldout_acc"] - start["heldout_acc"] >= 0.4,
                        fb["heldout_acc"] - start["heldout_acc"]),
                       (f"seed{seed} lift over shuffled >= 0.4", fb["heldout_acc"] - sh["heldout_acc"] >= 0.4,
                        fb["heldout_acc"] - sh["heldout_acc"]),
                       (f"seed{seed} 2hop >= 0.4", fb["two_hop_acc"] >= 0.4, fb["two_hop_acc"])]
        for c in checks:
            print(("PASS " if c[1] else "FAIL ") + c[0], round(c[2], 4))
        (OUT / f"test_sched{a.rounds}_eval{er}_{a.tag}.json").write_text(json.dumps(
            {"hp": HP, "schedule_rounds": a.rounds, "eval_round": er, "results": res, "acceptance": [(c, bool(o), float(v)) for c, o, v in checks]}, indent=1, default=float))


if __name__ == "__main__":
    main()
