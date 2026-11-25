"""#59 (Claude): learn the word layout from ray-traced next-word prediction error.

    python experiments/e12b_word_ray_feedback/run.py val  BOOK_DIR [--rounds N --w-start A --w-end B]
    python experiments/e12b_word_ray_feedback/run.py test BOOK_DIR --rounds N --eval-round R ...

Start from a random layout of the E12 dish-owning words. Each round: rebuild the
dish scene at the current layout, fire an annular beam on RTX from every word
(never its own dish), and credit each ray that hit neighbour w' with
sum_v c(w, v) [P_w'(v) - P_uni(v)]_+ / c(w): how much better than chance the
neighbour's stored next-word distribution predicts w's actual training
successors. Credits accumulate as springs; inverse-square repulsion; rescale.
"shuffled" permutes credits among the rays that hit a dish. Readout = frozen E12
(sigma 1.0, lambda 0.1). Options (selected on validation): --credit llr (mean
log-likelihood ratio vs chance, centred per beam) and --topk K (springs use only
each word's K strongest accumulated ray credits, as the E12 force layout used a
10-nearest graph; the credits themselves come only from ray hits).
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
sys.path.insert(0, str(ROOT / "experiments" / "e12_word_layout"))
import run as e12  # noqa: E402
from word_beam import WordRTX, beam_origins, open_word_kernel  # noqa: E402

from rtxllm.learning.placement import normalise, random_lattice  # noqa: E402

SIGMA, LAM = 1.0, 0.1                 # frozen E12 readout
LEARN_RAYS = 256
OUT = Path(__file__).resolve().parent / "results"


class WordLearner:
    def __init__(self, device, kernel, m, ids, seed, mode, credit="gain", topk=0):
        self.device, self.kernel, self.m, self.ids, self.mode, self.topk = device, kernel, m, ids, mode, topk
        self.n = len(ids)
        self.X = random_lattice(self.n, seed).astype(np.float64)
        self.W = np.zeros((self.n, self.n))
        self.rng = np.random.default_rng(20_000 + seed)
        C = m.B[ids]                                              # successor counts of each dish word
        uni = m.B.sum(0) / m.B.sum()
        self.centre = credit == "llr"
        if credit == "gain":
            G = np.maximum(m.Pml[ids] - uni[None], 0)             # neighbour memory's gain over chance
            self.credit = (C @ G.T) / np.maximum(m.rowtot[ids], 1)[:, None]
        else:
            # mean log-likelihood ratio of w's actual successors under the neighbour's memory vs chance
            a = 0.1
            Lr = np.log((m.Pml[ids] + a * uni[None]) / ((1 + a) * uni[None] + 1e-300))
            self.credit = (C @ Lr.T) / np.maximum(m.rowtot[ids], 1)[:, None]
        np.fill_diagonal(self.credit, 0)
        self.stats = []

    def round(self, width):
        t0 = time.perf_counter()
        pos, radii, sc = e12.scene_for(self.m, self.ids, self.X)
        gpu = WordRTX(self.device, self.kernel, sc)
        o = beam_origins(pos, self.ids, radii, e12.D, width, LEARN_RAYS, back=width)
        hit, ms = gpu.hits(o, e12.D, width * 2 + e12.FACT_DISTANCE + 0.5)
        hit = hit.reshape(self.n, LEARN_RAYS)
        bi, ki = np.nonzero(hit >= 0)
        nj = hit[bi, ki]
        cr = self.credit[bi, nj]
        if self.centre:                                           # relative to this beam's other hits
            mean = np.bincount(bi, weights=cr, minlength=self.n) / np.maximum(np.bincount(bi, minlength=self.n), 1)
            cr = np.maximum(cr - mean[bi], 0)
        if self.mode == "shuffled":
            cr = self.rng.permutation(cr)
        A = np.zeros((self.n, self.n))
        np.add.at(A, (bi, nj), cr)
        A = A + A.T
        self.W += A
        Ws = self.W
        if self.topk:                                             # keep each word's k strongest ray-credit springs
            nn = np.argpartition(-Ws, self.topk, 1)[:, :self.topk]
            Ws = np.zeros_like(self.W)
            rows = np.arange(self.n)[:, None]
            Ws[rows, nn] = self.W[rows, nn]
            Ws = np.maximum(Ws, Ws.T)
        Wn = Ws / max(1e-12, Ws.sum(1).mean())
        X = self.X
        for _ in range(10):
            Dd = X[:, None] - X[None]
            d2 = (Dd ** 2).sum(-1) + 1e-2
            f = (Dd / d2[..., None] ** 1.5).sum(1) * 2.0 - (Wn[..., None] * Dd).sum(1)
            X = X + 0.5 * f / np.maximum(1.0, np.linalg.norm(f, axis=1, keepdims=True))
        self.X = normalise(X).astype(np.float64)
        self.stats.append({"width": width, "rays": int(hit.size), "rays_hit": int(len(bi)),
                           "credit_sum": float(cr.sum()), "gpu_ms": ms, "wall_s": time.perf_counter() - t0})

    def evaluate(self, prev, nxt):
        pos, radii, sc = e12.scene_for(self.m, self.ids, self.X)
        gpu = WordRTX(self.device, self.kernel, sc)
        q, has, _, _ = e12.beam_q(self.m, self.ids, pos, radii, gpu.hits, SIGMA, 2.5 * SIGMA)
        return e12.score(self.m, self.ids, q, has, LAM, prev, nxt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("val", "test"))
    ap.add_argument("books", type=Path)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--w-start", type=float, default=6.0)
    ap.add_argument("--w-end", type=float, default=2.5)
    ap.add_argument("--eval-round", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--credit", choices=("gain", "llr"), default="gain")
    ap.add_argument("--topk", type=int, default=0, help="sparsify accumulated ray-credit springs to k per word")
    a = ap.parse_args()
    tr, va, te = e12.tokens(a.books)
    m = e12.Model(tr)
    ev = m.enc(va if a.split == "val" else te)
    prev, nxt = ev[:-1], ev[1:]
    ids, Xf = e12.force_word_layout(m)
    device, kernel = open_word_kernel()
    cps = ({r for r in (5, 10, 20, 30, 40, 60, 80, 120) if r <= a.rounds} | {a.rounds}) if a.split == "val" \
        else {a.eval_round or a.rounds}
    stop = max(cps)
    out = {"hp": vars(a) | {"books": str(a.books)}, "kn": e12.score(m, ids, np.zeros((len(ids), m.V)),
                                                                    np.zeros(len(ids), bool), 0.0, prev, nxt)}
    for mode in ("feedback", "shuffled"):
        L = WordLearner(device, kernel, m, ids, a.seed, mode, a.credit, a.topk)
        curve = {0: L.evaluate(prev, nxt)}
        for t in range(1, stop + 1):
            width = a.w_start + (a.w_end - a.w_start) * (t - 1) / max(1, a.rounds - 1)
            L.round(width)
            if t in cps:
                curve[t] = L.evaluate(prev, nxt)
                print(f"{mode} round {t}: {json.dumps({k: round(v, 3) for k, v in curve[t].items()})}", flush=True)
        out[mode] = {"curve": curve, "rounds": L.stats}
    OUT.mkdir(exist_ok=True)
    tag = f"{a.split}_{a.tag}" + (f"_eval{a.eval_round}" if a.eval_round else "")
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("KN:", out["kn"])


if __name__ == "__main__":
    main()
