"""#53 (Claude): word-level spatial class LM — beams over a learned word layout vs KN bigrams.

    python experiments/e12_word_layout/run.py val  BOOK_DIR     # selection on validation
    python experiments/e12_word_layout/run.py test BOOK_DIR --sigma S --lam L   # sealed test, once

Pride and Prejudice (#10 pin), lowercase tokens [a-z0-9]+|[^\\sa-z0-9], contiguous
70/15/15, vocabulary = training words seen >= 2 times (+ UNK). Everything below
(layout, dishes, payloads, KN) is fit on the training split only.

P(v | w) = (1 - lam) * KN_bigram(v | w) + lam * q_w(v), where q_w is read by an
annular beam fired on RTX from w's station: each ray's first dish hit contributes
that neighbour word's next-word distribution, weighted by a Gaussian beam profile
of width sigma; dishes have area proportional to evidence. Controls: random
layout, kNN in the same layout, analytic Gaussian kernel (no occlusion), KN alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "rtx"))
sys.path.insert(0, str(ROOT / "src"))
from word_beam import WordRTX, WordScene, beam_origins, hex_dishes, open_word_kernel  # noqa: E402

from rtxllm.learning.placement import normalise, random_lattice  # noqa: E402

SHA = "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163"
MIN_DISH = 5            # words need >= 5 training continuations to own a dish
RAYS = 512
DISH_R0 = 0.12          # dish radius at median evidence; radius ~ sqrt(count)
FACT_DISTANCE = 1.5
D = np.array([0.36, 0.48, 0.8]) / np.linalg.norm([0.36, 0.48, 0.8])
OUT = Path(__file__).resolve().parent / "results"


def tokens(folder):
    raw = (folder / "rtxllm-pride-1342.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError("pinned Pride edition required")
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1].lower()
    t = re.findall(rb"[a-z0-9]+|[^\sa-z0-9]", body)
    n = len(t)
    return t[: n * 70 // 100], t[n * 70 // 100: n * 85 // 100], t[n * 85 // 100:]


class Model:
    def __init__(self, train):
        cnt = Counter(train)
        self.vocab = [w for w, c in cnt.items() if c >= 2]
        self.idx = {w: i + 1 for i, w in enumerate(self.vocab)}
        self.V = len(self.vocab) + 1
        a = self.enc(train)
        V = self.V
        B = np.zeros((V, V))
        np.add.at(B, (a[:-1], a[1:]), 1)
        self.B = B
        self.rowtot = B.sum(1)
        # interpolated modified Kneser-Ney bigram (continuation unigram base)
        n1, n2, n3, n4 = ((B == k).sum() for k in (1, 2, 3, 4))
        Y = n1 / (n1 + 2 * n2)
        self.Dk = np.array([0, 1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3])
        cont = (B > 0).sum(0).astype(float)
        self.pcont = (cont + 1e-9) / (cont + 1e-9).sum()
        self.N = [((B == 1).sum(1)), ((B == 2).sum(1)), ((B >= 3).sum(1))]
        self.Pml = np.divide(B, self.rowtot[:, None], out=np.zeros_like(B), where=self.rowtot[:, None] > 0)

    def enc(self, ts):
        return np.array([self.idx.get(t, 0) for t in ts])

    def kn(self, prev, nxt):
        c = self.B[prev, nxt]
        t = self.rowtot[prev]
        Dc = self.Dk[np.minimum(c, 3).astype(int)]
        gam = (self.Dk[1] * self.N[0][prev] + self.Dk[2] * self.N[1][prev] + self.Dk[3] * self.N[2][prev]) / np.maximum(t, 1)
        return np.where(t > 0, np.maximum(c - Dc, 0) / np.maximum(t, 1) + gam * self.pcont[nxt], self.pcont[nxt])


def force_word_layout(m: Model, seed=0, k=10, iters=300):
    ids = np.flatnonzero((m.rowtot >= MIN_DISH) & (np.arange(m.V) > 0))
    F = np.sqrt(m.Pml[ids])
    sim = F @ F.T
    np.fill_diagonal(sim, 0)
    W = np.zeros_like(sim)
    nn = np.argsort(-sim, 1)[:, :k]
    rows = np.arange(len(ids))[:, None]
    W[rows, nn] = sim[rows, nn]
    W = np.maximum(W, W.T)
    W /= W.sum(1).mean()
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(len(ids), 3)) * 5
    step = 0.5
    for _ in range(iters):
        Dd = X[:, None] - X[None]
        d2 = (Dd ** 2).sum(-1) + 1e-2
        f = (Dd / d2[..., None] ** 1.5).sum(1) * 2.0 - (W[..., None] * Dd).sum(1)
        X += step * f / np.maximum(1, np.linalg.norm(f, axis=1, keepdims=True))
        step *= 0.995
    return ids, normalise(X).astype(np.float64)


def scene_for(m: Model, ids, X):
    pos = np.zeros((m.V, 3))
    pos[ids] = X
    radii = DISH_R0 * np.sqrt(m.rowtot[ids] / np.median(m.rowtot[ids]))
    centres = pos[ids] + FACT_DISTANCE * D
    return pos, radii, WordScene(hex_dishes(centres, radii, D))


def beam_q(m: Model, ids, pos, radii, hits_fn, sigma, width):
    """q_w for every dish-owning word w (rows indexed like ids); None rows -> no hits."""
    origins = beam_origins(pos, ids, radii, D, width, RAYS, back=width)
    hit, gpu_ms = hits_fn(origins, D, width * 2 + FACT_DISTANCE + 0.5)
    hit = hit.reshape(len(ids), RAYS)
    # ray radial offsets (same annulus construction) -> Gaussian beam profile weights
    rr = np.stack([np.sqrt((ri * 1.05) ** 2 + (max(width, ri * 1.1) ** 2 - (ri * 1.05) ** 2) * (np.arange(RAYS) + .5) / RAYS)
                   for ri in radii])
    w = np.exp(-rr ** 2 / (2 * sigma ** 2)) * (hit >= 0)
    q = np.zeros((len(ids), m.V))
    for k in range(len(ids)):
        h = hit[k] >= 0
        if not h.any():
            continue
        dish_word = ids[hit[k][h]]
        q[k] = (w[k][h][:, None] * m.Pml[dish_word]).sum(0)
    tot = q.sum(1, keepdims=True)
    q = np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)
    return q, tot[:, 0] > 0, hit, gpu_ms


def knn_q(m, ids, X, k):
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, 1)[:, :k]
    q = (m.rowtot[ids][nn][..., None] * m.Pml[ids][nn]).sum(1)
    return q / q.sum(1, keepdims=True)


def kernel_q(m, ids, X, sigma):
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * sigma ** 2)) * m.rowtot[ids][None]
    np.fill_diagonal(K, 0)
    q = K @ m.Pml[ids]
    return q / np.maximum(q.sum(1, keepdims=True), 1e-300)


def score(m, ids, q, has, lam, prev, nxt):
    p = m.kn(prev, nxt)
    row = np.full(m.V, -1)
    row[ids] = np.arange(len(ids))
    r = row[prev]
    use = (r >= 0)
    use[use] = has[r[use]]
    qq = np.where(use, q[np.maximum(r, 0), nxt], 0)
    p = np.where(use, (1 - lam) * p + lam * qq, p)
    unseen = m.B[prev, nxt] == 0
    lp = -np.log2(p)
    return {"ppl": float(2 ** lp.mean()), "ppl_unseen": float(2 ** lp[unseen].mean()),
            "ppl_seen": float(2 ** lp[~unseen].mean()), "unseen_frac": float(unseen.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("val", "test"))
    ap.add_argument("books", type=Path)
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--knn", type=int)
    a = ap.parse_args()
    tr, va, te = tokens(a.books)
    m = Model(tr)
    ev = m.enc(va if a.split == "val" else te)
    prev, nxt = ev[:-1], ev[1:]
    device, kernel = open_word_kernel()
    ids, Xf = force_word_layout(m)
    Xr = random_lattice(len(ids), 0).astype(np.float64)
    res = {"vocab": m.V, "dish_words": len(ids), "train_tokens": len(tr), "eval_tokens": len(ev),
           "kn": score(m, ids, np.zeros((len(ids), m.V)), np.zeros(len(ids), bool), 0.0, prev, nxt),
           "env": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}}
    print("KN bigram", res["kn"], flush=True)
    grid = [(a.sigma, a.lam)] if a.split == "test" else [(s, l) for s in (0.5, 1.0, 2.0) for l in (0.1, 0.2, 0.3)]
    kgrid = [a.knn] if a.split == "test" else [5, 10, 20, 40]
    for name, X in (("force", Xf), ("random", Xr)):
        pos, radii, sc = scene_for(m, ids, X)
        gpu = WordRTX(device, kernel, sc)
        for sigma, lam in grid:
            width = 2.5 * sigma
            t0 = perf_counter()
            q, has, hit, gpu_ms = beam_q(m, ids, pos, radii, gpu.hits, sigma, width)
            wall = perf_counter() - t0
            # CPU parity on the first 64 words' rays
            o = beam_origins(pos, ids[:64], radii[:64], D, width, RAYS, back=width)
            cpu = sc.cpu_hits(o, D, width * 2 + FACT_DISTANCE + 0.5)
            agree = float(np.mean(cpu == hit[:64].reshape(-1)))
            s = score(m, ids, q, has, lam, prev, nxt)
            key = f"beam_{name}_s{sigma}_l{lam}"
            res[key] = {**s, "sigma": sigma, "lam": lam, "width": width, "words_with_hits": float(has.mean()),
                        "rays": int(hit.size), "ray_hit_frac": float((hit >= 0).mean()), "gpu_ms": gpu_ms,
                        "host_s": wall, "cpu_parity_rays": int(o.shape[0]), "cpu_parity_agreement": agree,
                        "as_bytes": gpu.as_bytes}
            print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[key].items()}), flush=True)
        for kk in kgrid:
            for lam in (sorted({l for _, l in grid})):
                s = score(m, ids, knn_q(m, ids, X, kk), np.ones(len(ids), bool), lam, prev, nxt)
                res[f"knn_{name}_k{kk}_l{lam}"] = s
                print(f"knn_{name}_k{kk}_l{lam}", s, flush=True)
        for sigma, lam in grid:
            s = score(m, ids, kernel_q(m, ids, X, sigma), np.ones(len(ids), bool), lam, prev, nxt)
            res[f"kernel_{name}_s{sigma}_l{lam}"] = s
    OUT.mkdir(exist_ok=True)
    tag = "val" if a.split == "val" else f"test_s{a.sigma}_l{a.lam}_k{a.knn}"
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
