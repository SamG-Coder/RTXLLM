"""#53 post-hoc controls requested in review (Codex, #57). The sealed E12 result is unchanged.

    python experiments/e12_word_layout/posthoc_controls.py BOOK_DIR

1. Discrete class bigram (train only): k-means over the same Hellinger next-word
   features used by the layout; P_class(v | w) = pooled training successor
   distribution of w's class; P = (1 - lam) KN + lam P_class. K and lam are
   selected on validation, then test is scored once with those values.
2. Pencil control (RTX): one ray per word aimed inside its *own* dish (offset from
   the hexagon's centre vertex), so the readout is the word's own stored bigram;
   P = (1 - lam) KN + lam q_own, at the frozen E12 lam = 0.1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as e12  # noqa: E402
from word_beam import WordRTX, basis, open_word_kernel  # noqa: E402


def sqdist(F, C):
    return (F ** 2).sum(1)[:, None] - 2 * F @ C.T + (C ** 2).sum(1)[None]


def kmeans(F, K, seed=0, iters=50):
    rng = np.random.default_rng(seed)
    C = F[[rng.integers(len(F))]]
    dmin = sqdist(F, C)[:, 0]
    for _ in range(1, K):                                  # k-means++ initialisation
        c = F[[rng.choice(len(F), p=np.maximum(dmin, 0) / np.maximum(dmin, 0).sum())]]
        C = np.vstack([C, c])
        dmin = np.minimum(dmin, sqdist(F, c)[:, 0])
    for _ in range(iters):
        lab = np.argmin(sqdist(F, C), 1)
        for k in range(K):
            if (lab == k).any():
                C[k] = F[lab == k].mean(0)
    return lab


def class_q(m, ids, lab):
    K = lab.max() + 1
    pooled = np.zeros((K, m.V))
    np.add.at(pooled, lab, m.B[ids])
    pooled /= np.maximum(pooled.sum(1, keepdims=True), 1)
    return pooled[lab]


def main():
    books = Path(sys.argv[1])
    tr, va, te = e12.tokens(books)
    m = e12.Model(tr)
    ids, Xf = e12.force_word_layout(m)
    F = np.sqrt(m.Pml[ids])
    vp, vn = m.enc(va)[:-1], m.enc(va)[1:]
    tp, tn = m.enc(te)[:-1], m.enc(te)[1:]
    out = {"note": "post-hoc controls added after the sealed E12 test; selection on validation only",
           "val_transitions": len(vp), "test_transitions": len(tp), "class_val": {}}
    labs = {}
    for K in (25, 50, 100, 200, 400):
        labs[K] = kmeans(F, K)
        q = class_q(m, ids, labs[K])
        for lam in (0.05, 0.1, 0.2, 0.3):
            s = e12.score(m, ids, q, np.ones(len(ids), bool), lam, vp, vn)
            out["class_val"][f"K{K}_l{lam}"] = s
            print(f"class K={K} lam={lam} val: ppl {s['ppl']:.3f} unseen {s['ppl_unseen']:.1f}", flush=True)
    best = min(out["class_val"], key=lambda k: out["class_val"][k]["ppl"])
    K, lam = int(best.split("_")[0][1:]), float(best.split("_l")[1])
    out["class_selected"] = {"K": K, "lam": lam}
    out["class_test"] = e12.score(m, ids, class_q(m, ids, labs[K]), np.ones(len(ids), bool), lam, tp, tn)
    print("class selected", K, lam, "test", out["class_test"], flush=True)

    # Pencil: one ray per word inside its own dish (RTX), frozen lam 0.1
    device, kernel = open_word_kernel()
    pos, radii, sc = e12.scene_for(m, ids, Xf)
    gpu = WordRTX(device, kernel, sc)
    t1, t2 = basis(e12.D)
    o = (pos[ids] - 1.0 * e12.D + (0.25 * radii)[:, None] * (0.6 * t1 + 0.3 * t2)).astype(np.float32)
    hit, ms = gpu.hits(o, e12.D, 1.0 + e12.FACT_DISTANCE + 0.5)
    cpu = sc.cpu_hits(o, e12.D, 1.0 + e12.FACT_DISTANCE + 0.5)
    own = hit == np.arange(len(ids))
    q_own = np.where(own[:, None], m.Pml[ids][np.maximum(hit, 0)], 0)
    out["pencil"] = {"own_dish_hit_frac": float(own.mean()), "cpu_parity": float(np.mean(hit == cpu)),
                     "val": e12.score(m, ids, q_own, own, e12_lam := 0.1, vp, vn),
                     "test": e12.score(m, ids, q_own, own, e12_lam, tp, tn)}
    print("pencil", json.dumps(out["pencil"]), flush=True)
    (HERE / "results" / "posthoc_controls.json").write_text(json.dumps(out, indent=1, default=float),
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
