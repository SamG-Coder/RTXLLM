"""#83 (Claude): E12c — full-disc word beam vs matched self-inclusive smoothers.

    python experiments/e12c_full_beam/run.py val  BOOK_DIR
    python experiments/e12c_full_beam/run.py test BOOK_DIR      # frozen settings below, run once

Same data, vocabulary, layout, dishes and KN as E12 (#53). The only change to the
beam is that rays cover the full disc (inner radius 0), so the word's own dish can be
hit (ADR 0003 class T: which dishes are hit is decided by traversal over the learned
layout, with area proportional to evidence and nearest-hit occlusion). Every
comparator is allowed to include the word itself and is tuned on validation:
class bigram on high-D features, class bigram on the 3-D layout, Gaussian kernel with
self, kNN with self. Interventions (ADR 0003 b): the same beam over a random layout,
and the same rays with the hit-to-payload map permuted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "e12_word_layout"))
import posthoc_controls as pc  # noqa: E402
import run as e12  # noqa: E402
from word_beam import WordRTX, annulus, basis, open_word_kernel  # noqa: E402

from rtxllm.learning.placement import random_lattice  # noqa: E402

LAMS = (0.05, 0.1, 0.2, 0.3)
SIGMAS = (0.5, 1.0, 2.0)
KS = (25, 50, 100, 200, 400)
KNNS = (5, 10, 20, 40)
OUT = Path(__file__).resolve().parent / "results"


def disc_origins(pos, ids, width, rays):
    t1, t2 = basis(e12.D)
    o2 = annulus(rays, 0.0, width)
    O = pos[ids][:, None] - width * e12.D + o2[None, :, :1] * t1 + o2[None, :, 1:] * t2
    return O.reshape(-1, 3).astype(np.float32), np.hypot(o2[:, 0], o2[:, 1])


def disc_beam(m, ids, pos, gpu, scene, sigma, payload_rows=None, parity_words=64):
    """q for every dish word from a full-disc beam traced on RTX; CPU parity on the first words."""
    width = 2.5 * sigma
    O, rr = disc_origins(pos, ids, width, e12.RAYS)
    tmax = 2 * width + e12.FACT_DISTANCE + 0.5
    hit, ms = gpu.hits(O, e12.D, tmax)
    cpu = scene.cpu_hits(O[: parity_words * e12.RAYS], e12.D, tmax)
    parity = float(np.mean(cpu == hit[: parity_words * e12.RAYS]))
    hit = hit.reshape(len(ids), e12.RAYS)
    rows = np.arange(len(ids)) if payload_rows is None else payload_rows
    w = np.exp(-rr ** 2 / (2 * sigma ** 2))[None] * (hit >= 0)
    q = np.zeros((len(ids), m.V))
    for k in range(len(ids)):
        h = hit[k] >= 0
        if h.any():
            q[k] = (w[k][h][:, None] * m.Pml[ids[rows[hit[k][h]]]]).sum(0)
    tot = q.sum(1, keepdims=True)
    q = np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)
    stats = {"own_hit_frac": float((hit == np.arange(len(ids))[:, None]).mean()),
             "ray_hit_frac": float((hit >= 0).mean()), "rays": int(hit.size), "gpu_ms": ms,
             "cpu_parity_rays": parity_words * e12.RAYS, "cpu_parity": parity}
    return q, tot[:, 0] > 0, stats


def kernel_self(m, ids, X, sigma):
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    q = (np.exp(-d2 / (2 * sigma ** 2)) * m.rowtot[ids][None]) @ m.Pml[ids]
    return q / q.sum(1, keepdims=True)


def knn_self(m, ids, X, k):
    nn = np.argsort(((X[:, None] - X[None]) ** 2).sum(-1), 1)[:, :k]   # includes self (distance 0)
    q = (m.rowtot[ids][nn][..., None] * m.Pml[ids][nn]).sum(1)
    return q / q.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("val", "test"))
    ap.add_argument("books", type=Path)
    a = ap.parse_args()
    tr, va, te = e12.tokens(a.books)
    m = e12.Model(tr)
    ev = m.enc(va if a.split == "val" else te)
    prev, nxt = ev[:-1], ev[1:]
    ids, Xf = e12.force_word_layout(m)
    Xr = random_lattice(len(ids), 0).astype(np.float64)
    dev, ker = open_word_kernel()
    allw = np.ones(len(ids), bool)
    res = {"kn": e12.score(m, ids, np.zeros((len(ids), m.V)), np.zeros(len(ids), bool), 0.0, prev, nxt)}

    def sc(q, has, lam):
        return e12.score(m, ids, q, has, lam, prev, nxt)

    pos, radii, scene = e12.scene_for(m, ids, Xf)
    gpu = WordRTX(dev, ker, scene)
    if a.split == "val":
        frozen = {}
        grids = {}
        for sigma in SIGMAS:
            q, has, st = disc_beam(m, ids, pos, gpu, scene, sigma)
            for lam in LAMS:
                grids[f"beam_s{sigma}_l{lam}"] = sc(q, has, lam) | st
        for name, feats in (("class_highD", np.sqrt(m.Pml[ids])), ("class_3D", Xf)):
            for K in KS:
                q = pc.class_q(m, ids, pc.kmeans(feats, K))
                for lam in LAMS:
                    grids[f"{name}_K{K}_l{lam}"] = sc(q, allw, lam)
        for sigma in SIGMAS:
            q = kernel_self(m, ids, Xf, sigma)
            for lam in LAMS:
                grids[f"kernel_self_s{sigma}_l{lam}"] = sc(q, allw, lam)
        for k in KNNS:
            q = knn_self(m, ids, Xf, k)
            for lam in LAMS:
                grids[f"knn_self_k{k}_l{lam}"] = sc(q, allw, lam)
        for fam in ("beam", "class_highD", "class_3D", "kernel_self", "knn_self"):
            best = min((k for k in grids if k.startswith(fam + "_")), key=lambda k: grids[k]["ppl"])
            frozen[fam] = best
            print(f"{fam}: {best} ppl {grids[best]['ppl']:.3f} unseen {grids[best]['ppl_unseen']:.0f}", flush=True)
        res |= {"grid": grids, "selected": frozen}
    else:
        sel = json.loads((OUT / "val.json").read_text(encoding="utf-8"))["selected"]

        def parse(key):
            parts = dict((p[0], float(p[1:])) for p in key.split("_")[-2:])
            return parts

        pb = parse(sel["beam"])
        sigma, lam = pb["s"], pb["l"]
        q, has, st = disc_beam(m, ids, pos, gpu, scene, sigma)
        res["beam_full_disc"] = sc(q, has, lam) | st | {"setting": sel["beam"]}
        perm = np.random.default_rng(83).permutation(len(ids))
        q2, has2, st2 = disc_beam(m, ids, pos, gpu, scene, sigma, payload_rows=perm)
        res["beam_shuffled_payload"] = sc(q2, has2, lam) | st2
        posr, _, scr = e12.scene_for(m, ids, Xr)
        q3, has3, st3 = disc_beam(m, ids, posr, WordRTX(dev, ker, scr), scr, sigma)
        res["beam_random_layout"] = sc(q3, has3, lam) | st3
        for fam, feats in (("class_highD", np.sqrt(m.Pml[ids])), ("class_3D", Xf)):
            p = parse(sel[fam])
            res[fam] = sc(pc.class_q(m, ids, pc.kmeans(feats, int(p["K"]))), allw, p["l"]) | {"setting": sel[fam]}
        p = parse(sel["kernel_self"])
        res["kernel_self"] = sc(kernel_self(m, ids, Xf, p["s"]), allw, p["l"]) | {"setting": sel["kernel_self"]}
        p = parse(sel["knn_self"])
        res["knn_self"] = sc(knn_self(m, ids, Xf, int(p["k"])), allw, p["l"]) | {"setting": sel["knn_self"]}
        b = res["beam_full_disc"]["ppl"]
        comps = {k: res[k]["ppl"] for k in ("class_highD", "class_3D", "kernel_self", "knn_self")}
        res["criterion"] = {
            "beam_vs_best_comparator": 1 - b / min(comps.values()),
            "random_layout_worse_by": res["beam_random_layout"]["ppl"] / b - 1,
            "shuffled_payload_worse_by": res["beam_shuffled_payload"]["ppl"] / b - 1,
        }
        c = res["criterion"]
        res["passes"] = bool(c["beam_vs_best_comparator"] >= 0.005 and c["random_layout_worse_by"] >= 0.02
                             and c["shuffled_payload_worse_by"] >= 0.02)
        for k, v in res.items():
            print(k, json.dumps(v, default=float)[:300], flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{a.split}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
