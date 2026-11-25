"""Station placement (where information lives in the scene).

All layouts are rescaled so the median nearest-neighbour distance is 1 scene
unit, making beam widths comparable across layouts.

* ``random_lattice``  — E1-style jittered lattice, random assignment (no structure).
* ``force_layout``    — learned only from *written* facts: stations that share
  (relation, object) pairs attract, all stations repel. Deterministic given seed.
* ``oracle_layout``   — uses latent cluster labels (upper bound; never selected).
"""

from __future__ import annotations

import numpy as np


def normalise(pos):
    pos = np.asarray(pos, np.float64)
    pos = pos - pos.mean(0)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return (pos / np.median(d.min(1))).astype(np.float32)


def random_lattice(S, seed):
    rng = np.random.default_rng(seed)
    g = int(np.ceil(S ** (1 / 3)))
    idx = np.stack(np.meshgrid(*(np.arange(g),) * 3, indexing="ij"), -1).reshape(-1, 3)[:S]
    pos = idx + rng.uniform(-0.15, 0.15, size=idx.shape)
    return normalise(pos[rng.permutation(S)])


def shared_fact_similarity(written, S):
    """A[i, j] = number of (relation, object) pairs both i and j have written."""
    by_ro = {}
    for (s, r), objs in written.items():
        for o in objs:
            by_ro.setdefault((r, o), []).append(s)
    A = np.zeros((S, S), np.float32)
    for members in by_ro.values():
        m = np.asarray(members)
        A[np.ix_(m, m)] += 1
    np.fill_diagonal(A, 0)
    return A


def force_layout(written, S, seed, iters=400, attract=1.0, repel=2.0, min_shared=2):
    """Spring layout: springs between stations sharing >= ``min_shared`` written
    (r, o) pairs; inverse-square repulsion between all stations."""
    rng = np.random.default_rng(seed)
    A = shared_fact_similarity(written, S)
    W = np.where(A >= min_shared, A, 0).astype(np.float64)
    W /= max(1.0, W.sum(1).mean())
    P = rng.normal(size=(S, 3)) * 5.0
    step = 0.5
    for it in range(iters):
        D = P[:, None, :] - P[None, :, :]
        dist2 = (D ** 2).sum(-1) + 1e-2
        f_rep = (D / dist2[..., None] ** 1.5).sum(1) * repel
        f_att = -(W[..., None] * D).sum(1) * attract
        F = f_att + f_rep
        n = np.linalg.norm(F, axis=1, keepdims=True)
        P += step * F / np.maximum(1.0, n)
        step *= 0.995
    return normalise(P)


def oracle_layout(cluster, seed, cluster_spacing=6.0):
    rng = np.random.default_rng(seed)
    C = int(cluster.max()) + 1
    g = int(np.ceil(C ** (1 / 3)))
    centres = np.stack(np.meshgrid(*(np.arange(g),) * 3, indexing="ij"), -1).reshape(-1, 3)[:C] * cluster_spacing
    pos = centres[cluster] + rng.normal(scale=0.6, size=(len(cluster), 3))
    return normalise(pos)


def layout_quality(pos, cluster):
    """Fraction of each station's 5 nearest neighbours that share its latent cluster."""
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :5]
    return float((cluster[nn] == cluster[:, None]).mean())
