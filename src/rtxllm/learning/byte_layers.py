"""Layered transmission geometry for physical backoff (issue #23), analytic form.

Geometry (analytic evaluator; no RTX realisation in this module):

* Each byte value b owns a cell ``g(b) = (gx, gy)`` of a 16x16 grid.
* A context of the last m bytes ``c = (b_{t-m} .. b_{t-1})`` owns the square
  dish ``[x_m(c), x_m(c) + 16^-m)^2`` on layer m, where
  ``x_m(c) = sum_{j=1..m} 16^-j g(b_{t-j})``. Layers are planes ordered along
  the beam from the most specific (m = k) to the empty context (m = 0, one dish
  covering the unit square). Coordinates are dyadic, so exact in float32 for
  k <= 5 (finest cell 2^-20).
* Dish reflectance R_c = 1 - gamma_KN(c); its payload is the KN-discounted
  continuation distribution. Light transmitted by every dish ends in a uniform
  backstop (1/256 per byte).
* A query is a beam along the layer normal through ``x_k(q) + (0.5, 0.25) 16^-k``
  (strictly inside every suffix cell, never on a cell edge or a triangle
  diagonal) plus optional offset rays.

With a single pencil ray this is *exactly* interpolated Kneser-Ney; offset rays
read dishes of contexts whose bytes have neighbouring grid cells.
"""

from __future__ import annotations

import numpy as np

from rtxllm.baselines.ngram import B, BU, U, KneserNey, _find

G = 16


def identity_grid():
    b = np.arange(B)
    return np.stack([b % G, b // G], 1).astype(np.int64)


def similarity_grid(train_docs, seed=0):
    """Learned byte->cell assignment from training data only: bytes with similar
    left/right neighbour distributions get nearby cells (PCA to 2-D, then a
    rank-based snake fill of the 16x16 grid)."""
    L = np.zeros((B, B))
    for d in train_docs:
        np.add.at(L, (d[1:], d[:-1]), 1)      # left neighbour of byte
    R = L.T.copy()                             # right neighbour
    F = np.hstack([L, R]) + 1e-3
    F = np.sqrt(F / F.sum(1, keepdims=True))   # Hellinger features
    F -= F.mean(0)
    u, s, vt = np.linalg.svd(F, full_matrices=False)
    xy = u[:, :2] * s[:2]
    order = np.argsort(xy[:, 0], kind="stable")
    g = np.zeros((B, 2), np.int64)
    for col in range(G):
        idx = order[col * G:(col + 1) * G]
        idx = idx[np.argsort(xy[idx, 1], kind="stable")]
        if col % 2:
            idx = idx[::-1]                    # snake keeps columns adjacent
        g[idx, 0] = col
        g[idx, 1] = np.arange(G)
    return g


def random_grid(seed):
    return identity_grid()[np.random.default_rng(seed).permutation(B)]


class LayeredBackoff:
    """Analytic evaluator of the layered dish geometry."""

    def __init__(self, kn: KneserNey, grid: np.ndarray):
        self.kn, self.g = kn, grid
        self.k = kn.order - 1
        inv = np.zeros((G, G), np.int64)
        inv[grid[:, 0], grid[:, 1]] = np.arange(B)
        self.cell_to_byte = inv

    def query_points(self, data, idx):
        """[n, 2] lateral coordinates of each scored position's pencil ray."""
        p = np.zeros((len(idx), 2))
        for j in range(1, self.k + 1):
            p += self.g[data[idx - j]] * float(G) ** -j
        return p + np.array([0.5, 0.25]) * float(G) ** -self.k

    def cell_context(self, pts, m):
        """(uint64 key of the last-m-byte context, oldest byte first, whose layer-m
        dish contains each point; inside mask for the unit square)."""
        inside = np.all((pts >= 0) & (pts < 1), axis=1)
        q = np.clip(pts, 0, 1 - 1e-12)
        key = np.zeros(len(pts), U)
        digits = []
        for j in range(1, m + 1):
            cell = np.floor(q * float(G) ** j).astype(np.int64) % G
            digits.append(self.cell_to_byte[cell[:, 0], cell[:, 1]].astype(U))
        for b in reversed(digits):            # oldest byte (largest j) first
            key = key * BU + b
        return key, inside

    def target_prob(self, pts, targets):
        """P(target) for rays at ``pts`` (one ray per row)."""
        p = np.full(len(pts), 1.0 / B)
        for m in range(0, self.k + 1):        # transmission chain, coarse to fine in recursion order
            t = self.kn.tables[m + 1]
            if not len(t["cu"]):
                continue
            ctx, inside = self.cell_context(pts, m)
            gram = ctx * BU + np.asarray(targets).astype(U)
            ci, present = _find(t["cu"], ctx)
            seen = inside & present
            tot = np.where(seen, t["tot"][ci], 1.0)
            gi, gpres = _find(t["gk"], gram)
            c = np.where(seen & gpres, t["gc"][gi] if len(t["gk"]) else 0, 0).astype(np.float64)
            D = t["D"][np.minimum(c, 3).astype(np.int64)]
            gamma = (t["D"][1] * t["N1"][ci] + t["D"][2] * t["N2"][ci] + t["D"][3] * t["N3"][ci]) / tot
            p = np.where(seen, np.maximum(c - D, 0) / tot + gamma * p, p)
        return p

    def beam_bits(self, data, start, offsets, weights):
        """Mean bits/byte with a beam of lateral ``offsets`` [r, 2] (in units of
        the finest cell, 16^-k) and ``weights`` [r] summing to 1."""
        idx = np.arange(max(start, self.k), len(data))
        base = self.query_points(data, idx)
        tgt = data[idx]
        p = np.zeros(len(idx))
        cell = float(G) ** -self.k
        for o, w in zip(offsets, weights):
            p += w * self.target_prob(base + np.asarray(o) * cell, tgt)
        return float(-np.log2(p).mean()), p


def ring_beam(radii, rays_per_ring, core):
    """Core ray + rings; ring energy shared equally. Offsets in finest-cell units."""
    offs, w = [(0.0, 0.0)], [core]
    ring_w = (1 - core) / max(1, len(radii))
    for r in radii:
        th = 2 * np.pi * (np.arange(rays_per_ring) + 0.5) / rays_per_ring
        offs += list(zip(r * np.cos(th), r * np.sin(th)))
        w += [ring_w / rays_per_ring] * rays_per_ring
    return np.array(offs), np.array(w)
