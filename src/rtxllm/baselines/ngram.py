"""Interpolated modified Kneser-Ney byte n-gram baseline (Chen & Goodman 1998).

A strong count baseline for RTX byte models (issue #23). Vectorised with NumPy:
contexts of up to 7 bytes are packed into int64 keys, counts are found by
sorted-key search.

P_n(w | h) = max(c(h,w) - D_n(c), 0) / c(h) + gamma_n(h) * P_{n-1}(w | h')
gamma_n(h) = (D1 N1(h.) + D2 N2(h.) + D3 N3+(h.)) / c(h)
Highest order uses raw counts, lower orders continuation counts N1+(. h' w).
Unseen h backs off fully. Order 0 is uniform over 256 bytes.
"""

from __future__ import annotations

import numpy as np

B = 256


def _keys(data, n):
    """int64 keys of all n-grams (n bytes) in ``data``, aligned to the n-gram end."""
    k = np.zeros(len(data) - n + 1, np.int64)
    for j in range(n):
        k = k * B + data[j:len(data) - n + 1 + j]
    return k


def _count(keys):
    u, c = np.unique(keys, return_counts=True)
    return u, c


def _lookup(u, c, q):
    i = np.searchsorted(u, q)
    i = np.minimum(i, len(u) - 1)
    hit = u[i] == q
    return np.where(hit, c[i], 0)


class KneserNey:
    def __init__(self, order: int):
        assert 1 <= order <= 8
        self.order = order        # n-gram order (context length order-1)

    def fit(self, docs):
        n_max = self.order
        self.tables = {}
        # raw n-gram counts at every order, per document (no cross-boundary grams)
        raw = {n: np.concatenate([_keys(d, n) for d in docs if len(d) >= n]) for n in range(1, n_max + 1)}
        for n in range(1, n_max + 1):
            if n == n_max:
                gk, gc = _count(raw[n])
            else:
                # continuation count: number of distinct left extensions x of (h' w)
                u = np.unique(raw[n + 1])
                gk, gc = _count(u % (B ** n))
            # discounts from count-of-counts
            n1, n2, n3, n4 = (np.sum(gc == i) for i in (1, 2, 3, 4))
            Y = n1 / max(1, n1 + 2 * n2)
            D = np.array([0.0,
                          max(0.0, 1 - 2 * Y * n2 / max(1, n1)),
                          max(0.0, 2 - 3 * Y * n3 / max(1, n2)),
                          max(0.0, 3 - 4 * Y * n4 / max(1, n3))])
            ctx = gk // B
            # per-context totals and N1/N2/N3+ of its continuations
            cu, inv = np.unique(ctx, return_inverse=True)
            tot = np.bincount(inv, weights=gc).astype(np.float64)
            N1 = np.bincount(inv, weights=(gc == 1)).astype(np.float64)
            N2 = np.bincount(inv, weights=(gc == 2)).astype(np.float64)
            N3 = np.bincount(inv, weights=(gc >= 3)).astype(np.float64)
            self.tables[n] = dict(gk=gk, gc=gc, D=D, cu=cu, tot=tot, N1=N1, N2=N2, N3=N3)
        return self

    def prob(self, data, start=None):
        """P(data[i] | preceding bytes within the document) for i >= start
        (default: order-1, so every scored byte has a full context)."""
        start = self.order - 1 if start is None else start
        idx = np.arange(start, len(data))
        p = np.full(len(idx), 1.0 / B)
        for n in range(1, self.order + 1):
            t = self.tables[n]
            hlen = n - 1
            ok = idx - hlen >= 0
            gram = np.zeros(len(idx), np.int64)
            for j in range(hlen, -1, -1):
                gram = gram * B + np.where(ok, data[np.maximum(idx - j, 0)], 0)
            ctx = gram // B
            c_hw = _lookup(t["gk"], t["gc"], gram).astype(np.float64)
            ci = np.searchsorted(t["cu"], ctx)
            ci = np.minimum(ci, len(t["cu"]) - 1)
            seen = ok & (t["cu"][ci] == ctx)
            tot = np.where(seen, t["tot"][ci], 1.0)
            D = t["D"][np.minimum(c_hw, 3).astype(int)]
            gamma = (t["D"][1] * t["N1"][ci] + t["D"][2] * t["N2"][ci] + t["D"][3] * t["N3"][ci]) / tot
            pn = np.maximum(c_hw - D, 0) / tot + gamma * p
            p = np.where(seen, pn, p)
        return p

    def bits_per_byte(self, docs, start=None):
        bits = np.concatenate([-np.log2(self.prob(d, start)) for d in docs])
        return float(bits.mean())


def context_counts(train_docs, eval_doc, k, start):
    """Training count c(h) of each scored position's k-byte context."""
    keys = np.concatenate([_keys(d[:-1], k) for d in train_docs if len(d) >= k + 1])
    u, c = _count(keys)
    idx = np.arange(start, len(eval_doc))
    gram = np.zeros(len(idx), np.int64)
    for j in range(k, 0, -1):
        gram = gram * B + eval_doc[idx - j]
    return _lookup(u, c, gram)
