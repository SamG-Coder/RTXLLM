"""Interpolated modified Kneser-Ney byte n-gram baseline (Chen & Goodman 1998).

A strong count baseline for RTX byte models (issue #23). Vectorised with NumPy:
n-grams of up to 8 bytes are packed into **unsigned** 64-bit keys (byte data is
cast to uint64 before any arithmetic, so there is no sign extension and no float
promotion). Counts are found by sorted-key search. Documents shorter than an
order contribute no n-grams of that order; an empty table backs off fully.

P_n(w | h) = max(c(h,w) - D_n(c), 0) / c(h) + gamma_n(h) * P_{n-1}(w | h')
gamma_n(h) = (D1 N1(h.) + D2 N2(h.) + D3 N3+(h.)) / c(h)
Highest order uses raw counts, lower orders continuation counts N1+(. h' w).
Unseen h backs off fully. Order 0 is uniform over 256 bytes.
"""

from __future__ import annotations

import numpy as np

B = 256
U = np.uint64
BU = U(B)
EMPTY = np.zeros(0, U)


def _u(data):
    return np.asarray(data).astype(U)


def _keys(data, n):
    """uint64 keys of all n-grams (n bytes) in ``data``, aligned to the n-gram end."""
    data = _u(data)
    if len(data) < n:
        return EMPTY
    k = np.zeros(len(data) - n + 1, U)
    for j in range(n):
        k = k * BU + data[j:len(data) - n + 1 + j]
    return k


def _count(keys):
    u, c = np.unique(keys, return_counts=True)
    return u.astype(U), c


def _find(u, q):
    """(index, present) of keys ``q`` in sorted ``u``; index is valid only where present."""
    if len(u) == 0:
        return np.zeros(len(q), np.int64), np.zeros(len(q), bool)
    i = np.minimum(np.searchsorted(u, q), len(u) - 1)
    return i, u[i] == q


def _lookup(u, c, q):
    i, present = _find(u, q)
    if len(u) == 0:
        return np.zeros(len(q), np.int64)
    return np.where(present, c[i], 0)


class KneserNey:
    def __init__(self, order: int):
        assert 1 <= order <= 8, "uint64 keys hold at most 8 bytes"
        self.order = order        # n-gram order (context length order-1)

    def fit(self, docs):
        n_max = self.order
        self.tables = {}
        # raw n-gram counts at every order, per document (no cross-boundary grams)
        raw = {n: np.concatenate([EMPTY] + [_keys(d, n) for d in docs]) for n in range(1, n_max + 1)}
        for n in range(1, n_max + 1):
            if n == n_max:
                gk, gc = _count(raw[n])
            else:
                # continuation count: number of distinct left extensions x of (h' w)
                gk, gc = _count(np.unique(raw[n + 1]) % (BU ** U(n)))
            # discounts from count-of-counts
            n1, n2, n3, n4 = (int(np.sum(gc == i)) for i in (1, 2, 3, 4))
            Y = n1 / max(1, n1 + 2 * n2)
            D = np.array([0.0,
                          max(0.0, 1 - 2 * Y * n2 / max(1, n1)),
                          max(0.0, 2 - 3 * Y * n3 / max(1, n2)),
                          max(0.0, 3 - 4 * Y * n4 / max(1, n3))])
            ctx = gk // BU
            # per-context totals and N1/N2/N3+ of its continuations
            cu, inv = np.unique(ctx, return_inverse=True)
            tot = np.bincount(inv, weights=gc, minlength=len(cu)).astype(np.float64)
            N1 = np.bincount(inv, weights=(gc == 1), minlength=len(cu)).astype(np.float64)
            N2 = np.bincount(inv, weights=(gc == 2), minlength=len(cu)).astype(np.float64)
            N3 = np.bincount(inv, weights=(gc >= 3), minlength=len(cu)).astype(np.float64)
            self.tables[n] = dict(gk=gk, gc=gc, D=D, cu=cu.astype(U), tot=tot, N1=N1, N2=N2, N3=N3)
        return self

    def prob(self, data, start=None):
        """P(data[i] | preceding bytes within the document) for i >= start
        (default: order-1, so every scored byte has a full context)."""
        start = self.order - 1 if start is None else start
        data = _u(data)
        idx = np.arange(start, len(data))
        p = np.full(len(idx), 1.0 / B)
        for n in range(1, self.order + 1):
            t = self.tables[n]
            if len(t["cu"]) == 0:
                continue                      # nothing of this order was seen: full backoff
            hlen = n - 1
            ok = idx - hlen >= 0
            gram = np.zeros(len(idx), U)
            for j in range(hlen, -1, -1):
                gram = gram * BU + np.where(ok, data[np.maximum(idx - j, 0)], U(0))
            ctx = gram // BU
            c_hw = _lookup(t["gk"], t["gc"], gram).astype(np.float64)
            ci, present = _find(t["cu"], ctx)
            seen = ok & present
            tot = np.where(seen, t["tot"][ci], 1.0)
            D = t["D"][np.minimum(c_hw, 3).astype(np.int64)]
            gamma = (t["D"][1] * t["N1"][ci] + t["D"][2] * t["N2"][ci] + t["D"][3] * t["N3"][ci]) / tot
            pn = np.maximum(c_hw - D, 0) / tot + gamma * p
            p = np.where(seen, pn, p)
        return p

    def bits_per_byte(self, docs, start=None):
        bits = np.concatenate([-np.log2(self.prob(d, start)) for d in docs])
        return float(bits.mean())


def context_counts(train_docs, eval_doc, k, start):
    """Training count c(h) of each scored position's k-byte context."""
    keys = np.concatenate([EMPTY] + [_keys(np.asarray(d)[:-1], k) for d in train_docs if len(d) >= k + 1])
    u, c = _count(keys)
    ev = _u(eval_doc)
    idx = np.arange(start, len(ev))
    gram = np.zeros(len(idx), U)
    for j in range(k, 0, -1):
        gram = gram * BU + ev[idx - j]
    return _lookup(u, c, gram)
