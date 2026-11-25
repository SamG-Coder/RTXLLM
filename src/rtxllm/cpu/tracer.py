"""Brute-force float32 CPU reference tracer.

Every ray is tested against every triangle (no BVH) so correctness is easy to
audit; the RTX backend must reproduce the receiver-hit sequences produced here.

Bounce semantics (shared with the RTX backend, see docs/decisions/0001):

* A query is ``(start station, relation program)``. Hop ``h`` emits a ray from
  ``station_pos[x] + emit_offset * d`` along ``d = relation_dir[program[h]]``
  with ``tmax = emit_range``.
* Closest hit with ``t_min < t < tmax`` (two-sided Moller-Trumbore).
* Miss -> the lineage's energy is lost (``miss``).
* Receiver hit (any leg) -> hop complete at that station. If the program has
  more relations, re-emit from that station, else deposit energy as an answer.
* Fact hit -> split: ``R * e`` follows the fact (``specular``: reflected about
  the triangle normal, ``tmax = transport_range``; ``transport``: origin is
  displaced by ``tri_displacement`` and fired along it for ``arrival_range``),
  ``(1 - R) * e`` transmits straight on with the remaining ``tmax``.
  Branches below ``energy_floor`` are dropped (``floor``).
* A lineage with more than ``max_events`` intersections is dropped (``overflow``).
* If ``transport_hits_facts`` is False, non-emission legs only see receivers.

Energy conservation: answer + miss + floor + overflow == 1 for every query.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rtxllm.core.scene import KIND_FACT, KIND_RECEIVER, SPECULAR, Scene

LEG_EMIT, LEG_TRANSPORT, LEG_ARRIVAL = 0, 1, 2


@dataclass
class TraceResult:
    answer: np.ndarray        # f32 [Q, S] energy deposited at final stations
    miss: np.ndarray          # f32 [Q]
    floor: np.ndarray         # f32 [Q]
    overflow: np.ndarray      # f32 [Q]
    events: list              # per query: list of (lineage, leg, hop, tri, t, energy)
    receiver_paths: list      # per query: list of (station sequence tuple, energy) per finished lineage
    fact_hits_off_emit: np.ndarray  # i32 [Q] fact hits on non-emission legs (crosstalk indicator)
    rays_traced: int

    def predict(self, threshold=0.5):
        """argmax station, or -1 (abstain) when total answer energy < threshold."""
        tot = self.answer.sum(1)
        pred = self.answer.argmax(1)
        return np.where(tot >= threshold, pred, -1)


def mt_closest(v0, e1, e2, o, d, tmax, t_min, visible=None, chunk=96):
    """Closest-hit two-sided Moller-Trumbore over the given triangles (float32).

    ``visible`` optional bool [T] or callable(a, b) -> bool [b-a, T].
    Returns (t, local tri index or -1)."""
    N = len(o)
    out_t = np.full(N, np.inf, np.float32)
    out_i = np.full(N, -1, np.int64)
    if len(v0) == 0 or N == 0:
        return out_t, out_i
    eps = np.float32(1e-12)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        D = d[a:b, None, :]
        O = o[a:b, None, :]
        p = np.cross(D, e2[None])
        det = np.einsum("btk,tk->bt", p, e1)
        ok = np.abs(det) > eps
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0).astype(np.float32)
        s = O - v0[None]
        u = np.einsum("btk,btk->bt", s, p) * inv
        q = np.cross(s, e1[None])
        v = np.einsum("btk,btk->bt", D, q) * inv
        t = np.einsum("btk,tk->bt", q, e2) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > t_min) & (t < tmax[a:b, None])
        if visible is not None:
            hit &= visible(a, b) if callable(visible) else visible[None]
        t = np.where(hit, t, np.inf)
        i = np.argmin(t, axis=1)
        tb = t[np.arange(b - a), i]
        out_t[a:b] = tb
        out_i[a:b] = np.where(np.isfinite(tb), i, -1)
    return out_t, out_i


def mt_pairs(v0, e1, e2, o, d, ri, ti, tmax, t_min):
    """Moller-Trumbore on explicit (ray, triangle) candidate pairs; returns t (inf = no hit)."""
    D, O = d[ri], o[ri]
    p = np.cross(D, e2[ti])
    det = (p * e1[ti]).sum(1)
    ok = np.abs(det) > np.float32(1e-12)
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0).astype(np.float32)
    s = O - v0[ti]
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1[ti])
    v = (D * q).sum(1) * inv
    t = (q * e2[ti]).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > t_min) & (t < tmax[ri])
    return np.where(hit, t, np.inf).astype(np.float32)


def closest_of_pairs(N, ri, ti, t):
    """Reduce pair hits to the closest per ray; ties -> lowest triangle index."""
    out_t = np.full(N, np.inf, np.float32)
    out_i = np.full(N, -1, np.int64)
    if len(ri) == 0:
        return out_t, out_i
    np.minimum.at(out_t, ri, t)
    best = np.isfinite(t) & (t == out_t[ri])
    big = np.iinfo(np.int64).max
    tmp = np.full(N, big, np.int64)
    np.minimum.at(tmp, ri[best], ti[best])
    out_i = np.where(tmp == big, -1, tmp)
    return out_t, out_i


def intersect(scene: Scene, o, d, tmax, facts_visible, t_min, chunk=96):
    """Closest-hit brute force over the whole scene. Returns (t, tri) with tri = -1 for misses."""
    V = scene.tri_vertices
    v0 = V[:, 0]
    is_fact = scene.tri_kind == KIND_FACT
    vis = lambda a, b: facts_visible[a:b, None] | ~is_fact[None]  # noqa: E731
    return mt_closest(v0, V[:, 1] - v0, V[:, 2] - v0, o, d, tmax, t_min, vis, chunk)


def trace(scene: Scene, starts, programs, record_events=True) -> TraceResult:
    """Trace a batch of queries. ``programs`` is a list of relation-id sequences."""
    P = scene.params
    S = len(scene.station_pos)
    Q = len(starts)
    V = scene.tri_vertices
    normals = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals.astype(np.float32)

    answer = np.zeros((Q, S), np.float32)
    miss = np.zeros(Q, np.float32)
    floor = np.zeros(Q, np.float32)
    overflow = np.zeros(Q, np.float32)
    xtalk = np.zeros(Q, np.int32)
    events = [[] for _ in range(Q)]
    rpaths = [[] for _ in range(Q)]
    progs = [tuple(int(r) for r in p) for p in programs]

    # frontier (struct of arrays) + per-lineage python state
    def emit(qs, stations, hops):
        dirs = scene.relation_dir[[progs[q][h] for q, h in zip(qs, hops)]]
        org = scene.station_pos[stations] + np.float32(P.emit_offset) * dirs
        return org.astype(np.float32), dirs.astype(np.float32)

    q = np.arange(Q)
    o, d = emit(q, np.asarray(starts), np.zeros(Q, int))
    e = np.ones(Q, np.float32)
    hop = np.zeros(Q, int)
    leg = np.full(Q, LEG_EMIT)
    tmax = np.full(Q, P.emit_range, np.float32)
    nev = np.zeros(Q, int)
    lineage = [((int(s),), i) for i, s in enumerate(starts)]  # (station seq, lineage id)
    next_lid = Q
    rays = 0

    while len(q):
        rays += len(q)
        vis = (leg == LEG_EMIT) | bool(P.transport_hits_facts)
        t, tri = intersect(scene, o, d, tmax, vis, np.float32(P.t_min))
        nq, no, nd, ne, nh, nl, nt, nn, nlin = [], [], [], [], [], [], [], [], []

        def push(qi, oi, di, ei, hi, li, ti, ni, lin):
            nq.append(qi); no.append(oi); nd.append(di); ne.append(ei)
            nh.append(hi); nl.append(li); nt.append(ti); nn.append(ni); nlin.append(lin)

        for k in range(len(q)):
            qi, ek, hk, lk = int(q[k]), float(e[k]), int(hop[k]), int(leg[k])
            seq, lid = lineage[k]
            ti = int(tri[k])
            if ti < 0:
                miss[qi] += ek
                rpaths[qi].append((seq + (-1,), ek))
                continue
            if record_events:
                events[qi].append((lid, lk, hk, ti, float(t[k]), ek))
            if nev[k] + 1 > P.max_events:
                overflow[qi] += ek
                continue
            hit_p = o[k] + t[k] * d[k]
            if scene.tri_kind[ti] == KIND_RECEIVER:
                x = int(scene.tri_station[ti])
                nseq = seq + (x,)
                if hk + 1 < len(progs[qi]):
                    oo, dd = emit([qi], [x], [hk + 1])
                    push(qi, oo[0], dd[0], ek, hk + 1, LEG_EMIT, P.emit_range, nev[k] + 1, (nseq, lid))
                else:
                    answer[qi, x] += ek
                    rpaths[qi].append((nseq, ek))
                continue
            # fact hit
            if lk != LEG_EMIT:
                xtalk[qi] += 1
            R = float(scene.tri_reflectance[ti])
            er, et = ek * R, ek * (1.0 - R)
            if er > 0:
                if er < P.energy_floor:
                    floor[qi] += er
                elif P.hop_mode == SPECULAR:
                    n = normals[ti]
                    dr = d[k] - 2.0 * np.dot(d[k], n) * n
                    dr = (dr / np.linalg.norm(dr)).astype(np.float32)
                    push(qi, hit_p, dr, er, hk, LEG_TRANSPORT, P.transport_range, nev[k] + 1, (seq, lid))
                else:
                    disp = scene.tri_displacement[ti]
                    dr = (disp / np.linalg.norm(disp)).astype(np.float32)
                    push(qi, (hit_p + disp).astype(np.float32), dr, er, hk, LEG_ARRIVAL,
                         P.arrival_range, nev[k] + 1, (seq, lid))
            if et > 0:
                if et < P.energy_floor:
                    floor[qi] += et
                else:
                    push(qi, hit_p, d[k], et, hk, lk, float(tmax[k] - t[k]), nev[k] + 1, (seq, next_lid))
                    next_lid += 1

        if not nq:
            break
        q = np.asarray(nq)
        o = np.asarray(no, np.float32)
        d = np.asarray(nd, np.float32)
        e = np.asarray(ne, np.float32)
        hop = np.asarray(nh)
        leg = np.asarray(nl)
        tmax = np.asarray(nt, np.float32)
        nev = np.asarray(nn)
        lineage = nlin

    return TraceResult(answer, miss, floor, overflow, events, rpaths, xtalk, rays)
