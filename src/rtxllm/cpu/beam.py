"""Hop-synchronous beam propagation (E3 / proposed ADR 0002), CPU reference.

Differences from the E1 pencil-ray tracer:

* **Beams.** Each hop, every station ``x`` holding energy ``E`` emits one beam
  along ``d = relation_dir[r]``: a *core* pencil ray through the station centre
  carrying ``core * E`` and ``rays`` halo rays on a deterministic sunflower disk
  of radius ``width`` perpendicular to ``d`` sharing ``(1 - core) * E``. The beam
  starts ``span`` behind the station and reaches ``fact_distance + 2 span``, so it
  can meet mirrors that *neighbouring* stations wrote for relation ``r``.
* **Relation channels.** Emission rays only see fact triangles of their own
  relation (``tri_relation == r``; on RTX: one instance-mask bit per relation).
  Transport rays only see receivers.
* **Hop-synchronous merge.** Energy arriving at a station during hop ``h`` is
  summed and re-emitted as one beam at hop ``h + 1`` (a station re-radiates what
  it received), so rays per hop scale with occupied stations, not with B**h.

With ``width = 0`` (or ``core = 1``) this is the E1 pencil-ray semantics with
relation channels. Facts must be single-valued (R = 1) here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rtxllm.core.scene import KIND_FACT, KIND_RECEIVER, Scene
from rtxllm.cpu.tracer import closest_of_pairs, mt_closest, mt_pairs


@dataclass(frozen=True)
class BeamParams:
    width: float = 1.0
    rays: int = 256
    core: float = 0.5
    span: float | None = None      # default: width
    fact_distance: float = 1.5     # must match the GeometryConfig used to write facts
    energy_floor: float = 1e-5     # station energies below this are not re-emitted


def sunflower(n, radius):
    """Deterministic, area-uniform points in a disk."""
    i = np.arange(n) + 0.5
    rr = radius * np.sqrt(i / n)
    th = i * np.pi * (3 - 5 ** 0.5)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1)


def _basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(d, a)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(d, t1)


@dataclass
class BeamResult:
    answer: np.ndarray        # f32 [Q, S]
    lost: np.ndarray          # f32 [Q] missed or below-floor energy
    rays: int
    mirror_hit_energy: np.ndarray  # f32 [Q, hops] energy that met any mirror per hop
    record: list = None            # per (hop, relation): dict(hop, r, qs, xs, landing [nb, B])

    def predict(self, threshold=1e-9):
        tot = self.answer.sum(1)
        return np.where(tot > threshold, self.answer.argmax(1), -1)


class BeamTracer:
    """``broadphase=True`` (default) prunes candidates with conservative bounding
    tests (beam cylinder for emission, bounding sphere for receivers) before the
    same Moller-Trumbore test; ``broadphase=False`` is the all-pairs reference.
    Both must give identical hits (tested)."""

    def __init__(self, scene: Scene, beam: BeamParams, broadphase=True):
        self.broadphase = broadphase
        assert np.all(scene.tri_reflectance[scene.tri_kind == KIND_FACT] == 1.0), "single-valued facts only"
        self.scene, self.beam = scene, beam
        V = scene.tri_vertices.astype(np.float32)
        rec = np.flatnonzero(scene.tri_kind == KIND_RECEIVER)
        self.rec = (rec, V[rec, 0], V[rec, 1] - V[rec, 0], V[rec, 2] - V[rec, 0])
        self.facts = {}
        for r in range(len(scene.relation_dir)):
            f = np.flatnonzero((scene.tri_kind == KIND_FACT) & (scene.tri_relation == r))
            n = np.cross(V[f, 1] - V[f, 0], V[f, 2] - V[f, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
            cen = V[f].mean(1)
            rad = np.linalg.norm(V[f] - cen[:, None], axis=-1).max(1)
            self.facts[r] = (f, V[f, 0], V[f, 1] - V[f, 0], V[f, 2] - V[f, 0], n.astype(np.float32), cen, rad)
        self.rec_r = float(np.linalg.norm(V[rec] - scene.station_pos[scene.tri_station[rec]][:, None], axis=-1).max()) * 1.001 + 1e-6
        w = beam.width
        self.span = w if beam.span is None else beam.span
        halo = sunflower(beam.rays, w) if (w > 0 and beam.core < 1) else np.zeros((0, 2))
        self.offsets = np.vstack([[0.0, 0.0], halo]).astype(np.float32)
        hw = (1.0 - beam.core) / max(1, len(halo)) if len(halo) else 0.0
        self.weights = np.concatenate([[beam.core if len(halo) else 1.0], np.full(len(halo), hw)]).astype(np.float32)

    def beam_offsets(self, r):
        """float32 [B, 3] world-space ray offsets for relation r (core first)."""
        d = self.scene.relation_dir[r].astype(np.float32)
        t1, t2 = _basis(d.astype(np.float64))
        return (self.offsets[:, :1] * t1 + self.offsets[:, 1:] * t2).astype(np.float32)

    def fact_normals(self):
        """float32 [T, 3] unit normals of fact triangles (zeros for receivers)."""
        n = np.zeros((self.scene.n_triangles, 3), np.float32)
        for f_ids, *_, fn, _, _ in self.facts.values():
            n[f_ids] = fn
        return n

    def trace(self, starts, programs, record=False) -> BeamResult:
        sc, P = self.scene, self.scene.params
        programs = [tuple(p) for p in programs]
        L = len(programs[0])
        assert all(len(p) == L for p in programs), "one program length per call"
        Q, S = len(starts), len(sc.station_pos)
        E = np.zeros((Q, S), np.float32)
        E[np.arange(Q), np.asarray(starts)] = 1.0
        lost = np.zeros(Q, np.float32)
        mhit = np.zeros((Q, L), np.float32)
        prog = np.asarray(programs)
        rays = 0
        rec = [] if record else None
        t_min = np.float32(P.t_min)
        reach = np.float32(self.beam.fact_distance + 2 * self.span + 0.5)
        rec_ids, rv0, re1, re2 = self.rec
        for h in range(L):
            small = (E > 0) & (E < self.beam.energy_floor)
            lost += np.where(small, E, 0).sum(1)
            E[small] = 0
            newE = np.zeros_like(E)
            for r in np.unique(prog[:, h]):
                qs, xs = np.nonzero((E > 0) & (prog[:, h] == r)[:, None])
                if len(qs) == 0:
                    continue
                d = sc.relation_dir[r].astype(np.float32)
                off = self.beam_offsets(r)
                B = len(off)
                landing = np.full(len(qs) * B, -1, np.int32)
                org = (sc.station_pos[xs][:, None, :] - self.span * d + off[None]).reshape(-1, 3)
                eng = (E[qs, xs][:, None] * self.weights[None]).reshape(-1)
                qq = np.repeat(qs, B)
                dirs = np.broadcast_to(d, org.shape).astype(np.float32)
                rays += len(org)
                f_ids, fv0, fe1, fe2, fn, fcen, frad = self.facts[r]
                tmax = np.full(len(org), reach, np.float32)
                if self.broadphase:
                    t, li = self._emit_broadphase(xs, d, B, org, dirs, tmax, t_min, fv0, fe1, fe2, fcen, frad, reach)
                else:
                    t, li = mt_closest(fv0, fe1, fe2, org, dirs, tmax, t_min)
                hit = li >= 0
                lost += np.bincount(qq[~hit], weights=eng[~hit], minlength=Q).astype(np.float32)
                np.add.at(mhit[:, h], qq[hit], eng[hit])
                if rec is not None:
                    rec.append(dict(hop=h, r=int(r), qs=qs, xs=xs, landing=landing.reshape(len(qs), B)))
                if not hit.any():
                    continue
                p = org[hit] + t[hit, None] * dirs[hit]
                n = fn[li[hit]]
                dr = dirs[hit] - 2 * (dirs[hit] * n).sum(1, keepdims=True) * n
                dr = (dr / np.linalg.norm(dr, axis=1, keepdims=True)).astype(np.float32)
                q2, e2 = qq[hit], eng[hit]
                rays += len(p)
                tmax2 = np.full(len(p), P.transport_range, np.float32)
                if self.broadphase:
                    t2_, lr = self._receiver_broadphase(p.astype(np.float32), dr, tmax2, t_min)
                else:
                    t2_, lr = mt_closest(rv0, re1, re2, p.astype(np.float32), dr, tmax2, t_min)
                ok = lr >= 0
                lost += np.bincount(q2[~ok], weights=e2[~ok], minlength=Q).astype(np.float32)
                st = sc.tri_station[rec_ids[lr[ok]]]
                landing[np.flatnonzero(hit)[ok]] = st
                np.add.at(newE, (q2[ok], st), e2[ok])
            E = newE
        return BeamResult(E, lost, rays, mhit, rec)

    def _emit_broadphase(self, xs, d, B, org, dirs, tmax, t_min, fv0, fe1, fe2, fcen, frad, reach):
        """Candidates per beam: facts whose bounding sphere meets the beam cylinder."""
        sc = self.scene
        axis0 = sc.station_pos[xs] - self.span * d            # [nb, 3]
        w = float(np.abs(self.offsets).max()) if len(self.offsets) else 0.0
        w = max(w, float(np.linalg.norm(self.offsets, axis=1).max()))
        ri_all, ti_all = [], []
        for a in range(0, len(xs), 64):
            b = min(len(xs), a + 64)
            rel = fcen[None] - axis0[a:b, None]               # [nb, F, 3]
            depth = rel @ d
            perp = np.linalg.norm(rel - depth[..., None] * d, axis=-1)
            cand = (perp <= w + frad[None] + 1e-4) & (depth >= -frad[None] - 1e-4) & (depth <= reach + frad[None] + 1e-4)
            bi, fi = np.nonzero(cand)
            if len(bi) == 0:
                continue
            # every ray of beam bi against facet fi
            bi_g = bi + a
            ri = (bi_g[:, None] * B + np.arange(B)[None]).reshape(-1)
            ti = np.repeat(fi, B)
            ri_all.append(ri)
            ti_all.append(ti)
        if not ri_all:
            return np.full(len(org), np.inf, np.float32), np.full(len(org), -1, np.int64)
        ri = np.concatenate(ri_all)
        ti = np.concatenate(ti_all)
        t = mt_pairs(fv0, fe1, fe2, org, dirs, ri, ti, tmax, t_min)
        keep = np.isfinite(t)
        return closest_of_pairs(len(org), ri[keep], ti[keep], t[keep])

    def _receiver_broadphase(self, o, d, tmax, t_min):
        """Candidates: receivers whose bounding sphere the ray segment can touch."""
        sc = self.scene
        rec_ids, rv0, re1, re2 = self.rec
        st_of_local = sc.tri_station[rec_ids]
        order = np.argsort(st_of_local, kind="stable")
        starts = np.searchsorted(st_of_local[order], np.arange(len(sc.station_pos)))
        counts = np.bincount(st_of_local, minlength=len(sc.station_pos))
        ri_all, ti_all = [], []
        for a in range(0, len(o), 2048):
            b = min(len(o), a + 2048)
            rel = sc.station_pos[None] - o[a:b, None]           # [n, S, 3]
            tc = (rel * d[a:b, None]).sum(-1)
            perp2 = (rel ** 2).sum(-1) - tc ** 2
            cand = (perp2 <= self.rec_r ** 2) & (tc >= -self.rec_r) & (tc <= tmax[a:b, None] + self.rec_r)
            r_i, s_i = np.nonzero(cand)
            if len(r_i) == 0:
                continue
            k = counts[s_i]
            ri = np.repeat(r_i + a, k)
            base = np.repeat(starts[s_i], k)
            within = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
            ri_all.append(ri)
            ti_all.append(order[base + within])
        if not ri_all:
            return np.full(len(o), np.inf, np.float32), np.full(len(o), -1, np.int64)
        ri = np.concatenate(ri_all)
        ti = np.concatenate(ti_all)
        t = mt_pairs(rv0, re1, re2, o, d, ri, ti, tmax, t_min)
        keep = np.isfinite(t)
        return closest_of_pairs(len(o), ri[keep], ti[keep], t[keep])
