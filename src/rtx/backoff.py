"""#32: layered-dish Kneser-Ney byte model on RTX hardware (Claude).

Writer (``write_scene``, NumPy, training time): layer m (contexts of the last m
bytes, m = 0..k) is the plane z = 1 + (k - m). Order-m context c owns the square
dish [x_m(c), x_m(c) + 16^-m)^2, x_m(c) = sum_{j=1..m} 16^-j g(b_{t-j})
(g = byte -> 16x16 grid cell). Each (layer, newest byte) is a separate instance
whose transform zooms object space x16 about that byte's cell; all coordinates
are dyadic and exact in float32 up to k = 5 (order 6).

Every dish carries a **hit-local payload**: its transmission gamma (KN backoff
weight) and its sparse discounted emissions {byte: max(c - D, 0) / c(h)}; per
dish, sum(emissions) + gamma = 1. The scene file (``save``/``load``) holds the
geometry, instance table and payloads; decoding needs nothing else — no KN object
or training counts.

Reader (``LayeredScene``): one hardware ray per scored byte travels +z from
x_k(query) + (0.5, 0.25) 16^-k (computed from the query's own bytes) and meets,
by intersection, at most one dish per layer — the dishes of its seen suffix
contexts. The host folds the hit dishes' payloads coarse -> fine:
p <- emission_d(w) + gamma_d * p (uniform base), i.e. interpolated KN.
Launching below the top layers drops the oldest bytes (bounce-budget ablation).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "~implicit~")

import numpy as np  # noqa: E402
import slangpy as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from run_specular import UpdatableStructure, gpu_buffer, open_kernel  # noqa: E402

from rtxllm.baselines.ngram import B, BU, U, KneserNey, _find  # noqa: E402
from rtxllm.learning.byte_layers import G, identity_grid  # noqa: E402

PAYLOAD = ("dish_layer", "dish_key", "dish_gamma", "em_ptr", "em_byte", "em_prob")


def context_bytes(key, m):
    """[n, m] bytes of uint64 context keys, oldest first."""
    out = np.zeros((len(key), m), np.int64)
    k = key.copy()
    for j in range(m - 1, -1, -1):
        out[:, j] = (k % BU).astype(np.int64)
        k //= BU
    return out


def write_scene(kn: KneserNey, grid=None) -> dict:
    """Write fitted KN statistics as layered dish geometry with per-dish payloads."""
    k = kn.order - 1
    if k > 5:
        raise ValueError("dyadic coordinates are exact in float32 only up to order 6")
    g = identity_grid() if grid is None else np.asarray(grid)
    tris, inst = [], []
    layer_l, key_l, gam_l, ptr_l, eb_l, ep_l = [], [], [], [0], [], []
    dish = 0
    for m in range(k, -1, -1):
        t = kn.tables[m + 1]
        cu = t["cu"]
        gamma_all = (t["D"][1] * t["N1"] + t["D"][2] * t["N2"] + t["D"][3] * t["N3"]) / t["tot"]
        z = 1.0 + (k - m)
        if m == 0:
            groups = [(None, np.arange(len(cu)))]
        else:
            newest = (cu % BU).astype(np.int64)
            order = np.argsort(newest, kind="stable")
            cuts = np.flatnonzero(np.diff(newest[order])) + 1
            groups = [(int(newest[o[0]]), o) for o in np.split(order, cuts)]
        gctx = t["gk"] // BU                      # emissions grouped by context (gk sorted -> ctx sorted)
        for nb, rows in groups:
            keys = cu[rows]
            if m == 0:
                xy = np.zeros((len(keys), 2))
                size = 1.0
                T = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, z]
            else:
                by = context_bytes(keys, m)          # oldest .. newest
                xy = np.zeros((len(keys), 2))
                for j in range(2, m + 1):            # object space: drop newest digit, zoom x16
                    xy += g[by[:, m - j]] * float(G) ** -(j - 1)
                size = float(G) ** -(m - 1)
                c = g[nb] / float(G)
                T = [1.0 / G, 0, 0, c[0], 0, 1.0 / G, 0, c[1], 0, 0, 1.0, z]
            x0, y0 = xy[:, 0], xy[:, 1]
            zz = np.zeros_like(x0)
            a, b = np.stack([x0, y0, zz], 1), np.stack([x0 + size, y0, zz], 1)
            cc, d = np.stack([x0 + size, y0 + size, zz], 1), np.stack([x0, y0 + size, zz], 1)
            tris.append(np.stack([np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)], 1).reshape(-1, 3, 3))
            inst.append((len(keys) * 2, dish, T))
            # payloads, in dish order
            lo = np.searchsorted(gctx, keys, "left")
            hi = np.searchsorted(gctx, keys, "right")
            for r, l0, h0 in zip(rows, lo, hi):
                cnt = t["gc"][l0:h0].astype(np.float64)
                D = t["D"][np.minimum(cnt, 3).astype(np.int64)]
                eb_l.append((t["gk"][l0:h0] % BU).astype(np.uint8))
                ep_l.append(np.maximum(cnt - D, 0) / t["tot"][r])
                ptr_l.append(ptr_l[-1] + (h0 - l0))
            layer_l.append(np.full(len(keys), m, np.int8))
            key_l.append(keys)
            gam_l.append(gamma_all[rows])
            dish += len(keys)
    V64 = np.concatenate(tris)
    V = V64.astype(np.float32)
    if not np.array_equal(V.astype(np.float64), V64):
        raise ValueError("dish coordinates are not exact in float32")
    first = np.cumsum([0] + [n for n, _, _ in inst])[:-1]
    return {
        "order": np.int64(kn.order), "grid": g.astype(np.int64), "vertices": V,
        "inst_first": first.astype(np.int64), "inst_count": np.array([n for n, _, _ in inst], np.int64),
        "inst_dish0": np.array([d0 for _, d0, _ in inst], np.int64),
        "inst_transform": np.array([T for _, _, T in inst], np.float32),
        "dish_layer": np.concatenate(layer_l), "dish_key": np.concatenate(key_l),
        "dish_gamma": np.concatenate(gam_l), "em_ptr": np.asarray(ptr_l, np.int64),
        "em_byte": np.concatenate([np.zeros(0, np.uint8)] + eb_l), "em_prob": np.concatenate([np.zeros(0)] + ep_l),
    }


def save(path, scene: dict):
    np.savez_compressed(path, **scene)


def load(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def payload_bytes(scene: dict) -> int:
    return int(sum(scene[k].nbytes for k in PAYLOAD))


class LayeredScene:
    """GPU acceleration structures + hit-local decoding from a scene dict only."""

    def __init__(self, device, scene: dict):
        self.device, self.s = device, scene
        self.k = int(scene["order"]) - 1
        self.g = scene["grid"]
        V = np.ascontiguousarray(scene["vertices"], np.float32)
        self.n_dishes = len(scene["dish_layer"])
        self.n_triangles = len(V)
        # emission lookup table keyed by (dish, byte)
        dish_of = np.repeat(np.arange(self.n_dishes, dtype=np.uint64), np.diff(scene["em_ptr"]))
        self.em_key = dish_of * BU + scene["em_byte"].astype(U)
        t0 = perf_counter()
        self.vertices = gpu_buffer(device, V, sp.BufferUsage.shader_resource
                                   | sp.BufferUsage.acceleration_structure_build_input)
        n_inst = len(scene["inst_first"])
        self.instances = device.create_acceleration_structure_instance_list(n_inst)
        self.blas = []
        for i in range(n_inst):
            first, n = int(scene["inst_first"][i]), int(scene["inst_count"][i])
            geom = sp.AccelerationStructureBuildInputTriangles({
                "vertex_buffers": [{"buffer": self.vertices, "offset": first * 36}],
                "vertex_count": n * 3, "vertex_format": sp.Format.rgb32_float, "vertex_stride": 12,
                "flags": sp.AccelerationStructureGeometryFlags.opaque})
            bl = UpdatableStructure(device, geom, sp.AccelerationStructureKind.bottom_level)
            self.blas.append(bl)
            self.instances.write(i, {"transform": sp.float3x4([float(x) for x in scene["inst_transform"][i]]),
                                     "instance_mask": 0xFF, "instance_id": int(scene["inst_dish0"][i]),
                                     "acceleration_structure": bl.structure.handle})
        self.tlas = UpdatableStructure(device, self.instances.build_input_instances(),
                                       sp.AccelerationStructureKind.top_level)
        self.metrics = {"build_wall_ms": (perf_counter() - t0) * 1000, "instances": n_inst,
                        "dishes": self.n_dishes, "triangles": self.n_triangles,
                        "vertex_bytes": int(V.nbytes), "payload_bytes": payload_bytes(scene),
                        "as_bytes": int(sum(b.sizes.acceleration_structure_size for b in self.blas)
                                        + self.tlas.sizes.acceleration_structure_size)}

    def points(self, data, idx):
        """Ray launch coordinates from the query's own bytes (no context lookup)."""
        p = np.zeros((len(idx), 2))
        for j in range(1, self.k + 1):
            p += self.g[data[idx - j]] * float(G) ** -j
        p += np.array([0.5, 0.25]) * float(G) ** -self.k
        p32 = p.astype(np.float32)
        if not np.array_equal(p32.astype(np.float64), p):
            raise ValueError("query point not exact in float32")
        return p32

    def trace(self, kernel, pts, budget=None):
        """Suffix search on RTX: dish index hit on each layer slot (0 = most specific),
        -1 = no dish. Returns (dishes, GPU dispatch ms) — the dispatch time is search only."""
        L = self.k + 1
        budget = L if budget is None else budget
        out = gpu_buffer(self.device, np.full(len(pts) * L, -2, np.int32), sp.BufferUsage.unordered_access)
        pool = self.device.create_query_pool(sp.QueryType.timestamp, 2)
        start_z = float(L - budget) + 0.5        # the ray meets only the bottom `budget` layers
        kernel.dispatch(thread_count=[len(pts), 1, 1], vars={
            "scene": self.tlas.structure, "points": gpu_buffer(self.device, np.ascontiguousarray(pts)),
            "dishes": out, "queryCount": len(pts), "layers": L, "startZ": start_z},
            query_pool=pool, query_index_before=0, query_index_after=1)
        self.device.wait_for_idle()
        a, b = pool.get_results(0, 2)
        return out.to_numpy().view(np.int32).reshape(len(pts), L), (b - a) * 1000 / self.device.info.timestamp_frequency

    def target_prob(self, dishes, targets, budget=None):
        """Fold the hit dishes' payloads coarse -> fine (scene data only)."""
        L = self.k + 1
        budget = L if budget is None else budget
        p = np.full(len(targets), 1.0 / B)
        tg = np.asarray(targets).astype(U)
        gam, ep = self.s["dish_gamma"], self.s["em_prob"]
        for s in range(L - 1, L - 1 - budget, -1):     # coarsest layer first
            d = dishes[:, s]
            seen = d >= 0
            dd = np.maximum(d, 0)
            i, present = _find(self.em_key, dd.astype(U) * BU + tg)
            em = np.where(seen & present, ep[i] if len(ep) else 0.0, 0.0)
            p = np.where(seen, em + gam[dd] * p, p)
        return p

    def distribution(self, dishes_row):
        """Full next-byte distribution for one query's hits (for normalisation checks)."""
        return np.array([self.target_prob(dishes_row[None, :], np.array([w]))[0] for w in range(B)])

    def expected_dishes(self, data, idx):
        """CPU reference: dish of each seen suffix context (by the scene's stored
        dish identities), -1 if unseen."""
        L = self.k + 1
        exp = np.full((len(idx), L), -1, np.int64)
        data = np.asarray(data).astype(U)
        lay, key = self.s["dish_layer"], self.s["dish_key"]
        for s, m in enumerate(range(self.k, -1, -1)):
            rows = np.flatnonzero(lay == m)
            q = np.zeros(len(idx), U)
            for j in range(m, 0, -1):
                q = q * BU + data[idx - j]
            order = np.argsort(key[rows], kind="stable")
            i, present = _find(key[rows][order], q)
            exp[:, s] = np.where(present, rows[order][i], -1)
        return exp


def rtx_available():
    return any(a.vendor_id == 0x10DE and "RTX" in a.name
               for a in sp.Device.enumerate_adapters(sp.DeviceType.vulkan))


def open_backoff_kernel():
    return open_kernel("backoff.slang")
