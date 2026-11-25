"""#32: layered-dish Kneser-Ney byte model on RTX hardware (Claude).

Scene: layer m (contexts of the last m bytes, m = 0..k) is the plane
z = 1 + (k - m). Order-m context c owns the square dish
[x_m(c), x_m(c) + 16^-m)^2 with x_m(c) = sum_{j=1..m} 16^-j g(b_{t-j})
(g = byte -> 16x16 grid cell). For precision, each (layer, newest byte) is a
separate instance whose transform scales object space by 1/16 about that
byte's cell, so object-space dish coordinates carry 4 extra bits; all
coordinates are dyadic and exact in float32 up to k = 5 (order 6).

One ray per scored byte travels +z from x_k(query) + (0.5, 0.25) 16^-k and
meets at most one dish per layer — the dishes of the query's seen suffix
contexts, found by hardware intersection. Each dish's payload is its context's
KN statistics (reflectance 1 - gamma, discounted continuation counts); the host
folds hits from the coarsest layer up, i.e. interpolated KN. Starting the ray
deeper (``startZ``) removes the top layers: the bounce-budget ablation.
"""

from __future__ import annotations

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


def context_bytes(key, m):
    """[n, m] bytes of uint64 context keys, oldest first."""
    out = np.zeros((len(key), m), np.int64)
    k = key.copy()
    for j in range(m - 1, -1, -1):
        out[:, j] = (k % BU).astype(np.int64)
        k //= BU
    return out


class LayeredScene:
    def __init__(self, device, kn: KneserNey, grid=None):
        self.kn, self.device = kn, device
        self.k = kn.order - 1
        if self.k > 5:
            raise ValueError("dyadic coordinates are exact in float32 only up to order 6")
        self.g = identity_grid() if grid is None else grid
        tris, inst, self.dish_layer, self.dish_key = [], [], [], []
        dish = 0
        for m in range(self.k, -1, -1):
            cu = kn.tables[m + 1]["cu"]
            z = 1.0 + (self.k - m)
            if m == 0:
                groups = [(None, cu)]
            else:
                newest = (cu % BU).astype(np.int64)
                order = np.argsort(newest, kind="stable")
                cu = cu[order]
                newest = newest[order]
                cuts = np.flatnonzero(np.diff(newest)) + 1
                groups = [(int(nb[0]), part) for nb, part in zip(np.split(newest, cuts), np.split(cu, cuts))]
            for nb, keys in groups:
                if m == 0:
                    xy = np.zeros((len(keys), 2))
                    size = 1.0
                    T = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, z]
                else:
                    by = context_bytes(keys, m)          # oldest .. newest
                    xy = np.zeros((len(keys), 2))
                    for j in range(2, m + 1):            # object space: drop newest digit, zoom x16
                        xy += self.g[by[:, m - j]] * float(G) ** -(j - 1)
                    size = float(G) ** -(m - 1)
                    c = self.g[nb] / float(G)
                    T = [1.0 / G, 0, 0, c[0], 0, 1.0 / G, 0, c[1], 0, 0, 1.0, z]
                x0, y0 = xy[:, 0], xy[:, 1]
                a = np.stack([x0, y0, np.zeros_like(x0)], 1)
                b = np.stack([x0 + size, y0, np.zeros_like(x0)], 1)
                cc = np.stack([x0 + size, y0 + size, np.zeros_like(x0)], 1)
                d = np.stack([x0, y0 + size, np.zeros_like(x0)], 1)
                t = np.stack([np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)], 1).reshape(-1, 3, 3)
                inst.append((len(tris), len(t), dish, T))
                tris.append(t)
                self.dish_layer.append(np.full(len(keys), m))
                self.dish_key.append(keys)
                dish += len(keys)
        offs = np.cumsum([0] + [len(t) for t in tris])
        inst = [(int(offs[i]), n, d0, T) for i, (_, n, d0, T) in enumerate(inst)]
        V = np.concatenate(tris).astype(np.float32)
        if not np.array_equal(V.astype(np.float64), np.concatenate(tris)):
            raise ValueError("dish coordinates are not exact in float32")
        self.dish_layer = np.concatenate(self.dish_layer)
        self.dish_key = np.concatenate(self.dish_key)
        self.n_dishes = dish
        self.n_triangles = len(V)
        t0 = perf_counter()
        self.vertices = gpu_buffer(device, V, sp.BufferUsage.shader_resource
                                   | sp.BufferUsage.acceleration_structure_build_input)
        self.instances = device.create_acceleration_structure_instance_list(len(inst))
        self.blas = []
        for i, (first, n, d0, T) in enumerate(inst):
            geom = sp.AccelerationStructureBuildInputTriangles({
                "vertex_buffers": [{"buffer": self.vertices, "offset": first * 36}],
                "vertex_count": n * 3, "vertex_format": sp.Format.rgb32_float, "vertex_stride": 12,
                "flags": sp.AccelerationStructureGeometryFlags.opaque})
            bl = UpdatableStructure(device, geom, sp.AccelerationStructureKind.bottom_level)
            self.blas.append(bl)
            self.instances.write(i, {"transform": sp.float3x4(T), "instance_mask": 0xFF,
                                     "instance_id": d0, "acceleration_structure": bl.structure.handle})
        self.tlas = UpdatableStructure(device, self.instances.build_input_instances(),
                                       sp.AccelerationStructureKind.top_level)
        self.metrics = {"build_wall_ms": (perf_counter() - t0) * 1000, "instances": len(inst),
                        "dishes": self.n_dishes, "triangles": self.n_triangles,
                        "as_bytes": int(sum(b.sizes.acceleration_structure_size for b in self.blas)
                                        + self.tlas.sizes.acceleration_structure_size)}

    def points(self, data, idx):
        p = np.zeros((len(idx), 2))
        for j in range(1, self.k + 1):
            p += self.g[data[idx - j]] * float(G) ** -j
        p += np.array([0.5, 0.25]) * float(G) ** -self.k
        p32 = p.astype(np.float32)
        if not np.array_equal(p32.astype(np.float64), p):
            raise ValueError("query point not exact in float32")
        return p32

    def trace(self, kernel, pts, budget=None):
        """Dish index hit on each layer slot (0 = most specific), -1 = no dish."""
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

    def expected_dishes(self, data, idx):
        """CPU reference: dish of each seen suffix context, -1 if unseen."""
        L = self.k + 1
        exp = np.full((len(idx), L), -1, np.int64)
        base = 0
        data = np.asarray(data).astype(U)
        for s, m in enumerate(range(self.k, -1, -1)):
            cu = self.kn.tables[m + 1]["cu"]
            if m == 0:
                exp[:, s] = base
            else:
                key = np.zeros(len(idx), U)
                for j in range(m, 0, -1):
                    key = key * BU + data[idx - j]
                # dishes of layer m are stored grouped by newest byte; map key -> dish index
                layer = np.flatnonzero(self.dish_layer == m)
                order = np.argsort(self.dish_key[layer], kind="stable")
                sk = self.dish_key[layer][order]
                i, present = _find(sk, key)
                exp[:, s] = np.where(present, layer[order][i], -1)
            base += len(cu)
        return exp

    def target_prob(self, dishes, targets, budget=None):
        """Fold the hit dishes' KN payloads from coarse to fine."""
        L = self.k + 1
        budget = L if budget is None else budget
        p = np.full(len(targets), 1.0 / B)
        tg = np.asarray(targets).astype(U)
        for s in range(L - 1, L - 1 - budget, -1):     # coarsest layer first
            m = self.k - s
            t = self.kn.tables[m + 1]
            d = dishes[:, s]
            seen = d >= 0
            key = np.where(seen, self.dish_key[np.maximum(d, 0)], U(0))
            ci, present = _find(t["cu"], key)
            seen &= present
            tot = np.where(seen, t["tot"][ci], 1.0)
            gi, gp = _find(t["gk"], key * BU + tg)
            c = np.where(seen & gp, t["gc"][gi], 0).astype(np.float64)
            D = t["D"][np.minimum(c, 3).astype(np.int64)]
            gamma = (t["D"][1] * t["N1"][ci] + t["D"][2] * t["N2"][ci] + t["D"][3] * t["N3"][ci]) / tot
            p = np.where(seen, np.maximum(c - D, 0) / tot + gamma * p, p)
        return p


def open_backoff_kernel():
    return open_kernel("backoff.slang")
