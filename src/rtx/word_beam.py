"""#53 (Claude): word-layout dish scene on RTX, with a brute-force CPU reference.

Each word with enough training evidence owns a hexagonal dish (6 triangles) at
``station_pos[w] + fact_distance * d`` facing the single "next" direction d,
radius growing with sqrt(count). A query beam is fired along d from the previous
word's station on an **annulus** whose inner radius exceeds the word's own dish,
so it reads only neighbours' dishes. Each ray's first hit returns a dish index;
the dish's payload (its next-word distribution, stored with the scene) is the
hit-local evidence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "~implicit~")

import numpy as np  # noqa: E402
import slangpy as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from run_specular import UpdatableStructure, gpu_buffer, open_kernel  # noqa: E402

from rtxllm.cpu.tracer import mt_closest  # noqa: E402

T_MIN = 1e-3


def basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(d, a)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(d, t1)


def hex_dishes(centres, radii, d):
    """[n*6, 3, 3] float32 triangles: a hexagonal fan per dish, normal along d."""
    t1, t2 = basis(np.asarray(d, np.float64))
    ang = np.arange(7) * np.pi / 3
    rim = np.cos(ang)[:, None] * t1 + np.sin(ang)[:, None] * t2     # [7, 3]
    c = np.asarray(centres, np.float64)[:, None, :]
    r = np.asarray(radii, np.float64)[:, None, None]
    pts = c + r * rim[None]                                           # [n, 7, 3]
    tris = np.stack([np.broadcast_to(c, pts[:, :6].shape), pts[:, :6], pts[:, 1:7]], axis=2)  # [n, 6, 3, 3]
    return tris.reshape(-1, 3, 3).astype(np.float32)


def annulus(n, inner, outer):
    """[n, 2] deterministic area-uniform points on inner < r <= outer."""
    i = np.arange(n) + 0.5
    rr = np.sqrt(inner ** 2 + (outer ** 2 - inner ** 2) * i / n)
    th = i * np.pi * (3 - 5 ** 0.5)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1)


def beam_origins(station_pos, words, own_radius, d, width, rays, back):
    """Ray origins [len(words)*rays, 3] for annular beams (inner = own dish radius * 1.05)."""
    t1, t2 = basis(np.asarray(d, np.float64))
    out = np.zeros((len(words), rays, 3))
    for k, (w, ri) in enumerate(zip(words, own_radius)):
        o2 = annulus(rays, ri * 1.05, max(width, ri * 1.1))
        out[k] = station_pos[w] - back * d + o2[:, :1] * t1 + o2[:, 1:] * t2
    return out.reshape(-1, 3).astype(np.float32)


class WordScene:
    def __init__(self, tris):
        self.tris = np.ascontiguousarray(tris, np.float32)
        V = self.tris
        self.v0, self.e1, self.e2 = V[:, 0], V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]

    def cpu_hits(self, origins, d, tmax):
        dirs = np.broadcast_to(np.asarray(d, np.float32), origins.shape).astype(np.float32)
        _, li = mt_closest(self.v0, self.e1, self.e2, origins, dirs,
                           np.full(len(origins), tmax, np.float32), np.float32(T_MIN), chunk=64)
        return np.where(li >= 0, li // 6, -1)


class WordRTX:
    def __init__(self, device, kernel, scene: WordScene):
        self.device, self.kernel = device, kernel
        self.vbuf = gpu_buffer(device, scene.tris, sp.BufferUsage.shader_resource
                               | sp.BufferUsage.acceleration_structure_build_input)
        geom = sp.AccelerationStructureBuildInputTriangles({
            "vertex_buffers": [{"buffer": self.vbuf, "offset": 0}], "vertex_count": len(scene.tris) * 3,
            "vertex_format": sp.Format.rgb32_float, "vertex_stride": 12,
            "flags": sp.AccelerationStructureGeometryFlags.opaque})
        self.blas = UpdatableStructure(device, geom, sp.AccelerationStructureKind.bottom_level)
        self.inst = device.create_acceleration_structure_instance_list(1)
        self.inst.write(0, {"transform": sp.float3x4.identity(), "instance_mask": 0xFF, "instance_id": 0,
                            "acceleration_structure": self.blas.structure.handle})
        self.tlas = UpdatableStructure(device, self.inst.build_input_instances(),
                                       sp.AccelerationStructureKind.top_level)
        self.as_bytes = int(self.blas.sizes.acceleration_structure_size + self.tlas.sizes.acceleration_structure_size)

    def hits(self, origins, d, tmax):
        out = gpu_buffer(self.device, np.full(len(origins), -2, np.int32), sp.BufferUsage.unordered_access)
        pool = self.device.create_query_pool(sp.QueryType.timestamp, 2)
        self.kernel.dispatch(thread_count=[len(origins), 1, 1], vars={
            "scene": self.tlas.structure, "origins": gpu_buffer(self.device, np.ascontiguousarray(origins)),
            "hitDish": out, "rayCount": len(origins), "direction": sp.float3(*[float(x) for x in d]),
            "tMin": float(T_MIN), "tMax": float(tmax)},
            query_pool=pool, query_index_before=0, query_index_after=1)
        self.device.wait_for_idle()
        a, b = pool.get_results(0, 2)
        return out.to_numpy().view(np.int32), (b - a) * 1000 / self.device.info.timestamp_frequency


def open_word_kernel():
    return open_kernel("word_beam.slang")
