"""RTX backend for ADR 0002 beams (Claude, issue #11) and CPU parity check.

Builds one BLAS for receivers (instance mask 0x1) and one per relation's fact
triangles (mask 1 << (1 + r)); each hop is one dispatch with one thread per ray.
Energy bookkeeping between hops (which stations hold energy, sums of arrivals)
is done on the host exactly as in ``rtxllm.cpu.beam``; every intersection is a
hardware ray query.

    python src/rtx/beam.py            # parity on a small clustered world
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

from rtxllm.core.scene import KIND_FACT, KIND_RECEIVER  # noqa: E402
from rtxllm.cpu.beam import BeamResult, BeamTracer  # noqa: E402


class BeamRTX:
    def __init__(self, device, kernel, cpu: BeamTracer):
        sc = cpu.scene
        self.device, self.kernel, self.cpu, self.scene = device, kernel, cpu, sc
        R = len(sc.relation_dir)
        if R > 7:
            raise ValueError("instance mask has 8 bits: at most 7 relation channels")
        key = np.where(sc.tri_kind == KIND_RECEIVER, 0, 1 + sc.tri_relation).astype(np.int64)
        if np.any((sc.tri_kind == KIND_FACT) & (sc.tri_relation < 0)):
            raise ValueError("fact triangles need tri_relation")
        self.order = np.argsort(key, kind="stable")
        verts = np.ascontiguousarray(sc.tri_vertices[self.order], np.float32)
        self.vertices = gpu_buffer(device, verts, sp.BufferUsage.shader_resource
                                   | sp.BufferUsage.acceleration_structure_build_input)
        groups, counts = np.unique(key[self.order], return_counts=True)
        self.instances = device.create_acceleration_structure_instance_list(len(groups))
        self.blas = []
        t0 = perf_counter()
        offset = 0
        for i, (g, c) in enumerate(zip(groups, counts)):
            geom = sp.AccelerationStructureBuildInputTriangles({
                "vertex_buffers": [{"buffer": self.vertices, "offset": offset * 36}],
                "vertex_count": int(c) * 3, "vertex_format": sp.Format.rgb32_float,
                "vertex_stride": 12, "flags": sp.AccelerationStructureGeometryFlags.opaque})
            blas = UpdatableStructure(device, geom, sp.AccelerationStructureKind.bottom_level)
            self.blas.append(blas)
            self.instances.write(i, {"transform": sp.float3x4.identity(), "instance_mask": 1 << int(g),
                                     "instance_id": offset, "acceleration_structure": blas.structure.handle})
            offset += int(c)
        self.tlas = UpdatableStructure(device, self.instances.build_input_instances(),
                                       sp.AccelerationStructureKind.top_level)
        self.metrics = {"as_build_wall_ms": (perf_counter() - t0) * 1000,
                        "as_bytes": sum(s.sizes.acceleration_structure_size for s in (*self.blas, self.tlas))}
        B = len(cpu.offsets)
        offs = np.concatenate([cpu.beam_offsets(r) for r in range(R)]).astype(np.float32)
        self.B = B
        self.bind = {
            "scene": self.tlas.structure,
            "stations": gpu_buffer(device, sc.station_pos.astype(np.float32)),
            "directions": gpu_buffer(device, sc.relation_dir.astype(np.float32)),
            "offsets": gpu_buffer(device, offs),
            "normals": gpu_buffer(device, cpu.fact_normals()[self.order]),
            "owners": gpu_buffer(device, sc.tri_station[self.order].astype(np.int32)),
            "raysPerBeam": B,
            "span": float(np.float32(cpu.span)),
            "reach": float(np.float32(cpu.beam.fact_distance + 2 * cpu.span + 0.5)),
            "transportRange": float(sc.params.transport_range),
            "tMin": float(sc.params.t_min),
        }

    def dispatch(self, stations, relations):
        nb = len(stations)
        out = gpu_buffer(self.device, np.full(nb * self.B, -2, np.int32), sp.BufferUsage.unordered_access)
        pool = self.device.create_query_pool(sp.QueryType.timestamp, 2)
        self.kernel.dispatch(thread_count=[nb * self.B, 1, 1], vars=dict(
            self.bind, beamStation=gpu_buffer(self.device, stations.astype(np.int32)),
            beamRelation=gpu_buffer(self.device, relations.astype(np.int32)), landing=out, beamCount=nb),
            query_pool=pool, query_index_before=0, query_index_after=1)
        self.device.wait_for_idle()
        a, b = pool.get_results(0, 2)
        return out.to_numpy().view(np.int32).reshape(nb, self.B), (b - a) * 1000 / self.device.info.timestamp_frequency

    def trace(self, starts, programs, record=False):
        """Same hop-synchronous semantics and energy bookkeeping as BeamTracer.trace."""
        cpu = self.cpu
        programs = [tuple(p) for p in programs]
        L = len(programs[0])
        assert all(len(p) == L for p in programs)
        Q, S = len(starts), len(self.scene.station_pos)
        E = np.zeros((Q, S), np.float32)
        E[np.arange(Q), np.asarray(starts)] = 1.0
        lost = np.zeros(Q, np.float32)
        prog = np.asarray(programs)
        rec, gpu_ms, rays = ([] if record else None), 0.0, 0
        w = cpu.weights
        for h in range(L):
            small = (E > 0) & (E < cpu.beam.energy_floor)
            lost += np.where(small, E, 0).sum(1)
            E[small] = 0
            newE = np.zeros_like(E)
            groups = [(r, *np.nonzero((E > 0) & (prog[:, h] == r)[:, None])) for r in np.unique(prog[:, h])]
            groups = [g for g in groups if len(g[1])]
            if not groups:
                E = newE
                continue
            st = np.concatenate([g[2] for g in groups])
            rl = np.concatenate([np.full(len(g[1]), g[0]) for g in groups])
            land, ms = self.dispatch(st, rl)
            gpu_ms += ms
            rays += land.size
            a = 0
            for r, qs, xs in groups:
                lg = land[a:a + len(qs)]
                a += len(qs)
                if rec is not None:
                    rec.append(dict(hop=h, r=int(r), qs=qs, xs=xs, landing=lg))
                eng = E[qs, xs][:, None] * w[None]
                qq = np.repeat(qs, self.B)
                lf, ef = lg.reshape(-1), eng.reshape(-1)
                ok = lf >= 0
                lost += np.bincount(qq[~ok], weights=ef[~ok], minlength=Q).astype(np.float32)
                np.add.at(newE, (qq[ok], lf[ok]), ef[ok])
            E = newE
        res = BeamResult(E, lost, rays, np.zeros((Q, L), np.float32), rec)
        res.gpu_ms = gpu_ms
        return res


def parity(seed=3, width=1.5, dish=0.15, rays=64, n=200):
    from rtxllm.core.cluster_world import ClusterConfig, generate_clustered
    from rtxllm.core.scene import GeometryConfig, build_scene
    from rtxllm.cpu.beam import BeamParams
    from rtxllm.learning.placement import force_layout

    w = generate_clustered(ClusterConfig(seed=seed))
    w = w.with_positions(force_layout(w.written, w.n_stations, seed))
    geom = GeometryConfig(receiver_radius=0.075, fact_radius=0.05, dish_radius=dish)
    scene = build_scene(w, geom)
    cpu = BeamTracer(scene, BeamParams(width=width, rays=rays, fact_distance=geom.fact_distance))
    device, kernel = open_kernel("beam.slang")
    gpu = BeamRTX(device, kernel, cpu)
    out = {"triangles": scene.n_triangles, **gpu.metrics}
    for name, qs in (("heldout_1hop", [(s, (r,)) for s, r in w.heldout["val"][:n]]),
                     ("two_hop", [(s, (a, b)) for s, a, b, _ in w.two_hop["val"][:n]])):
        starts, progs = [q[0] for q in qs], [q[1] for q in qs]
        t0 = perf_counter()
        c = cpu.trace(starts, progs, record=True)
        cpu_s = perf_counter() - t0
        t0 = perf_counter()
        g = gpu.trace(starts, progs, record=True)
        gpu_s = perf_counter() - t0
        n_rays = sum(x["landing"].size for x in c.record)
        agree = sum(int((x["landing"] == y["landing"]).sum()) for x, y in zip(c.record, g.record))
        same_beams = all(np.array_equal(x["xs"], y["xs"]) and np.array_equal(x["qs"], y["qs"])
                         for x, y in zip(c.record, g.record)) and len(c.record) == len(g.record)
        out[name] = {"queries": len(qs), "rays": n_rays, "ray_landing_agreement": agree / max(1, n_rays),
                     "same_beam_sets": bool(same_beams),
                     "max_answer_diff": float(np.abs(c.answer - g.answer).max()),
                     "pred_agreement": float(np.mean(c.predict() == g.predict())),
                     "cpu_s": cpu_s, "gpu_host_s": gpu_s, "gpu_dispatch_ms": g.gpu_ms}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(parity(), indent=1))
