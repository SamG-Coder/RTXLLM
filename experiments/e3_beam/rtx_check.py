"""E3 on RTX: run the selected configuration on hardware and check CPU parity.

    python experiments/e3_beam/rtx_check.py --dish 0.15 --width 2.0 --split test

For each seed and layout, traces held-out facts, the written-fact sample and the
two-hop queries on both backends; reports per-ray landing agreement, decoded
agreement, accuracy from the RTX trace, and timings. Requires slangpy + RTX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "rtx"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from beam import BeamRTX, open_kernel  # noqa: E402
from run import CORE, FACET, RAYS, RECEIVER, SEEDS, layouts, written_sample  # noqa: E402

from rtxllm.core.cluster_world import ClusterConfig, generate_clustered  # noqa: E402
from rtxllm.core.scene import GeometryConfig, build_scene  # noqa: E402
from rtxllm.cpu.beam import BeamParams, BeamTracer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dish", type=float, required=True)
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    a = ap.parse_args()
    device, kernel = open_kernel("beam.slang")
    report = {"config": vars(a), "seeds": {}}
    for seed in SEEDS:
        world = generate_clustered(ClusterConfig(seed=seed))
        for name, pos in layouts(world, seed).items():
            geom = GeometryConfig(receiver_radius=RECEIVER, fact_radius=FACET, dish_radius=a.dish)
            scene = build_scene(world.with_positions(pos), geom)
            cpu = BeamTracer(scene, BeamParams(width=a.width, core=CORE, rays=RAYS, fact_distance=geom.fact_distance))
            gpu = BeamRTX(device, kernel, cpu)
            ws = written_sample(world, seed=seed)
            sets = {
                "heldout": ([s for s, _ in world.heldout[a.split]], [(r,) for _, r in world.heldout[a.split]],
                            np.array([world.truth[k] for k in world.heldout[a.split]])),
                "written": ([s for s, _ in ws], [(r,) for _, r in ws], np.array([world.written[k][0] for k in ws])),
                "two_hop": ([q[0] for q in world.two_hop[a.split]], [(q[1], q[2]) for q in world.two_hop[a.split]],
                            np.array([q[3] for q in world.two_hop[a.split]])),
            }
            row = {"triangles": scene.n_triangles, **gpu.metrics}
            for key, (st, pr, gold) in sets.items():
                t0 = perf_counter()
                c = cpu.trace(st, pr, record=True)
                cs = perf_counter() - t0
                t0 = perf_counter()
                g = gpu.trace(st, pr, record=True)
                gs = perf_counter() - t0
                n = sum(x["landing"].size for x in c.record)
                agree = sum(int((x["landing"] == y["landing"]).sum()) for x, y in zip(c.record, g.record)) \
                    if len(c.record) == len(g.record) else 0
                row[key] = {"rtx_acc": float(np.mean(g.predict() == gold)),
                            "cpu_acc": float(np.mean(c.predict() == gold)),
                            "pred_agreement": float(np.mean(g.predict() == c.predict())),
                            "ray_landing_agreement": agree / max(1, n), "rays": n,
                            "max_answer_diff": float(np.abs(c.answer - g.answer).max()),
                            "cpu_s": cs, "rtx_host_s": gs, "rtx_dispatch_ms": g.gpu_ms}
            report["seeds"].setdefault(seed, {})[name] = row
            print(f"seed{seed} {name}: " + " | ".join(
                f"{k} rtx={v['rtx_acc']:.3f} cpu={v['cpu_acc']:.3f} rays={v['rays']} agree={v['ray_landing_agreement']:.6f} "
                f"gpu={v['rtx_dispatch_ms']:.3f}ms host={v['rtx_host_s']:.2f}s cpu={v['cpu_s']:.2f}s"
                for k, v in row.items() if isinstance(v, dict)), flush=True)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    (out / f"rtx_{a.split}_dish{a.dish}_w{a.width}.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
