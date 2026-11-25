"""E1: multi-bounce retrieval vs direct lookup, one hop and randomized geometry.

Usage (repository root):
    python experiments/e1_multihop/run.py sweep          # validation split, geometry sweep
    python experiments/e1_multihop/run.py test --receiver-radius 0.1   # sealed test split, once

Writes JSON to experiments/e1_multihop/results/. Seeds are fixed (0, 1, 2).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rtxllm.core.scene import GeometryConfig, TraceParams, build_scene  # noqa: E402
from rtxllm.core.world import WorldConfig, generate  # noqa: E402
from rtxllm.evaluation import check_acceptance, evaluate  # noqa: E402

SEEDS = (0, 1, 2)
MODES = ("specular", "transport")
OUT = Path(__file__).resolve().parent / "results"


def run(split, radius, masked):
    out = {}
    for mode in MODES:
        by_seed = {}
        for seed in SEEDS:
            world = generate(WorldConfig(seed=seed))
            geom = GeometryConfig(receiver_radius=radius, fact_radius=radius)
            scene = build_scene(world, geom, TraceParams(hop_mode=mode, transport_hits_facts=not masked))
            res = evaluate(world, scene, split, shuffle_seed=1000 + seed)
            res["scene"] = {"triangles": scene.n_triangles, "stations": world.n_stations,
                            "facts": len(world.facts), "params": asdict(scene.params), "geometry": asdict(geom)}
            by_seed[seed] = res
        out[mode] = by_seed
    return out


def summarize(out):
    lines = []
    for mode, by_seed in out.items():
        for seed, res in by_seed.items():
            row = [f"{mode:9s} seed{seed}"]
            for k in ("L2/unique", "L3/unique"):
                accs = " ".join(f"{m}={res[m]['metrics'][k]['accuracy']:.3f}"
                                for m in ("trace", "trace_1hop", "trace_shuffled", "lookup", "graph"))
                row.append(f"{k}: {accs}")
            for k in ("L2/miss", "L3/miss"):
                row.append(f"{k} abstain={res['trace']['metrics'][k]['abstain_accuracy']:.3f}")
            for k in ("L2/ambiguous", "L3/ambiguous"):
                m = res["trace"]["metrics"][k]
                row.append(f"{k} mass={m['mass_on_answer_set']:.3f} tv={m['total_variation']:.3f}")
            t = res["trace"]
            row.append(f"hop-agree={t['per_hop_agreement']:.4f} xtalk={t['crosstalk_rate']:.4f} "
                       f"ms/q={t['cpu_ms_per_query']:.2f} rays/q={t['rays_per_query']:.2f}")
            lines.append(" | ".join(row))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["sweep", "test"])
    ap.add_argument("--receiver-radius", type=float, default=0.1)
    ap.add_argument("--masked", action="store_true", help="transport legs see receivers only")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    env = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
           "processor": platform.processor()}
    if a.split == "sweep":
        allres = {}
        for radius in (0.05, 0.1, 0.2, 0.4):
            for masked in (False, True):
                key = f"r{radius}_{'masked' if masked else 'open'}"
                allres[key] = run("val", radius, masked)
                print(f"== val {key}\n{summarize(allres[key])}", flush=True)
        (OUT / "val_sweep.json").write_text(json.dumps({"env": env, "results": allres}, indent=1))
    else:
        out = run("test", a.receiver_radius, a.masked)
        print(summarize(out))
        acc = {mode: [(c, bool(ok), d) for c, ok, d in check_acceptance(out[mode])] for mode in MODES}
        for mode, checks in acc.items():
            failed = [c for c in checks if not c[1]]
            print(f"{mode}: {len(checks) - len(failed)}/{len(checks)} acceptance checks pass")
            for c in failed:
                print("  FAIL", c)
        name = f"test_r{a.receiver_radius}_{'masked' if a.masked else 'open'}.json"
        (OUT / name).write_text(json.dumps({"env": env, "results": out, "acceptance": acc}, indent=1))


if __name__ == "__main__":
    main()
