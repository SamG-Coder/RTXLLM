"""Compare independently traced CPU/GPU validation paths from the E1 package."""

import argparse
import json
from dataclasses import asdict
from time import perf_counter

import numpy as np
from e1 import E1Scene, open_kernel
from rtxllm.core.scene import (
    GeometryConfig,
    TraceParams,
    build_scene,
    shuffle_destinations,
)
from rtxllm.core.world import WorldConfig, generate
from rtxllm.cpu.tracer import trace
from rtxllm.evaluation import method_lookup, score


def compare(device, kernel, scene, paths, repeat, cap=None):
    start = np.array([p.start for p in paths], np.int32)
    programs = [p.program[:cap] for p in paths]
    width = max(map(len, programs))
    padded = np.full((len(paths), width), -1, np.int32)
    for i, program in enumerate(programs):
        padded[i, : len(program)] = program
    t = perf_counter()
    ref = trace(scene, start, programs, record_events=False)
    cpu_ms = (perf_counter() - t) * 1000
    arrays = {k: getattr(scene, k) for k in scene.ARRAYS}
    gpu = E1Scene(device, dict(arrays, params=json.dumps(asdict(scene.params))))
    result, metrics = gpu.trace(kernel, start, padded)
    for got, gold in [
        (result["answers"], ref.answer),
        (result["lost"], np.column_stack((ref.miss, ref.floor, ref.overflow))),
    ]:
        np.testing.assert_allclose(got, gold, rtol=0, atol=1e-5)
    energies = result["answers"]
    np.testing.assert_array_equal(
        np.where(energies.sum(1) >= 0.5, energies.argmax(1), -1), ref.predict()
    )
    np.testing.assert_array_equal(result["counters"][:, 1], ref.fact_hits_off_emit)
    np.testing.assert_equal(int(result["counters"][:, 0].sum()), ref.rays_traced)
    expected = np.full_like(result["paths"], -2)
    for i, paths_i in enumerate(ref.receiver_paths):
        if paths_i:
            seq = max(paths_i, key=lambda path: path[1])[0]
            expected[i, : len(seq)] = seq
    np.testing.assert_array_equal(result["paths"], expected)
    times = []
    wall = perf_counter()
    for _ in range(repeat):
        _, timing = gpu.trace(kernel, start, padded)
        times.append(timing["dispatch_gpu_ms"])
    metrics.update(
        gpu_batch_wall_ms=(perf_counter() - wall) * 1000 / repeat,
        dispatch_gpu_median_ms=float(np.median(times)),
        cpu_batch_ms=cpu_ms,
        queries=len(start),
        triangles=scene.n_triangles,
        rays=ref.rays_traced,
        floor_energy=float(ref.floor.sum()),
        overflow_energy=float(ref.overflow.sum()),
    )
    metrics["scores"] = score(energies, paths)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--stress-caps", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    world = generate(WorldConfig(seed=args.seed))
    paths = world.split(args.split)
    geometry = GeometryConfig(receiver_radius=args.radius, fact_radius=args.radius)
    device, kernel = open_kernel("e1.slang")
    for mode in ("specular", "transport"):
        params = TraceParams(hop_mode=mode, transport_hits_facts=not args.masked)
        scene = build_scene(world, geometry, params)
        if args.stress_caps:
            scene = scene.with_params(energy_floor=0.6, max_events=3)
        measured = compare(device, kernel, scene, paths, args.repeat)
        if args.controls:
            measured["one_hop"] = compare(
                device, kernel, scene, paths, args.repeat, cap=1
            )["scores"]
            shuffled = shuffle_destinations(scene, seed=1000 + args.seed)
            measured["shuffled"] = compare(
                device, kernel, shuffled, paths, args.repeat
            )["scores"]
            measured["lookup"] = score(method_lookup(world, paths)[0], paths)
        print(
            json.dumps(
                {"mode": mode, "seed": args.seed, "split": args.split, **measured}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
