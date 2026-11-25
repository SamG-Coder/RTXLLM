"""Write the small CPU/RTX parity fixtures (deterministic).

    python tests/fixtures/make_fixture.py

For each hop mode writes ``e1_small_<mode>.npz`` containing the scene arrays
(same names as ``scene.npz``), the queries and the CPU reference outputs:

* ``start i32[Q]``, ``program i32[Q,K]`` (-1 padded)
* ``expected_answer f32[Q,S]``, ``expected_miss/floor/overflow f32[Q]``
* ``expected_pred i32[Q]`` (decoded station or -1 = abstain)
* ``expected_hops i32[Q,K+1]`` station sequence of the dominant lineage
  (-1 = miss at that hop, -2 = padding)

Uses val-split paths of a 64-station world; no test-split queries.
"""

import sys
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtxllm.core.scene import Scene, TraceParams, build_scene  # noqa: E402
from rtxllm.core.world import WorldConfig, generate  # noqa: E402
from rtxllm.cpu.tracer import trace  # noqa: E402

CONFIG = WorldConfig(seed=7, grid=4, per_bucket={"unique": 20, "ambiguous": 5, "miss": 10})


def build(mode):
    world = generate(CONFIG)
    scene = build_scene(world, params=TraceParams(hop_mode=mode))
    paths = world.split("val")
    K = max(len(p.program) for p in paths)
    start = np.array([p.start for p in paths], np.int32)
    program = np.full((len(paths), K), -1, np.int32)
    for i, p in enumerate(paths):
        program[i, :len(p.program)] = p.program
    r = trace(scene, start, [p.program for p in paths])
    hops = np.full((len(paths), K + 1), -2, np.int32)
    for i, rp in enumerate(r.receiver_paths):
        seq = max(rp, key=lambda x: x[1])[0]
        hops[i, :len(seq)] = seq
    arrays = {k: getattr(scene, k) for k in Scene.ARRAYS}
    return dict(arrays, params=json.dumps(asdict(scene.params)), start=start, program=program,
                expected_answer=r.answer, expected_miss=r.miss, expected_floor=r.floor,
                expected_overflow=r.overflow, expected_pred=r.predict().astype(np.int32),
                expected_hops=hops)


def main():
    out = Path(__file__).resolve().parent
    for mode in ("specular", "transport"):
        np.savez_compressed(out / f"e1_small_{mode}.npz", **build(mode))
        print("wrote", out / f"e1_small_{mode}.npz")


if __name__ == "__main__":
    main()
