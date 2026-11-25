"""#47 (Milestone 6): scale and profile the RTX layered-backoff byte scene.

    python experiments/m6_scale_backoff/run.py BOOK_DIR

Trains KN-6 on 1, 2 and 4 pinned books (Alice; +Pride; +Frankenstein, Dracula),
writes the layered scene, reloads it, and scores the pinned War of the Worlds
(used by #38 for the copy gate; nothing is selected here). Reports scene size,
build/fit time, GPU search time, transfers, payload fold, rays/s, parity with
NumPy KN, and an honest CPU sorted-key index comparison on the same positions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "rtx"))
sys.path.insert(0, str(ROOT / "src"))
from backoff import LayeredScene, load, open_backoff_kernel, save, write_scene  # noqa: E402

from rtxllm.baselines.ngram import KneserNey  # noqa: E402

ORDER, START = 6, 8
BOOKS = {
    "alice-11": "49a0b2726606e1290ac03a63978fa1dd1bd38a8d805704d98265f393533ea094",
    "pride-1342": "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163",
    "frankenstein-84": "c29a4f56dd91aa894a3f17d5f96b356ccb499e3e2c5a27964de911845e39efab",
    "dracula-345": "c802711470ddb2aaca81432fc101ed29c154e7c3d32df4172478695c33f68eae",
    "war-36": "0c9bea907452abd6e9a31d9152188ef5007f0e2d3e4a52d7a6a627a37616249f",
}
SETS = {"1 book": ["alice-11"], "2 books": ["alice-11", "pride-1342"],
        "4 books": ["alice-11", "pride-1342", "frankenstein-84", "dracula-345"]}


def body(folder, name):
    raw = (folder / f"rtxllm-{name}.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != BOOKS[name]:
        raise ValueError(f"digest mismatch: {name}")
    m = re.search(rb"(?ms)^\*\*\* START OF.*?\*\*\*[^\S\n]*\n(.*?)^\*\*\* END OF", raw)
    return np.frombuffer(m[1], np.uint8).astype(np.int64)


def timed(fn, repeat=5):
    ts, out = [], None
    for _ in range(repeat):
        t0 = perf_counter()
        out = fn()
        ts.append(perf_counter() - t0)
    return out, float(np.median(ts))


def main():
    folder = Path(sys.argv[1])
    test = body(folder, "war-36")
    idx = np.arange(START, len(test))
    device, kernel = open_backoff_kernel()
    report = {"order": ORDER, "test_targets": len(idx), "env": {
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "processor": platform.processor()}, "sizes": {}}
    for label, names in SETS.items():
        docs = [body(folder, n) for n in names]
        t0 = perf_counter()
        kn = KneserNey(ORDER).fit(docs)
        fit_s = perf_counter() - t0
        t0 = perf_counter()
        w = write_scene(kn)
        write_s = perf_counter() - t0
        path = Path(__file__).resolve().parent / f"scene_{len(names)}.npz"
        save(path, w)
        t0 = perf_counter()
        scene = LayeredScene(device, load(path))
        load_build_s = perf_counter() - t0
        pts = scene.points(test, idx)
        scene.trace(kernel, pts)                                   # warm-up
        (dishes, gpu_ms), search_s = timed(lambda: scene.trace(kernel, pts))
        _, fold_s = timed(lambda: scene.target_prob(dishes, test[idx]))
        p = scene.target_prob(dishes, test[idx])
        ref, cpu_kn_s = timed(lambda: kn.prob(test, start=START))
        parity_dishes = bool(np.array_equal(dishes, scene.expected_dishes(test, idx)))
        _, cpu_index_s = timed(lambda: scene.expected_dishes(test, idx))
        row = {"train_bytes": int(sum(len(d) for d in docs)), "fit_s": fit_s, "write_s": write_s,
               "load_and_as_build_s": load_build_s, **scene.metrics,
               "scene_file_bytes": path.stat().st_size,
               "bpb_rtx": float(-np.log2(p).mean()), "bpb_numpy_kn": float(-np.log2(ref).mean()),
               "max_abs_dp": float(np.abs(p - ref).max()), "dish_parity": parity_dishes,
               "gpu_search_dispatch_ms": gpu_ms, "search_host_wall_s": search_s,
               "transfer_and_sync_s": search_s - gpu_ms / 1000, "payload_fold_cpu_s": fold_s,
               "rays_per_s_dispatch": len(idx) / (gpu_ms / 1000),
               "ray_layer_intersections": int((dishes >= 0).sum()),
               "cpu_sorted_key_suffix_search_s": cpu_index_s, "cpu_numpy_kn_full_s": cpu_kn_s}
        report["sizes"][label] = row
        print(label, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    report["passes"] = all(r["dish_parity"] and abs(r["bpb_rtx"] - r["bpb_numpy_kn"]) < 1e-6
                           for r in report["sizes"].values())
    print("passes:", report["passes"])
    (Path(__file__).resolve().parent / "results.json").write_text(json.dumps(report, indent=1, default=float),
                                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
