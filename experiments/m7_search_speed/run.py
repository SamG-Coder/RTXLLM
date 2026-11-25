"""#91 (Claude): M7 — RT-core episodic search (E10 `indexedTrace`) vs a plain compute scan.

    python experiments/m7_search_speed/run.py BOOK_DIRECTORY report.json

Both paths run on the same Vulkan device. Outputs (recent, prediction, length) must be
bit-identical to each other and to the E10 CPU oracle (checked on Treasure Island in
full and on a 65,536-target prefix of Moby Dick). Timings are the median of 7 warm
repeats after one warm-up:
  gpu_ms   GPU timestamp interval around the search dispatch only;
  wall_ms  end to end: RT = acceleration-structure build + dispatch + readback;
           scan = byte upload + dispatch + readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "src" / "rtx", ROOT / "experiments" / "optical_cavity"):
    sys.path.insert(0, str(p))
from optics import TriangleScene, mesh, open_kernel  # noqa: E402
from run_specular import gpu_buffer  # noqa: E402

# Same scene construction as experiments/episodic (run.geometry, online.TILT); copied to
# avoid importing another module named `run`.
TILT = np.array([(-0.3, 0.3, -0.3), (0.3, -0.3, -0.3), (0, 0, 0.3)], np.float32)


def geometry(centers, offsets):
    vertices = (centers[:, None] + np.asarray(offsets, np.float32)).astype(np.float32)
    zeros = np.zeros(len(vertices), np.int32)
    return mesh(vertices, zeros, zeros, np.array([[1, 0, 0]]))

import slangpy as sp  # noqa: E402

BOOKS = {
    "treasure-120": "0fee2fef1b2ffde22e2c962bd8bc2b75ea1ae9b02b3fe85ef08d44f4cc1bb2de",
    "moby-2701": "baf9b04fcf6e0008f30eaaa6697277751448c1070b20175e0082f28cbb161fd3",
}
REPEATS = 7


def body(folder, name):
    raw = (folder / f"rtxllm-{name}.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != BOOKS[name]:
        raise ValueError(f"Digest mismatch for {name}")
    return re.search(rb"(?ms)^\*\*\* START OF.*?\*\*\*[^\S\n]*\n(.*?)^\*\*\* END OF", raw)[1]


def oracle(raw, window, paired, n):
    """E10 CPU oracle (experiments/pair_address/run_index.py), first n targets."""
    width = 2 if paired else 1
    result = np.empty((n, 3), np.int32)
    for t in range(8, 8 + n):
        lower = max(0, t - window)
        key = raw[t - width: t]
        cursor = t - 1
        first = pred = -1
        best = 0
        for _ in range(16):
            begin = raw.rfind(key, lower, cursor)
            if begin < 0:
                break
            end = begin + width - 1
            length = width
            while length < 8 and end - length >= lower and raw[end - length] == raw[t - 1 - length]:
                length += 1
            if length > best:
                best = length
                pred = raw[end + 1]
                if first < 0:
                    first = pred
            if best == 8:
                break
            cursor = end
        result[t - 8] = [first, pred, best]
    return result


def timed(device, kernel, n, vars_):
    out = gpu_buffer(device, np.empty((n, 4), np.int32), sp.BufferUsage.unordered_access)
    pool = device.create_query_pool(sp.QueryType.timestamp, 2)
    kernel.dispatch(thread_count=[n, 1, 1], vars={**vars_, "outcomes": out, "targets": n},
                    query_pool=pool, query_index_before=0, query_index_after=1)
    res = out.to_numpy().view(np.int32).reshape(n, 4).copy()
    a, b = pool.get_results(0, 2)
    return res, (b - a) * 1000 / device.info.timestamp_frequency


def rt_run(device, kernel, raw, window, paired):
    t0 = perf_counter()
    column = np.frombuffer(raw, np.uint8).astype(np.int32)
    if paired:
        column[1:] += 256 * column[:-1]
    centers = np.column_stack((column, np.arange(len(raw)), np.zeros(len(raw))))
    scene = TriangleScene(device, geometry(centers, TILT))
    res, gpu = timed(device, kernel, len(raw) - 8,
                     {"scene": scene.tlas.structure, "windowSize": window, "pairAddress": int(paired)})
    return res, gpu, (perf_counter() - t0) * 1000, scene.metrics.get("as_bytes")


def scan_run(device, kernel, raw, window, paired):
    t0 = perf_counter()
    buf = gpu_buffer(device, np.frombuffer(raw, np.uint8).astype(np.uint32))
    res, gpu = timed(device, kernel, len(raw) - 8, {"bytes": buf, "windowSize": window, "pairAddress": int(paired)})
    return res, gpu, (perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("books", type=Path)
    ap.add_argument("report", type=Path)
    a = ap.parse_args()
    device, _ = open_kernel("episodic.slang")
    rt_k = device.create_compute_kernel(device.load_program("episodic.slang", ["indexedTrace"]))
    sc_k = device.create_compute_kernel(device.load_program("episodic_scan.slang", ["scan"]))
    report = {"gpu": getattr(device.info, "adapter_name", "NVIDIA RTX (see stdout)"), "platform": platform.platform(), "repeats": REPEATS, "runs": {}}
    for name in BOOKS:
        raw = body(a.books, name)
        n = len(raw) - 8
        check_n = n if name == "treasure-120" else 65_536
        for paired in (False, True):
            for window in (256, 2048):
                key = f"{name}_{'pair' if paired else 'one'}_{window}"
                rt_res, _, _, as_bytes = rt_run(device, rt_k, raw, window, paired)        # warm-up
                sc_res, _, _ = scan_run(device, sc_k, raw, window, paired)
                np.testing.assert_array_equal(rt_res[:, :3], sc_res[:, :3])
                np.testing.assert_array_equal(sc_res[:check_n, :3], oracle(raw, window, paired, check_n))
                rt_g, rt_w, sc_g, sc_w = [], [], [], []
                for _ in range(REPEATS):
                    r, g, w, _ = rt_run(device, rt_k, raw, window, paired)
                    assert np.array_equal(r[:, :3], rt_res[:, :3])
                    rt_g.append(g); rt_w.append(w)
                    s, g, w = scan_run(device, sc_k, raw, window, paired)
                    assert np.array_equal(s[:, :3], sc_res[:, :3])
                    sc_g.append(g); sc_w.append(w)
                med = {k: float(np.median(v)) for k, v in
                       (("rt_gpu_ms", rt_g), ("rt_wall_ms", rt_w), ("scan_gpu_ms", sc_g), ("scan_wall_ms", sc_w))}
                report["runs"][key] = {
                    "targets": n, "oracle_checked_targets": check_n, "identical_outputs": True,
                    "mean_candidates_scan": float(sc_res[:, 3].mean()), "mean_ray_calls_rt": float(rt_res[:, 3].mean()),
                    "rt_as_bytes": as_bytes, **med,
                    "gpu_speedup_rt_over_scan": med["scan_gpu_ms"] / med["rt_gpu_ms"],
                    "wall_speedup_rt_over_scan": med["scan_wall_ms"] / med["rt_wall_ms"]}
                print(key, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                       for k, v in report["runs"][key].items()}), flush=True)
    hot = [report["runs"][f"{b}_pair_2048"] for b in BOOKS]
    report["rt_speed_advantage"] = all(r["gpu_speedup_rt_over_scan"] >= 1.5 and r["wall_speedup_rt_over_scan"] >= 1.5
                                       for r in hot)
    a.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
