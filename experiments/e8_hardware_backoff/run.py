"""#32: layered-dish KN-6 byte model traced on RTX; parity and bounce-budget curve.

    python experiments/e8_hardware_backoff/run.py BOOK_DIR

BOOK_DIR holds rtxllm-pride-1342.txt and rtxllm-dracula-345.txt (pinned GITenberg
raw files, SHA256-checked). Frozen protocol in issue #32; no tunable settings.
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
from backoff import LayeredScene, open_backoff_kernel  # noqa: E402

from rtxllm.baselines.ngram import KneserNey  # noqa: E402

ORDER = 6
BOOKS = {
    "pride-1342": ("48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163", 5),
    "dracula-345": ("c802711470ddb2aaca81432fc101ed29c154e7c3d32df4172478695c33f68eae", 8),
}


def splits(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"digest mismatch: {path.name}")
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1]
    d = np.frombuffer(body, np.uint8).astype(np.int64)
    return np.split(d, [len(d) * 70 // 100, len(d) * 85 // 100])


def main():
    folder = Path(sys.argv[1])
    device, kernel = open_backoff_kernel()
    report = {"order": ORDER, "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                                      "platform": platform.platform()}, "books": {}}
    for name, (digest, start) in BOOKS.items():
        tr, va, te = splits(folder / f"rtxllm-{name}.txt", digest)
        t0 = perf_counter()
        kn = KneserNey(ORDER).fit([tr])
        fit_s = perf_counter() - t0
        scene = LayeredScene(device, kn)
        book = {"train_bytes": len(tr), "fit_s": fit_s, **scene.metrics, "splits": {}}
        for split, data in (("val", va), ("test", te)):
            idx = np.arange(start, len(data))
            pts = scene.points(data, idx)
            t0 = perf_counter()
            dishes, gpu_ms = scene.trace(kernel, pts)
            wall = perf_counter() - t0
            exp = scene.expected_dishes(data, idx)
            p = scene.target_prob(dishes, data[idx])
            ref = kn.prob(data, start=start)
            row = {"scored": len(idx), "rtx_bpb": float(-np.log2(p).mean()),
                   "numpy_kn_bpb": float(-np.log2(ref).mean()),
                   "dish_parity": bool(np.array_equal(dishes, exp)),
                   "max_abs_dp": float(np.abs(p - ref).max()),
                   "gpu_dispatch_ms": gpu_ms, "host_trace_wall_s": wall,
                   "layers_hit_mean": float((dishes >= 0).sum(1).mean()),
                   "budget_curve": {}}
            for b in range(1, ORDER + 1):
                d_b, _ = scene.trace(kernel, pts, budget=b)
                L = ORDER
                exp_b = exp.copy()
                exp_b[:, :L - b] = -1                    # top layers are above the ray's start
                p_b = scene.target_prob(d_b, data[idx], budget=b)
                sep = KneserNey(b).fit([tr]).prob(data, start=start)
                row["budget_curve"][b] = {"rtx_bpb": float(-np.log2(p_b).mean()),
                                          "dish_parity": bool(np.array_equal(d_b, exp_b)),
                                          "separately_fit_kn_bpb": float(-np.log2(sep).mean())}
            book["splits"][split] = row
            print(name, split, json.dumps({k: v for k, v in row.items() if k != "budget_curve"}), flush=True)
            print("  bounces -> bpb:", {b: round(v["rtx_bpb"], 4) for b, v in row["budget_curve"].items()},
                  "parity:", all(v["dish_parity"] for v in row["budget_curve"].values()), flush=True)
        report["books"][name] = book
    ok = all(s["dish_parity"] and abs(s["rtx_bpb"] - s["numpy_kn_bpb"]) < 1e-6
             and all(v["dish_parity"] for v in s["budget_curve"].values())
             for b in report["books"].values() for s in b["splits"].values())
    report["passes"] = bool(ok)
    print("passes:", ok)
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
