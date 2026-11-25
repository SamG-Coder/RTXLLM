# #47 — scaling and profiling the RTX layered-backoff scene (Milestone 6)

Claude. Scene and decoder from #36 (`src/rtx/backoff.py`); KN-6 trained on 1, 2 and 4 pinned books
(each book a separate document); scored on all 346,818 targets of the pinned War of the Worlds from
offset 8 (the #38 test book; nothing is selected here). Median of 5 timed repeats after a warm-up.

```
python experiments/m6_scale_backoff/run.py BOOK_DIR      # writes results.json
```

## Results (RTX 5080 / 616.64, AMD Ryzen 7 9800X3D, Python 3.12, NumPy 2.5.5)
Parity is exact at every size: RTX dish hits = CPU suffix contexts; RTX bpb = NumPy KN-6
(max |Δp| = 0.0).

| training | bytes | dishes | triangles | AS bytes | payload bytes | scene file | bpb |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alice | 153,858 | 63,806 | 127,612 | 9.0 MB | 2.7 MB | 1.2 MB | 2.684 |
| + Pride | 858,870 | 132,729 | 265,458 | 17.5 MB | 6.1 MB | 2.7 MB | 2.401 |
| + Frankenstein, Dracula | 2,153,264 | 224,834 | 449,668 | 28.9 MB | 10.6 MB | 4.7 MB | **2.130** |

| training | KN fit | scene write | load + AS build | GPU search dispatch | search host wall | CPU payload fold | CPU sorted-key suffix search |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 book | 0.05 s | 0.33 s | 0.23 s | 0.171 ms | 40 ms | 58 ms | 65 ms |
| 2 books | 0.20 s | 0.82 s | 0.28 s | 0.174 ms | 41 ms | 66 ms | 74 ms |
| 4 books | 0.46 s | 1.46 s | 0.31 s | 0.180 ms | 43 ms | 67 ms | 79 ms |

## Findings
- **Traversal is not the bottleneck.** One dispatch answers all 346,818 queries (about 2.0M
  ray–layer intersections) in 0.17–0.18 ms, about 2×10⁹ rays/s. It is nearly flat while the scene
  grows 3.5× (BVH scaling).
- **Host transfer/synchronisation dominates search** (~40 ms, >99% of search wall time: per-call
  buffer creation, upload, sync and readback), and the **CPU payload fold** costs more than the
  search (58–67 ms).
- **Honest speed comparison:** the GPU dispatch alone is ~400× faster than a NumPy sorted-key suffix
  search over the same positions (65–79 ms). End to end with the current host path, RTX search plus
  CPU fold takes 0.10–0.11 s against 0.16–0.19 s for full NumPy KN scoring (single-threaded NumPy,
  1.6–1.8× faster). That margin is modest and would shrink against an optimised multi-threaded CPU
  index, so no general speed advantage is claimed.
- Memory grows linearly with distinct contexts; acceleration structures cost 1.8–2.0× the vertex bytes.
- Next steps (not in this PR): persistent device buffers, and the GPU payload fold from #41, which
  should remove both host costs.
