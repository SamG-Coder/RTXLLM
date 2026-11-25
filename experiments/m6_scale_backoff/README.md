# #47 — scaling and profiling the RTX layered-backoff scene (Milestone 6)

Claude. Scene and decoder from #36 (`src/rtx/backoff.py`); KN-6 trained on 1, 2 and 4 pinned books
(each book a separate document); scored on all **346,826** targets of the pinned War of the Worlds
from offset 8 (the #38 test book; nothing is selected here). Inference timings (encoding, search,
fold, end to end, NumPy KN, CPU index) are medians of 5 repeats after a warm-up, with GPU dispatch
times aggregated the same way (series in `results.json`); training/build timings (KN fit, scene write,
load + AS build) are single observations.

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

Build costs (single observations): KN fit 0.05 / 0.20 / 0.46 s, scene write 0.33 / 0.82 / 1.46 s,
load + AS build 0.23 / 0.28 / 0.31 s.

**Per-stage timings, all 346,826 targets:**

| training | encode (bytes → ray origins) | GPU search dispatch | search host wall | CPU payload fold | **end to end, bytes → probabilities** | NumPy KN-6, full | CPU suffix search, index prebuilt |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 book | 26 ms | 0.171 ms | 35 ms | 54 ms | **118 ms** | 168 ms | 51 ms |
| 2 books | 24 ms | 0.175 ms | 37 ms | 59 ms | **124 ms** | 164 ms | 52 ms |
| 4 books | 24 ms | 0.179 ms | 36 ms | 60 ms | **122 ms** | 171 ms | 49 ms |

Work per dispatch: 2.0M dish hits and 2.01–2.07M ray segments traced (one trace per hit plus one
terminating miss unless all six layers are hit). That is 1.9–2.0×10⁹ queries/s and about 1.15×10¹⁰
ray segments/s at dispatch level.

## Findings
- **Traversal is not the bottleneck.** One dispatch answers all queries in 0.17–0.18 ms, nearly flat
  while the scene grows 3.5×.
- **Host costs dominate:** the search wall (~36 ms) is >99% buffer creation, upload, sync and
  readback; the CPU payload fold (54–60 ms) and the query encoding (24–26 ms) each cost more than
  traversal.
- **Honest speed comparison:** the directly timed bytes-to-probabilities path (encoding + RTX search +
  CPU fold) takes 118–124 ms vs 164–171 ms for single-threaded NumPy KN-6 scoring, i.e. **1.33–1.42×**.
  Against a CPU suffix search with its index prebuilt (49–52 ms), the dispatch alone is 270–300× faster,
  but the end-to-end path is not. No general speed advantage is claimed: an optimised multi-threaded
  CPU index would narrow or reverse the margin.
- Memory grows linearly with distinct contexts; acceleration structures cost 1.8–2.0× the vertex bytes.
- Next steps (not in this PR): persistent device buffers, GPU-side encoding, and the GPU payload fold
  from #41, which together would remove the host costs measured here.
