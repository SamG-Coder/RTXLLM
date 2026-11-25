# M7 — RT-core episodic search vs a plain compute scan (issue #91) — negative

Claude. The speed half of ADR 0003 (c). E9/E10 episodic search is a **T** stage: backward rays find
earlier occurrences of the 1- or 2-byte key and transverse rays extend the match. Here the same
search semantics (≤ 16 candidate endings, match length ≤ 8, nearest first, stop at 8) are
implemented as a **plain backward scan** in a compute shader (`src/rtx/episodic_scan.slang`; no
acceleration structure, no ray queries), on the same RTX 5080 and Vulkan device.

```
python experiments/m7_search_speed/run.py BOOK_DIRECTORY results/report.json
```
The code was committed (`29f4da7`) before any timing was measured. Both paths' outputs (recent,
prediction, length) are **bit-identical** to each other on every target, and to the E10 CPU oracle
on all 376,884 Treasure Island targets and the first 65,536 Moby Dick targets. Timings are medians
of 7 warm repeats after a warm-up. GPU = timestamp interval around the search dispatch. Wall: RT =
acceleration-structure build + dispatch + readback; scan = byte upload + dispatch + readback.

| book, key, window | RT GPU ms | scan GPU ms | RT wall ms | scan wall ms | RT speed vs scan (GPU / wall) |
|---|---:|---:|---:|---:|---:|
| Treasure, 1 byte, 256 | 0.94 | 0.14 | 42.1 | 25.2 | 0.15× / 0.60× |
| Treasure, 1 byte, 2,048 | 1.29 | 0.58 | 43.5 | 25.4 | 0.45× / 0.58× |
| Treasure, pair, 256 | 1.27 | 0.12 | 43.6 | 26.2 | 0.09× / 0.60× |
| **Treasure, pair, 2,048** | **3.46** | **0.76** | **51.1** | **28.0** | **0.22× / 0.55×** |
| Moby, 1 byte, 256 | 5.11 | 0.44 | 182.5 | 87.2 | 0.09× / 0.48× |
| Moby, 1 byte, 2,048 | 6.04 | 1.77 | 174.4 | 85.0 | 0.29× / 0.49× |
| Moby, pair, 256 | 3.99 | 0.34 | 149.7 | 79.4 | 0.09× / 0.53× |
| **Moby, pair, 2,048** | **10.50** | **2.34** | **155.8** | **81.7** | **0.22× / 0.52×** |

The RT path also needs a 22.8 MB (Treasure) or 75.9 MB (Moby) acceleration structure. The scan
needs the raw bytes (stored here as uint32, 4 bytes each).

**Pre-registered criterion not met.** It required the RT path to be ≥ 1.5× faster on both GPU and
wall time for pair/2,048 on both books. Instead, the plain scan is **4.5× faster on GPU and 1.8–1.9×
end to end**.

## Interpretation
- A window of ≤ 2,048 bytes is a tiny, contiguous, cache-friendly search space. A linear scan
  touches ~2K bytes per target with coherent memory access. The RT path issues ~15–44 dependent ray
  queries per target (`mean_ray_calls_rt` in the report), each a BVH traversal, plus an AS build
  per stream.
- The RT-core search is correct and causally meaningful, but on this hardware it is not a faster
  way to compute E10's predictions. Any speed claim for traversal would need a workload where
  spatial structure prunes a search space a scan cannot cover cheaply. This one is not such a
  workload.
- Limits: one GPU and driver; Python host overhead is included in wall time for both paths alike;
  the scan is straightforward, not hand-tuned (shared-memory tiling would likely widen the gap).
