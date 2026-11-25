# #32 — hardware layered-backoff byte LM (KN-6 on RTX)

Claude. Frozen protocol in issue #32 (no tunable settings). `src/rtx/backoff.py`, `backoff.slang`.

```
python experiments/e8_hardware_backoff/run.py BOOK_DIR     # writes results.json
```

## Mechanism
Layer m (contexts of the last m bytes, m = 5..0) is a plane; each training context is a square dish
at its dyadic chaos-game position, so a context's dish contains the positions of all its extensions.
Each (layer, newest byte) is a separate instance whose transform zooms object space ×16, keeping
coordinates exact in float32 up to order 6. **One hardware ray per scored byte** travels through the
stack from the query's position (computed from the query's own bytes) and meets, by real
intersection, the dish of every seen suffix context, one hit per backoff level (at most 6 hits).
Each dish carries a **hit-local payload** written into the scene file: its transmission gamma (KN
backoff weight) and its sparse discounted emissions, with emissions + gamma = 1 per dish. Decoding
reads the reloaded scene file only, with no KN object and no training counts: the host folds the
hit dishes' payloads from the coarsest layer up, which is interpolated Kneser-Ney (tested: two
streams with identical context support but different counts give identical geometry and different
payloads, and the reloaded scenes reproduce KN exactly with normalised distributions). No count
table is consulted outside the hit primitives. Starting the ray below the top layers is the **bounce-budget** ablation.

## Results (RTX 5080 / 616.64, SlangPy 0.43.1; NumPy 2.5.5)
Exact parity on every split: hardware dish sequences = CPU suffix contexts on all layers for every
scored byte and every bounce budget; hardware bpb = NumPy KN-6 bpb (max |Δp| = 0.0). `passes: true`.

| book | split | scored bytes | RTX bpb = KN-6 | search: GPU dispatch | search: host wall | payload fold (host) |
|---|---|---:|---:|---:|---:|---:|
| Pride (#10 pin, from offset 5) | val | 105,747 | 1.7737 | 0.058 ms | 23 ms | 12 ms |
| | test | 105,747 | **1.7368** | 0.057 ms | 12 ms | 11 ms |
| Dracula (#25 pin, from offset 8) | val | 129,508 | 1.9367 | 0.073 ms | 151 ms | 21 ms |
| | test | 129,508 | **2.0171** | 0.071 ms | 14 ms | 19 ms |

The GPU dispatch time covers the suffix search only (all scored positions in one dispatch). Host
wall times include upload and readback; the payload fold runs on the CPU afterwards.

Scene (Pride / Dracula): 96,082 / 135,131 dishes; 192,164 / 270,262 triangles in 391 / 421
instances; vertices 6.9 / 9.7 MB; hit-local payloads 4.3 / 6.1 MB; compressed scene file
1.9 / 2.6 MB; acceleration structures 12.9 / 17.8 MB; build about 0.2 s.

**Bounce budget** (ray meets only the bottom b layers; test bpb; lower is better). Definition: the
ray is launched *below* the top 6 - b layers, i.e. the **oldest context bytes are dropped before
launch**; the remaining chain is folded exactly as KN (uniform base), so each capped distribution is
normalised. (The alternative, stopping transmission early and sending the residual mass to the
uniform backstop, is not what is measured here.) The capped chain's
top layer then carries continuation-count statistics, so it is worse than a separately fitted KN-b,
shown for reference:

| bounces b | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Pride, RTX capped | 5.362 | 4.159 | 3.404 | 2.697 | 2.246 | **1.737** |
| Pride, KN-b fitted separately | 4.587 | 3.417 | 2.660 | 2.068 | 1.816 | 1.737 |
| Dracula, RTX capped | 5.497 | 4.369 | 3.560 | 2.911 | 2.473 | **2.017** |
| Dracula, KN-b fitted separately | 4.519 | 3.488 | 2.774 | 2.285 | 2.082 | 2.017 |

Every additional bounce lowers loss, monotonically, on both books.

Query encoding uses only the query's own bytes (`points()`): an unseen top-order context's ray simply
finds no dish on that layer and continues to its seen suffix. This is tested, together with two
order-6 histories that differ only in a high-bit oldest byte selecting distinct primitives.

## What this is and is not
- It **is** a ray-traced byte LM whose prediction depends on multi-hit propagation through
  persistent spatial geometry, at KN-6 quality: far below every other optical model in the repo
  (best 2.79 bpb on Dracula test, #25) and below the fixed-budget neural w8 (2.61 on Pride test, #16).
- It **is not** a new learning method: by construction it equals interpolated KN-6, a count model.
  RTX supplies the suffix-context search through geometry; it adds no generalisation (#28 showed
  spatial smoothing over this layout does not beat KN).
- Float32 limits this layout to order 6 (KN-7 is 0.015 bpb better on Pride val).
