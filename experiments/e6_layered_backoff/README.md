# #23 pilot — physical backoff by layered dishes (validation only; negative)

Claude. Pride and Prejudice as pinned in #10 (SHA256 `48e05228…`), contiguous 70/15/15, scored
from offset 2; **only the validation split is read**; the test split is untouched.

```
python experiments/e6_layered_backoff/pilot.py PRIDE.txt     # writes pilot_val.json
```

## Strong count baseline: interpolated modified Kneser–Ney (`rtxllm.baselines.ngram`)
| n-gram order | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| val bpb | 4.567 | 3.422 | 2.677 | 2.096 | 1.852 | 1.774 | **1.759** | 1.762 |

For reference on the same validation split: 20%-interpolated "exact trigram" 2.7585 (#10),
neural w8 2.6477 (#16), best optical model 2.8829 (#12).

## Geometry (`rtxllm.learning.byte_layers.LayeredBackoff`)
Byte → cell of a 16×16 grid. An order-m context is a square dish on layer m at the dyadic position
`Σ_{j≤m} 16^{-j} g(b_{t-j})` (exact in float32 to order 5), reflectance `1 − γ_KN(context)`, payload
the KN-discounted continuation distribution; transmitted light continues to the next (shorter
context) layer and finally a uniform backstop. A query ray is aimed strictly inside every suffix
cell (never on an edge or triangle diagonal).

- **Pencil ray = interpolated KN exactly** (max |Δp| = 0.0 at orders 3–5, for identity, random and
  learned grids). The transmission chain through layers is KN's interpolation.
- **Beam** = core + rings of 8 offset rays (radii 0.5–4 finest cells, core 0.5–0.95): offset rays
  land in dishes of contexts whose oldest byte(s) have neighbouring grid cells.

## Result: beams do not beat KN
Best overall: order 5, random grid, radius 0.5, core 0.95: **+0.0012 bpb** (worse).
Best on contexts seen ≤ 2 times: order 4, similarity grid, radius 0.5, core 0.7: **−0.0025 bpb**
on 1,162 rare positions, while +0.028 overall. With the learned (similarity) grid the sign on
rare/unseen contexts is consistently slightly favourable and with a random grid slightly
unfavourable, but the size is negligible. Full grid in `pilot_val.json`.

Why: 98% of order-5 validation contexts occur in training, so there is little headroom; and
substituting a grid-neighbour byte is a worse backoff than KN's own (drop the oldest byte) — a 2-D
grid encodes little byte similarity. The preregistration draft in #23 (≥ 0.02 overall or ≥ 0.05 on
rare) cannot be met, so the full test run was not executed.

## What remains useful
- The KN baseline (now the strongest count reference in the repo).
- The layered scene is a ray-traceable byte LM that reproduces KN exactly — a possible strong
  RTX-native reference model (its RTX realisation is not in this PR).
