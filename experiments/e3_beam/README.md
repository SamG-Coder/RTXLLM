# E3 — shared geometry: answering never-written facts with beams (issue #11)

Claude. Contract: `docs/decisions/0002-beams-channels-dishes.md` (proposed).

## Question
E1 and the byte pilots are lookup-equivalent: every query only reaches geometry written for exactly
that query. Can **shared geometry** — a finite beam that meets mirrors written by *neighbouring*
stations — answer facts that were **never written**, and does that survive composition?

## World and protocol
Per seed (0, 1, 2): 512 entities in 32 latent clusters of 16, 6 functional relations. Each
(cluster, relation) has a prototype object; an entity follows it with probability 0.8, else a random
object. 20% of the 3,072 facts are **held out** (never written), split into val/test halves (307 each);
2,458 facts become geometry. Two-hop queries `(s, r1, r2)` start with a held-out link and continue
through a written one (300 per split). Latent labels are used only by the oracle layout.

Placement (median nearest-neighbour distance normalised to 1):
`random` lattice (control); `force` layout learned **only from written facts** (springs between
stations sharing ≥ 2 written (relation, object) pairs + inverse-square repulsion); `oracle` (latent
clusters; upper bound, never selected). *Deviation from the issue text:* the issue said spectral
placement; it was replaced by the force layout during development, before the formal sweep, because
3 Laplacian eigenvectors cannot separate 32 clusters. No spectral results exist.

Geometry: receivers r = 0.075; facts are **focusing dishes** (facets of circumradius 0.05 tiling a
disk, each aimed from its own centroid at the object), relation channels (emission rays see only
their relation's facts; transport rays see only receivers). Beam = core ray (energy 0.5) + 256
sunflower halo rays of radius `width`, hop-synchronous merge (ADR 0002).

Selection rule (fixed before the 3-seed sweep): maximise mean val held-out accuracy on the force
layout subject to written-fact accuracy ≥ 0.90. Before that sweep, exploratory configurations were
run on seed-0 validation while developing the dish. Two-hop accuracy is not used for selection.

```
python experiments/e3_beam/run.py sweep
python experiments/e3_beam/run.py test --dish 0.3 --width 3.0 --knn 10 --backend rtx
python experiments/e3_beam/rtx_check.py --dish 0.3 --width 3.0 --split val
```

## Validation sweep (force layout, mean of 3 seeds)
Layout quality (fraction of 5 nearest neighbours in the same latent cluster): random 0.03, force
0.97–0.98, oracle 1.00.

| dish radius \ beam width | 0 (pencil) | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | written acc |
|---|---|---|---|---|---|---|---|
| none (single triangle) | 0.000 | 0.092 | 0.177 | 0.218 | 0.192 | 0.129 | 0.984 |
| 0.15 | — | 0.211 | 0.512 | 0.634 | 0.710 | 0.696 | 0.979 |
| 0.30 | — | 0.306 | 0.544 | 0.669 | 0.718 | **0.733** | 0.962 |
| 0.50 | — | 0.417 | 0.558 | 0.637 | 0.682 | 0.697 | 0.933 |

kNN vote on the same force layout (lookup for written facts, k nearest written neighbours for
held-out): k = 1 / 3 / 5 / 10 / 20 → 0.552 / 0.678 / 0.758 / **0.771** / 0.719.

Selected: dish 0.3, width 3.0; kNN k = 10.

Observations: without dishes a flat mirror catches almost none of a wide beam (≥ 64% abstentions);
focusing dishes decouple capture area from receiver size and raise held-out accuracy from 0.22 to
0.73. Larger dishes cost memorisation (written 0.98 → 0.93): a written exception is out-voted by its
neighbours.

## Sealed test (RTX backend, CPU parity re-checked in-run)
Selected configuration, run once on the test split (`results/test_dish0.3_w3.0_k10_rtx.json`).
Seeds 0 / 1 / 2; 307 held-out facts and 300 two-hop queries per seed; written = 300 sampled
written facts. Chance ≈ 0.002.

| method | held-out facts | two-hop (first link held out) | written facts |
|---|---|---|---|
| **beam, force layout** (learned from written facts) | **0.726 / 0.704 / 0.733** | **0.673 / 0.703 / 0.710** | 0.970 / 0.953 / 0.963 |
| beam, oracle layout | 0.769 / 0.752 / 0.785 | 0.757 / 0.763 / 0.787 | 0.953 / 0.970 / 0.970 |
| beam, random layout | 0.026 / 0.023 / 0.016 | 0.043 / 0.053 / 0.043 | 0.823 / 0.813 / 0.823 |
| pencil (width 0), force layout | 0.016 / 0.016 / 0.016 | 0.013 / 0.013 / 0.013 | 0.983 / 0.963 / 0.970 |
| kNN (k = 10), force layout | 0.765 / 0.779 / 0.788 | 0.770 / 0.787 / 0.820 | 1.000 |
| kNN, oracle layout | 0.814 / 0.827 / 0.818 | 0.833 / 0.833 / 0.843 | 1.000 |
| kNN, random layout | 0.023 / 0.016 / 0.029 | 0.037 / 0.027 / 0.027 | 1.000 |
| cluster majority (latent labels; ceiling) | 0.814 / 0.827 / 0.818 | — | — |
| symbolic graph over written facts | 0.000 | — | — |

**Acceptance: 12/12 pre-registered checks pass** (beam on the force layout ≥ 0.5 held-out, ≥ 0.4
lift over the random layout and over the pencil beam, two-hop ≥ 0.4, every seed).
Written exceptions (entities that do not follow their cluster) are still recalled 0.94–0.97 by the
force-layout beam. Held-out beams abstain 3–6% and lose ~92% of their energy to misses.

**RTX/CPU parity** (54 re-checked sets: every single-hop set in full, first 50 two-hop queries):
predictions agree 100% in every set. The force layout (primary) is exact in every set. In the
dense oracle layout one ray out of 78,899 differs per held-out set (a facet-edge decision: one
backend hits, the other misses; energy 1/512), and the random layout's two-hop sets differ by
≤ 2e-5 in answer energy.

## Conclusions
1. **Shared geometry generalises.** Facts that were never written are answered at 0.70–0.73
   (vs 0.02 for the same beams over a random layout and 0.016 for pencil rays), and the inferred
   link composes over a second bounce at 0.67–0.71. The only difference between the conditions is
   *where* the information sits in space.
2. **The layout matters as much as the beam**: the force layout learned from written facts recovers
   97–98% cluster purity and gets within ~0.05 of the oracle layout.
3. **The physical beam is a slightly worse retriever than kNN in the same space** (−0.04 to
   −0.06 held-out), and sharing costs some memorisation (written 0.95–0.97 vs 1.00). Its footprint
   is a fixed-radius cylinder rather than an adaptive k, it loses energy to misses, and neighbours
   occlude each other. No advantage over vector retrieval is claimed; the result is that ray
   traversal over learned geometry *implements* neighbourhood generalisation, and composes it
   across bounces.
4. Mirror-only (no dish) geometry fails at this task: a wide beam needs focusing capture area
   decoupled from receiver size.

## Engineering notes
- RTX beam backend (`src/rtx/beam.py`, `beam.slang`): one instance per relation channel (mask bit
  `1 << (1 + r)`) plus receivers (0x1); one dispatch per hop, one thread per ray; exact per-ray CPU
  parity (100% landing agreement, answers/lost/mirror-hit energy identical).
- CPU reference has an all-pairs mode and an exact, memory-bounded broadphase, tested identical.
