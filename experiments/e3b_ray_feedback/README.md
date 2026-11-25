# E3b — the scene learns its own layout from ray-traced prediction error (issue #40)

Claude. Builds on E3 (#11 / #43): same clustered world, held-out facts, focusing dishes and frozen
inference beam (dish 0.3, width 3.0, core 0.5, 256 rays). The only change is **how station positions
are learned**.

## Learning rule (RTX in the loop)
Start from the random layout (E3 control). Each round:
1. Rebuild the scene at the current layout; every dish is re-aimed at its object's current position.
2. For every written fact (s, r, o), fire an **annular** beam from s along d_r on RTX (inner radius
   0.4 > dish radius 0.3, so s's own dish is never hit: a physical leave-one-out).
3. A halo ray that hits a neighbour's dish and reflects into receiver o (the *correct* answer) adds a
   spring between s and that neighbour. Wrong or missed rays add nothing. (Which station wrote a dish
   is training-time bookkeeping only; inference never uses it.)
4. Springs plus inverse-square repulsion move the stations (10 steps), rescale to median nearest-
   neighbour distance 1, repeat. The beam width anneals from wide (exploration) to narrow.

**Shuffled-credit control:** identical rays, but the correct/incorrect flags are permuted among the
rays that hit a dish, which removes the error signal and keeps everything else.

```
python experiments/e3b_ray_feedback/run.py val --rounds 80 --decay 1.0 --w-start 10 --w-end 4 --learn-rays 256 --tag wide
python experiments/e3b_ray_feedback/run.py test --rounds 80 --eval-round 30 --decay 1.0 --w-start 10 --w-end 4 --learn-rays 256 --tag wide
```

## Validation (all configurations tried; logs in `results/`)
| configuration | rounds | held-out (seeds 0/1/2) | shuffled credit | layout purity |
|---|---|---|---|---|
| default: spring decay 0.9, width 6→3, 128 rays | 40 | 0.472 / 0.476 / 0.567 | 0.020 / 0.023 / 0.029 | 0.79–0.83 |
| seed-0 probe: no decay, width 6→4 | 80 | 0.554 (seed 0 only) | — | 0.84 |
| **wide: no decay, width 10→4 over 80 rounds, 256 rays** | 20 / **30** / 40 | mean 0.648 / **0.661** / 0.643 | 0.013–0.036 | 0.92–0.96 |

Frozen before the test (issue #40): *wide*, 80-round schedule, evaluate at round 30.
Why *wide* helps: with narrow beams early, a cluster splits into fragments that never see each other
again; wide early beams find distant partners, and keeping every discovered spring (no decay) lets
the fragments merge.

## Sealed test (run once; `results/test_sched80_eval30_wide.json`)
Seeds 0 / 1 / 2, round 30 of the frozen schedule, E3 inference beam on RTX.

| layout | held-out facts | two-hop | written | 5-NN cluster purity |
|---|---|---|---|---|
| random start (round 0) | 0.026 / 0.023 / 0.016 | 0.043 / 0.053 / 0.043 | 0.82 / 0.81 / 0.82 | 0.04 / 0.03 / 0.03 |
| **ray-feedback learned** | **0.606 / 0.651 / 0.642** | **0.610 / 0.633 / 0.653** | 0.97 / 0.97 / 0.96 | 0.94 / 0.96 / 0.95 |
| shuffled credit (same rays) | 0.020 / 0.020 / 0.026 | 0.037 / 0.050 / 0.037 | 0.72 / 0.76 / 0.72 | 0.04 / 0.03 / 0.03 |
| *reference: CPU force layout (E3, #43)* | *0.726 / 0.704 / 0.733* | *0.673 / 0.703 / 0.710* | *0.97 / 0.95 / 0.96* | *0.97–0.98* |

**Acceptance: 12/12 pre-registered checks pass** (held-out ≥ 0.6; ≥ 0.4 above the random start
and above shuffled credit; two-hop ≥ 0.4, on every seed). Seed 0 clears the 0.6 bar narrowly
(0.606).

Cost: 30 rounds × 629,248 learning rays (2,458 annular beams × 256 rays); 0.05–0.08 ms GPU dispatch
per round, 0.63–0.79 s wall per round (dominated by the NumPy scene rebuild and spring steps);
about 85–89k rays per run landed on the correct receiver and became springs.

## Conclusions
1. **The scene can learn where to put its own information from ray-traced error alone.** Starting
   from a random layout, crediting only rays that a neighbour's dish reflected into the right
   receiver drives cluster purity from 0.03 to 0.95 and held-out accuracy on never-written facts from
   0.02 to 0.61–0.65; composed two-hop queries follow (0.61–0.65).
2. **The signal is the error, not the rays**: shuffling which rays count as correct, with identical
   rays and updates otherwise, learns nothing (0.02–0.03).
3. It reaches about 85–90% of the accuracy of the CPU force layout, which reads written-fact
   co-occurrence directly; the gap is consistent with sampled, beam-local springs. No claim is made
   that this is a better *optimiser* than a symbolic layout: the point is that learning happens
   through traversal and reflection in the scene.
4. Exploration matters physically: narrow early beams fragment clusters; wide early beams plus
   persistent springs fix it (validation table).
