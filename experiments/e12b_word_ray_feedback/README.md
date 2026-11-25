# E12b — learning the word layout from ray-traced prediction error (issue #59) — negative

Claude. Combines E3b (#48: a fact layout learned from ray-traced error) with E12 (#53: beams over
a word layout beat a KN word bigram). Question: can the scene learn the **word** layout from
ray-traced next-word evidence, starting from a random layout, and keep E12's gain over KN?

## Setup
Everything from E12 is reused unchanged: pinned Pride and Prejudice, lowercase tokens, contiguous
70/15/15 split, 3,338-word vocabulary, 1,685 dish-owning words, hexagonal dishes with area ∝
evidence, the modified-KN word bigram, and the frozen E12 readout (annular 512-ray beam, σ 1.0,
λ 0.1). Only the layout differs.

**Learning rule.** Start from a random lattice. Each round, rebuild the dish scene at the current
layout and fire an annular beam on RTX from every dish-owning word w (256 rays, never its own dish).
Each ray that hits neighbour w′ earns a credit from w's **training** successors only:
- `gain`: Σ_v c(w, v)·[P_w′(v) − P_uni(v)]₊ / c(w) — does w′'s stored next-word distribution
  predict w's successors better than chance;
- `llr`: mean log-likelihood ratio of w's successors under w′'s (smoothed) distribution vs chance,
  centred on the mean over that beam's hits and clipped at 0.

Credits accumulate as springs w–w′; inverse-square repulsion; rescale to median nearest-neighbour
distance 1; beam width annealed 6 → 2.5. `--topk K` lets the springs use only each word's K
strongest accumulated ray credits (E12's force layout used a 10-nearest-neighbour graph); the
credits themselves still come only from ray hits. **Shuffled-credit control:** credits are permuted
among the rays that hit a dish, so the layout gets the same amount of spring force with the
word-specific signal destroyed.

```
python experiments/e12b_word_ray_feedback/run.py val  BOOK_DIR --rounds 120 --credit llr --topk 10 --tag llr_top10
python experiments/e12b_word_ray_feedback/run.py test BOOK_DIR --rounds 120 --eval-round 120 --credit llr --topk 10 --tag llr_top10
```

## Validation (selection; `results/val_*.json`, logs alongside)
KN 83.69; E12 force layout (CPU, Hellinger 10-NN springs) 81.59; random layout (round 0) 86.53.

| credit rule | best feedback (round) | shuffled at the same round |
|---|---:|---:|
| `gain`, 60 rounds | 85.71 (40) | 86.51 |
| `llr`, 30 rounds (stopped at the owner's pause; `results/val_llr_partial.log`) | 85.15 (30) | — |
| **`llr` + top-10 springs, 120 rounds** | **84.57 (120)** | 86.73 |

Frozen on #59 before the test split was scored: `llr`, top-10, 120 rounds, seed 0.

## Sealed test (run once; `results/test_llr_top10_eval120.json`)
| layout (frozen E12 readout) | perplexity | unseen-bigram perplexity (20.5%) | seen |
|---|---:|---:|---:|
| modified-KN word bigram | 82.26 | 2,431 | 34.32 |
| **ray-feedback layout, 120 rounds (RTX)** | **82.86 (+0.7%)** | **2,198 (−9.6%)** | 35.55 |
| shuffled-credit layout, 120 rounds | 85.19 (+3.6%) | 2,295 (−5.6%) | 36.40 |
| *reference:* E12 CPU force layout (#53) | 80.31 (−2.4%) | 2,075 (−14.7%) | 34.69 |
| *reference:* E12 random layout (#53) | 84.92 (+3.2%) | 2,261 | 36.40 |

Criterion checks: overall ≥ 1.5% below KN — **fail** (+0.7%); unseen ≥ 8% below KN — pass (−9.6%);
better than shuffled on both — pass. Overall: **not met**.

## Interpretation
- **Pre-registered success criterion not met.** It required test perplexity ≥ 1.5% below KN and
  unseen-bigram perplexity ≥ 8% below KN, and beating shuffled credit on both.
- The ray-traced credit **does carry signal**: feedback improves steadily from the random start while
  shuffled credit does not (validation 84.57 vs 86.73; test 82.86 vs 85.19), and on test it clears
  the unseen-bigram bar (−9.6%). But overall it remains worse than KN because seen-bigram
  perplexity is worse (test 35.55 vs 34.32). The neighbours it finds are not similar enough
  for the λ = 0.1 mixture to pay for itself.
- Unlike E3b, where synthetic facts had a clean latent structure, the word layout learned from
  local beam hits recovers only part of the CPU force layout's quality (81.59) in 120 rounds. A
  beam only discovers words already near it, so partners that start far apart are rarely found.
  Learning is still slowly improving at round 120; a longer schedule might narrow the gap but on
  this trend would not reach the bar.
- One book, one seed, and three credit rules tried on validation. This is a negative result for
  learning the word layout from ray hits under this rule and budget. It is not a negative result
  for E12 itself, whose layout came from the CPU similarity graph.
