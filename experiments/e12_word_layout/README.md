# E12 — word-level spatial class LM: beams over a learned word layout (issue #53)

Claude. Builds on E3 (#43): what made shared geometry generalise for synthetic facts (latent
similarity, missing pairs) is present for **word bigrams** and absent for bytes (#23 addendum).

## Setup
Pride and Prejudice (#10 pin), lowercase tokens `[a-z0-9]+|[^\sa-z0-9]`, contiguous 70/15/15
(107,916 / 23,126 / 23,126 tokens). Vocabulary = training words seen ≥ 2 times plus UNK (3,338).
Everything is fit on the training split only.

- **Baseline:** interpolated modified Kneser–Ney word bigram (D1/D2/D3+, continuation base).
- **Layout:** 1,685 words with ≥ 5 training continuations placed in 3-D by a force layout whose
  springs join each word to its 10 nearest neighbours by Hellinger similarity of right-context
  (next-word) distributions; random layout as control.
- **Geometry:** each such word owns a hexagonal dish along the single *next* direction, radius
  ∝ √evidence; its **hit-local payload** is its training next-word distribution.
- **Query:** an **annular** beam (512 rays, inner radius 1.05 × the word's own dish, so it never
  reads itself) fired on RTX from the previous word's station; each ray's first dish hit contributes
  that neighbour's payload, weighted by a Gaussian profile (σ, width 2.5σ).
  P(v | w) = (1 − λ)·KN(v | w) + λ·q_w(v).

```
python experiments/e12_word_layout/run.py val  BOOK_DIR
python experiments/e12_word_layout/run.py test BOOK_DIR --sigma 1.0 --lam 0.1 --knn 10
```

## Validation (selection; `results/val.json`)
KN: perplexity 83.69, 2,319 on bigrams unseen in training (20.6% of targets). Beam over the force
layout, best overall σ 1.0, λ 0.1: 81.59 (−2.5%), unseen 1,969 (−15%); the most aggressive setting
(σ 2.0, λ 0.3) reaches unseen 1,664 (−28%) at 83.05 overall. Random-layout beams are worse than KN
overall in every setting (86–99). kNN best 81.97; analytic kernel best 81.65. Frozen on #53: σ 1.0,
λ 0.1, kNN k = 10.

## Sealed test (run once; `results/test_s1.0_l0.1_k10.json`)
| model | perplexity | unseen-bigram perplexity (20.5%) | seen-bigram perplexity |
|---|---:|---:|---:|
| modified-KN word bigram | 82.26 | 2,431 | 34.32 |
| **beam, learned layout (RTX)** | **80.31 (−2.4%)** | **2,075 (−14.7%)** | 34.69 |
| kNN k = 10, same layout | 80.65 | 2,089 | 34.81 |
| analytic Gaussian kernel, same layout | 80.70 | 2,090 | 34.84 |
| beam, random layout | 84.92 (+3.2%) | 2,261 | 36.40 |
| kNN, random layout | 85.96 | 2,385 | 36.45 |

**Pre-registered success criterion met:** unseen-bigram perplexity 14.7% below KN (required ≥ 3%)
and overall perplexity 2.4% below KN (required not > 1% worse). RTX vs CPU brute-force parity on
32,768 sampled rays per layout: 100%. Cost: 862,720 rays (one beam per dish-owning word), 0.07–0.11
ms GPU dispatch; acceleration structure 0.6 MB.

## Interpretation
- Spatial structure carries **language** generalisation here: neighbours in a layout learned from
  next-word behaviour supply evidence for bigrams never seen in training (−15% perplexity on them),
  at a small cost on seen bigrams (34.32 → 34.69). The same machinery over a random layout hurts.
- The physical beam is **on par with kNN and the analytic kernel** in the same layout (80.31 vs
  80.65 / 80.70). That is consistent with E3 (#43): ray traversal *implements* neighbourhood
  smoothing; it does not beat vector retrieval. Area ∝ evidence and nearest-hit occlusion give a
  slightly better weighting here, by a margin too small to claim.
- This is a class-based-LM effect (as in classic word-clustering LMs) realised in RTX geometry. It is
  word-level and bigram-only; byte-level smoothing gives nothing (#23).
- One book; one seed of the layout; vocabulary and tokeniser are simple.
