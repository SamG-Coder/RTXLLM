# E12c — full-disc word beam vs matched self-inclusive smoothers (issue #83) — on par, not better

Claude. ADR 0003 class **T** test. E12 (#53) used an *annular* beam that never read the word's own
dish. The discrete class bigram that beat it (#57) pools the word's **own** counts with its class,
so that comparison was not matched. Here the beam covers the full disc, so the own dish can be hit
(at the beam centre, area ∝ evidence, subject to occlusion). **Every** comparator may also include
self. Data, vocabulary, layout, dishes, KN and the 512-ray beam are unchanged from E12.

```
python experiments/e12c_full_beam/run.py val  BOOK_DIR    # selects every family on validation
python experiments/e12c_full_beam/run.py test BOOK_DIR    # reads results/val.json, scores test once
```
The runner was committed (`ad79d57`) before validation and test were run. Selections were posted on #83 before test.

## Validation selections (`results/val.json`; KN 83.69)
| family (grid) | selected | val perplexity |
|---|---|---:|
| full-disc beam (σ × λ) | σ 1.0, λ 0.2 | 80.67 |
| class bigram, high-D features (K × λ) | K 100, λ 0.2 | 80.51 |
| class bigram, 3-D layout coordinates | K 100, λ 0.2 | 81.01 |
| Gaussian kernel with self, same layout (σ × λ) | σ 1.0, λ 0.3 | 79.88 |
| kNN with self, same layout (k × λ) | k 10, λ 0.2 | 80.88 |

## Sealed test (`results/test.json`, run once)
| model | perplexity | unseen bigrams (20.5%) | seen |
|---|---:|---:|---:|
| modified-KN word bigram | 82.26 | 2,431 | 34.32 |
| **full-disc beam (RTX)** | **79.58 (−3.3%)** | 2,089 | 34.23 |
| E12 annular beam (#53, for reference) | 80.31 | 2,075 | 34.69 |
| class bigram, high-D | 79.45 | 2,009 | 34.51 |
| class bigram, 3-D layout | 79.65 | 2,022 | 34.56 |
| **Gaussian kernel with self** | **78.93** | 2,174 | 33.53 |
| kNN with self | 79.59 | 2,055 | 34.38 |
| *intervention:* same beam, random layout | 85.16 (+7.0%) | 2,309 | 36.32 |
| *intervention:* same rays, permuted hit→payload map | 91.90 (+15.5%) | 2,479 | 39.25 |

RTX = CPU brute force on 32,768 sampled rays in every scene; 862,720 rays per beam batch, 0.05 ms
GPU dispatch. 0.7% of rays hit the word's own dish.

**Pre-registered criterion not met.** It required the beam to be ≥ 0.5% below every matched
comparator *and* both interventions to be ≥ 2% worse. The interventions pass (+7.0%, +15.5%), but
the beam is **0.8% worse** than the self-inclusive Gaussian kernel. It is on par with the class
bigrams and kNN (within 0.2%).

## Interpretation
- Most of E12's gap to the class bigram came from **leaving out self**, not from traversal or
  3-D: with self allowed, the beam improves from 80.31 to 79.58 and the 3-D class model (79.65)
  is close to the high-D one (79.45).
- Traversal does **causal** work (ADR 0003 b): scrambling the layout or the hit-to-payload routing
  hurts badly. But on matched comparison (ADR 0003 c) it is a class-based smoother that a dense
  Gaussian kernel over the same coordinates slightly beats. Occlusion and finite ray sampling
  appear to cost a little rather than help.
- **Post-hoc, validation only** (`posthoc_occlusion.py REPO_ROOT BOOK_DIR`, run after the sealed test;
  no test numbers): nearest-hit occlusion *costs* accuracy. With transmission γ per dish crossed
  (exact CPU all-hit lists; γ = 0 reproduces the RTX closest-hit numbers exactly), validation
  perplexity at λ 0.2 is 80.67 (γ 0), 80.50 (0.25), 80.37 (0.5), 80.25 (1.0); the best is 80.14 at
  γ 1, λ 0.3. That approaches the self-inclusive kernel (79.88). Ray count does not limit it: 512
  vs 2,048 rays give 80.67 vs 80.65. So the closest-hit selection that makes this T-class is what
  separates it from the kernel.
- One book, one layout seed, bigram only. No traversal advantage is claimed.
