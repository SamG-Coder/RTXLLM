# Word generation from persistent emission geometry

**Traversal (T).** The learned layout and the first beam determine which neighbours contribute evidence. **Addressed storage (S).** The second stage reads a probability from an integer row and column. Its ray recovers the value encoded at that location.

The archive contains the E12 word dishes, launch positions and radii, the vocabulary and dish identities, plus a second scene of probability triangles. It is sufficient to score a document or generate a sequence in a fresh process. Every inference-time probability comes from triangle intersections and their weighted sum.

The first acceleration structure finds neighbouring word dishes with the 512-ray E12 annulus. On load, the host groups hits by dish identity and combines their Gaussian weights. This is exact because rays hitting the same dish lead to the same emission row. The GPU retains this list for each word. When scoring a candidate, each retained hit selects a row in the emission scene. The next ray's column selects the candidate word, and its hit distance gives the conditional probability mass. A missing triangle contributes zero. Other rows in the same scene encode discounted bigram mass, backoff weight and continuation unigram. The shader combines that prior with the weighted neighbour emissions. The model uses two dependent intersection stages with a cached first stage and spatially addressed ray launches. The final sampler draws from the returned vocabulary distribution on the CPU.

## Held-out results

The source experiment fits its layout and emissions on the first 70% of the pinned Pride and Prejudice word stream. E14 reproduces its frozen settings. Across the same 23,125 test transitions, 4,745 were unseen training bigrams.

| Predictor | Perplexity | Unseen-bigram perplexity |
| --- | ---: | ---: |
| Geometric modified Kneser–Ney word bigram | 82.2632 | 2430.95 |
| Prior plus traced neighbour emissions | 80.3079 | 2074.72 |
| Same prior, shuffled hit-to-emission-row mapping | 85.5019 | 2420.59 |

Every held-out target was compared with the float64 E12 evaluator on the same layout. Maximum absolute errors were 5.85e-8 for the prior and 7.76e-8 for the mixture. Five full vocabulary distributions, including UNK, were also checked against that evaluator. Shuffling left every prior probability identical. This measures implementation fidelity and dependence on the selected emission row. The original quality experiment, its scope and controls are in [E12](../e12_word_layout/README.md).

The archive occupied 988,351 bytes. The 10,110 dish triangles and 80,606 emission triangles used 617,728 and 4,870,144 acceleration-structure bytes respectively on RTX 5080. Scoring all held-out targets in one dispatch took 11.7 ms including buffer setup and readback, after loading and computing the first-stage hits. Three warm 128-token generation runs took 0.279–0.285 seconds, about 452 tokens/s at the median. Each step traces the full 3,338-token distribution. Two separate processes reproduced all 128 token IDs at seed 3.

## Reproduce

Use the project SlangPy environment with Vulkan hardware ray queries. Supply the E12 pinned book for the build and parity check. Choose a new archive filename. Scoring and generation take only this archive and the user's input. The tokeniser uses the same lowercase byte-level rule as E12; generated tokens are printed individually as hex to preserve punctuation and partial UTF-8 byte tokens without loss.

~~~sh
python experiments/word_emission/scene.py build BOOK_DIRECTORY words.npz
python experiments/word_emission/check.py BOOK_DIRECTORY words.npz report.json
python experiments/word_emission/scene.py score words.npz document.txt
python experiments/word_emission/scene.py generate words.npz Elizabeth --count 128 --seed 3
~~~

The published numbers use the host's default BLAS thread configuration. A one-thread build gave a slightly different force layout and 80.2943 perplexity; it also passed the registered 0.1% aggregate parity criterion and the per-target check against its own layout. Once the archive is built, generation does not fit a layout.
