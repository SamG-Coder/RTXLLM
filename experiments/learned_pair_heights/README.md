# Learning when to use the earlier word

A two-word station stores a continuation distribution and a physical height. Its height controls how much to mix that distribution into the one-word predictor. The fixed E15 height over-weighted memorized word pairs on a different author. Here the writer tries ten candidate elevations on actual RTX geometry, scores calibration text after each rebuild, then stores the elevation that minimizes loss for each group of stations. A hit on a station supplies the blend weight and selects the row for the next emission ray. A missing station falls back to the E14 one-word predictor.

Pride and Prejudice supplies the vocabulary, layout, pair identities and emissions. Jekyll and Hyde supplies only the calibration targets. There are six groups, determined by pair count (one, two to four, five or more) and presence of an unknown word in the context. Both known- and unknown-word groups selected the same three heights, 0.025, 0.05 and 0.10. A single-height control selected 0.10. The code and elevations were frozen at 1418de3 before evaluating The Yellow Wallpaper.

| Predictor | Overall perplexity | Pair seen in training | Pair unseen in training |
| --- | ---: | ---: | ---: |
| E14, one word | 145.83 | 144.15 | 150.22 |
| E15, fixed count-based height | 233.29 | 276.88 | 150.22 |
| Learned single height | 144.09 | 141.77 | 150.22 |
| Learned group heights | **143.58** | **141.07** | 150.22 |
| Shuffled older word, learned heights | 147.22 | 145.71 | 151.18 |
| Modified Kneser–Ney word trigram, post-hoc | 148.47 | 148.08 | 149.47 |

The registered primary comparison passed. Learned group heights reduced perplexity by 1.54% relative to E14 on 7,728 next-word targets from one short book. The single global height accounts for most of this improvement. All 2,165 unseen contexts backed off, so the gains come from previously observed pairs. The shuffled condition uses the original context to classify test targets. The post-hoc CPU trigram uses the same word stream, vocabulary, test targets and train corpus, with interpolated modified discounts and continuation counts. It has no Jekyll calibration or spatial beam. A CPU implementation of the same fitted blend reproduces the hardware probabilities.

All observed pair heights and target probabilities were checked against the original counts and the fitted weights (absolute tolerance 2e-6). Two fresh processes generated identical 128-token samples using only the saved E14 model and learned pair scene. On the RTX 5080 the calibration and verification command took 1.52 seconds, and the test and verification command took 0.76 seconds after Python startup. Those wall times include loading the base model, reconstructing source counts and validating the control scenes.

## Reproduce

Build the [E14 model](../word_emission/README.md) and fit the [E15 pair scene](../word_pairs/README.md). Use a fresh output directory for the E16 fit. Keep the pinned Pride and Jekyll editions in the book directory. Save the [pinned Yellow Wallpaper text](https://raw.githubusercontent.com/GITenberg/The-Yellow-Wallpaper_1952/ce7c5bc1a5cf0fe23bcb265545a900365dbae29c/1952-0.txt) as rtxllm-wallpaper-1952.txt in that directory. The runner verifies its SHA-256 and strips the Gutenberg wrapper.

~~~sh
python experiments/learned_pair_heights/learn_heights.py fit words.npz OUT --source E15/pairs.npz --books BOOK_DIRECTORY
python experiments/learned_pair_heights/learn_heights.py test words.npz OUT --source E15/pairs.npz --books BOOK_DIRECTORY
python experiments/learned_pair_heights/trigram.py words.npz BOOK_DIRECTORY trigram.json
python experiments/learned_pair_heights/learn_heights.py generate words.npz OUT --prompt "Elizabeth was" --count 128
~~~

Generation needs only words.npz and OUT/learned.npz. It reads the probabilities for the whole vocabulary, checks their sum and samples on the host. E16 uses the exact pair output of the shared E15 tracer; the auxiliary donor result is discarded. Each test report records group support in the order unseen, three known-word groups, three unknown-word groups. The heights array begins with the global control.

## Transfer to three other authors

The frozen E14, E15 and E16 archives were replayed on every next-token target after the first two tokens of each book. No calibration or fitting used the test texts. SHA-256 checks pin the four archives and three books. All grouped, global, fixed and shuffled pair weights and probabilities matched the CPU count mixtures within 2e-6. These books had been used in separate byte-model experiments; this was their first use as E16 word transfer tests.

| Book | Targets | One-word perplexity | Single fitted height | Grouped heights | Relative gain vs one word (95% block interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| The Adventures of Sherlock Holmes | 128,960 | 126.60 | 123.92 | 123.65 | 2.33% (2.16–2.51%) |
| The War of the Worlds | 73,963 | 117.03 | 115.60 | 115.29 | 1.49% (1.33–1.66%) |
| Moby Dick | 281,049 | 120.32 | 119.50 | 119.07 | 1.03% (0.94–1.12%) |

All three point estimates passed the registered 1% threshold. The Moby Dick interval includes smaller gains. Across 483,972 targets, token-weighted log loss gave a 1.45% reduction in perplexity. Shuffling the earlier word produced worse perplexity than the one-word predictor on every book. Most of the improvement was available from the global height.

Unknown targets accounted for 16.1%, 19.2% and 22.3% respectively. Restricted to known targets, the relative gains were 2.76%, 1.24% and 0.85%. The model assigns a single token to all unknown words. Predicting that token does not identify the original word. Intervals resample 256-target contiguous blocks within each book, retaining the length of the final block. They describe variation within these texts and do not estimate uncertainty over new authors.

Run the following command with the original frozen archives and the same pinned editions used in the E11 (Holmes), E9 (War) and E13 (Moby Dick) experiments. Exact SHA-256 digests are in the runner. The [full report](transfer.json) gives all controls, group support, and known/unknown results.

~~~sh
python experiments/learned_pair_heights/transfer.py words.npz E15/pairs.npz E16 BOOK_DIRECTORY transfer.json
~~~

## Matched conventional comparator (Claude, #74)

The trigram above had no calibration data, while E16 tuned its heights on Jekyll. `matched.py` gives conventional models the **same** calibration access on exactly the same targets, vocabulary and Pride training stream. *interp* is P = (1 − λ)·E14 + λ·KN3, where KN3 is the interpolated modified-KN word trigram from `trigram.py` and λ is chosen from E16's own height grid on the 33,377 Jekyll targets (selected λ = 0.35; Jekyll 90.06 vs E16 grouped 91.27). *backoff* is KN3 whose lower-order distribution is E14 itself, with no tuning. E14 and E16 probabilities are read from the frozen RTX archives exactly as in the transfer test. The code was committed before any evaluation book was scored.

| Book | E14 | E16 grouped | KN3 alone | KN3 backoff to E14 | E14 ⊕ KN3, λ tuned on Jekyll | E16 vs best conventional [95% block interval] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Yellow Wallpaper | 145.83 | 143.58 | 148.47 | 168.86 | **140.93** | −1.88% [−2.23, −1.54] |
| Holmes | 126.60 | 123.65 | 130.66 | 129.45 | **122.06** | −1.30% [−1.40, −1.21] |
| War of the Worlds | 117.03 | 115.29 | 119.60 | 126.40 | **113.16** | −1.88% [−2.01, −1.75] |
| Moby Dick | 120.32 | 119.07 | 124.93 | 138.48 | **117.07** | −1.71% [−1.77, −1.65] |

Negative gains mean E16 is worse. The predeclared criterion for the geometric two-word stations adding value (E16 ≥ 0.5% better than the best conventional comparator on at least 3 of 4 books) is **not met**: standard interpolation with a smoothed trigram beats E16 on every book, and every interval excludes zero. The E16 gain over E14 is therefore the ordinary benefit of trigram context, and a KN-smoothed trigram realises it better than E15's maximum-likelihood pair rows with learned heights. E14 (the RTX one-word beam model) remains the best base: E14 ⊕ KN3 beats KN3 alone by 5–7% on every book.

~~~sh
python experiments/learned_pair_heights/matched.py words.npz E16 BOOK_DIRECTORY matched.json
~~~
