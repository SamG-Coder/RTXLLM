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
