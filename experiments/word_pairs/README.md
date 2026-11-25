# Two-word continuation stations

E15 tests whether earlier words with similar next-word behaviour can share evidence while keeping the latest word fixed. The base predictor is the persisted E14 model. Each observed pair (older word, recent word) has a triangle. A hit returns its row identity and a height proportional to how much to trust its observations, c/(c+5). From that row a second RTX query reads the empirical successor probability from another triangle. The exact-pair model mixes this distribution with E14. The donor model additionally queries the same pair scene using the older word's beam neighbours and the actual recent word. A donor pair miss contributes no energy. When all donors miss, the exact-pair distribution is used. Source word beams are cached as in E14.

## Cross-book result

On 33,377 Jekyll and Hyde targets, trained on Pride and Prejudice, donors passed the registered 1% improvement over the fixed exact-pair rule. Both two-word predictors performed worse than the one-word model. Shuffling the older input helped overall. The full result argues against using this experiment's trust rule for generation by default.

| Model | Overall perplexity | On 7,885 previously unseen context pairs |
| --- | ---: | ---: |
| One-word E14 | 92.81 | 125.11 |
| Exact pair | 137.59 | 125.11 |
| Exact pair with Witten–Bell backoff (CPU diagnostic) | 115.11 | 125.11 |
| Exact pair plus donors | 134.55 | 127.15 |
| Shuffled older input, same model and target groups | 121.19 | 149.53 |

A donor station was reachable on 63.0% of targets. The donor weight of 0.10 was chosen on Pride validation, where perplexities were 81.59 (one word), 96.86 (exact pair) and 93.47 (pair plus donors). The training set, vocabulary, layout, weight grid, scene and scoring code were frozen before the cross-book test. The diagnostic Witten–Bell control substitutes the number of distinct successors for the constant five in the backoff formula. It was added after seeing validation and before opening the test book. All GPU outputs, including the shuffled control, were checked against independent CPU training counts using the same first-stage hit list. The pair scene contains 36,965 station triangles plus their emission triangles and uses 6,837,248 acceleration-structure bytes on RTX 5080.

## Reproduce

Build the [E14 archive](../word_emission/README.md) with the pinned Pride file. Include the pinned Jekyll file from the cross-book experiments as rtxllm-jekyll-43.txt. Fit writes new pair geometry and calibration into a fresh output directory. Test uses both files and checks the model and pair archive digests. Generation uses only the E14 archive, the pair scene and its calibration. Its prompt needs two tokens.

~~~sh
python experiments/word_pairs/run_pairs.py fit words.npz OUT --books BOOK_DIRECTORY
python experiments/word_pairs/run_pairs.py test words.npz OUT --books BOOK_DIRECTORY
python experiments/word_pairs/run_pairs.py generate words.npz OUT --prompt "Elizabeth was" --count 32
~~~
