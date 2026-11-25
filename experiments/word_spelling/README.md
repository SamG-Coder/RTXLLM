# Spelling previously unwritten tokens

The E14 word generator assigns probability to a shared unknown-word outcome. This extension distributes that probability over individual spellings. Each character transition is stored as a triangle whose height gives its probability. At load time 66,306 hardware rays read the entire grid; misses have zero probability. Scoring and sampling then reuse those heights on the host. The word scene still uses its two-stage RTX inference path. Per-character decoding uses cached ray results.

The spelling writer learns from tokens outside E14's vocabulary in the Pride training split. Both spell models share the first-byte distribution. The character model uses the preceding byte to choose a continuation row, while the control pools continuations into one row. Each legal transition gets 0.5 pseudocount. Alphanumeric tokens terminate with a stop outcome, and nonalphanumeric tokens contain one byte. This matches E14's lowercased byte tokenizer. Neither the spelling nor the word model predicts spaces or restores capitalization.

For an unknown token, the model multiplies its spelling probability by the E14 unknown-word probability and divides by the spelling mass remaining outside the vocabulary. Generation uses rejection sampling for that same conditional distribution. Known words retain their original E14 probabilities. The writer compares every GPU-read value to CPU counts. A separate CPU prefix-tree calculation sums the probability of leaving the known vocabulary and verifies the normalization for both models. The remaining mass was 0.92743 for character rows and 0.93657 for the shared row.

| Book | Unknown targets | Spelling bits per unknown byte, shared | Spelling bits per unknown byte, character | Whole model bits per token, shared → character |
| --- | ---: | ---: | ---: | ---: |
| The Adventures of Sherlock Holmes | 20,718 | 5.1644 | 4.6229 | 11.2540 → 10.8063 |
| The War of the Worlds | 14,210 | 4.8483 | 4.2214 | 12.8319 → 12.0611 |
| Moby Dick | 62,587 | 4.9596 | 4.3788 | 13.4742 → 12.7056 |

Every book exceeded the registered 0.05 bits per unknown-byte gain. These results measure spelling beyond the previous unknown-word label. All novel words still share the same history identity after emission. The character chain can produce pronounceable fragments and implausible spellings. In a 256-token sample at seed 69 it spelled four tokens, including “ouastygexcharaled”. Two independent processes reproduced the same complete sample. No language-quality advantage over a stronger character model has been established.

## Reproduce

Build the [E14 word archive](../word_emission/README.md). Use the pinned Pride training text and the Holmes, War and Moby Dick editions from issue #67. The test runner verifies their SHA-256 digests. Training requires a new spelling archive filename. Generation takes only the two archives and prompt.

~~~sh
python experiments/word_spelling/spell.py build words.npz spelling.npz --books BOOK_DIRECTORY
python experiments/word_spelling/spell.py test words.npz spelling.npz --books BOOK_DIRECTORY --report results.json
python experiments/word_spelling/spell.py generate words.npz spelling.npz --prompt Elizabeth --count 256 --seed 69
~~~

The spelling archive occupied 15,212 bytes, with 1,593 triangles and 104,704 acceleration-structure bytes on the RTX 5080. Generation returns a hex token and a flag identifying each spelling generated through the unknown-word branch.
