# Exact recent spelling recall

On an unknown-word decision the generator can spell a new token with the E17 character model or sample an occurrence from the last 64 unknown tokens. Once that memory contains at least one occurrence, copying gets a fixed probability of 0.25. Every observed or generated unknown then enters the memory, including repeated spellings. The E14 predictor still uses one shared unknown history identity to decide the next word class.

Each row is an occurrence. One triangle per byte stores byte value plus one as its height, followed by a triangle of height 257 marking the end. To score a target spelling, RTX rays read each historical row until a mismatch or the end marker, then count identical rows. Copy probability is the count divided by the number of eligible prior rows. For generation the sampler selects an occurrence index uniformly; rays reconstruct the bytes and end marker from the selected row. Live generation rebuilds the bounded geometry after each new unknown. The host retains recent bytes to write these triangles. Batch evaluation builds a larger book scene and uses the shader's past-only row bounds at each position.

| Book | Scored tokens | E17 bits/token | Recent 64 | Distant 64 | Unknown targets found in recent memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| The Adventures of Sherlock Holmes | 128,961 | 10.8063 | 10.1414 | 10.3708 | 36.74% |
| The War of the Worlds | 73,964 | 12.0611 | 11.5523 | 11.8766 | 19.42% |
| Moby Dick | 281,050 | 12.7056 | 12.0316 | 12.4554 | 23.53% |

The improvements of 0.665, 0.509 and 0.674 bits per token passed the registered minimum of 0.05 on every book. These are open-vocabulary, lowercased token scores with whitespace omitted. The displaced control skips the previous 256 unknown occurrences and uses the 64 directly preceding those. That version also improves on spelling alone, consistent with repeated vocabulary within each book. The recent window improves further on all three. These fixed-book tests measure exact repetition; they do not test whether the model understands what a repeated word denotes.

Both GPU count columns matched CPU byte comparisons at every unknown target for both windows. Replacing the book suffix preserved prefix results. A separate generation check reads prefix-related words, embedded zero and non-UTF-8 bytes, and a 512-byte spelling from geometry. Two fresh processes with 100 artificial prompt words and seed 71 generated the same 512 tokens. The observed copy was qzz89. Every copy was validated against the last 64 occurrences before that output.

## Reproduce

Use the E14 and E17 archives from [the spelling experiment](../word_spelling/README.md), the same three digest-checked books and the repository's SlangPy environment on RTX hardware.

~~~sh
python experiments/spelling_recall/recall.py test words.npz spelling.npz --books BOOK_DIRECTORY --report results.json
python experiments/spelling_recall/check_live.py words.npz spelling.npz
python experiments/spelling_recall/recall.py generate words.npz spelling.npz --prompt "Elizabeth met qzz89" --count 128
~~~
