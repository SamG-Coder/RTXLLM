# Context-dependent spelling recall

E18 copies an unfamiliar token from the last 64 unfamiliar occurrences with probability 0.25 when the model selects its unknown-word outcome. This experiment changes which occurrence it copies. Each row begins with two byte heights encoding the known vocabulary ID of its preceding word. A query reads those heights by RTX intersection. A matching known predecessor gives that row weight eight; every other row receives weight one. The row weights are normalized before copying. The unknown predecessor ID never triggers the extra weight. After row selection, further RTX intersections read the spelling and its terminator. The E14 word distribution and E17 spelling probability remain the other components of the model.

| Book | Uniform memory, bits/token | Displaced query | Actual preceding word | Gain vs uniform |
| --- | ---: | ---: | ---: | ---: |
| The Adventures of Sherlock Holmes | 10.14141 | 10.13651 | 10.10898 | 0.03243 |
| The War of the Worlds | 11.55233 | 11.55689 | 11.52162 | 0.03071 |
| Moby Dick | 12.03161 | 12.03696 | 11.98521 | 0.04640 |

All three gains exceeded the registered 0.005 bits/token threshold. On 50–54% of unfamiliar-word targets at least one eligible row received extra weight. The displaced control uses the predecessor of the immediately preceding unfamiliar occurrence. It is available causally but can overweight that last occurrence, so it is a temporal control rather than a random permutation. The implementation uses exact known-word ID equality. This tests lexical context and makes no inference about a word's meaning.

Both weighted matching-spelling counts and total masses matched independent CPU counts for every unfamiliar target under all three conditions. A fixture with conflicting spellings verifies that changing one predecessor changes the relative row masses from (8,1,1) to (1,8,8), and an unknown predecessor gives (1,1,1). Replaying the E18 benchmark with the shared shader reproduced its original JSON. Two fresh 512-token generations at seed 73 agreed and copied the prompted qzz89 exactly. Batch scoring uses complete book geometry with strictly historical bounds; live generation rebuilds the bounded rows as unfamiliar tokens arrive.

## Reproduce

Use the [E17 archives](../word_spelling/README.md), the same three digest-checked books as E18, and the repository SlangPy environment on RTX.

~~~sh
python experiments/context_spelling/route.py test words.npz spelling.npz --books BOOK_DIRECTORY --report results.json
python experiments/context_spelling/route.py generate words.npz spelling.npz --prompt "the qzz89 and qzz23 the qzz89 and qzz23 the" --count 512 --seed 73
~~~

## Save a word session

Start from a prompt, append observed text, or resume a saved sequence. The model archive and spelling archive must match the SHA-256 digests recorded in the session. Each command publishes a new checkpoint and prints the generated tokens as hex plus their copy flags. Count zero records observations without sampling.

~~~sh
python experiments/context_spelling/session.py words.npz spelling.npz first.npz --prompt "the qzz89 and qzz23 the" --seed 73 --count 128
python experiments/context_spelling/session.py words.npz spelling.npz second.npz --resume first.npz --observe further.txt --count 128
python experiments/context_spelling/check_session.py words.npz spelling.npz
~~~

Each observation is tokenized as a separate segment of complete tokens. The previous word and the 64-occurrence window carry across the segment boundary. Splitting a single word between calls therefore changes the observed tokens. The archive stores the current GPU vertex buffer, previous word ID, RNG state, format version and both model digests. Restore rebuilds the acceleration structure and reads each historical record from geometry before continuing. The writer uses a complete staged file and publishes it with a hard link, preserving any checkpoint already at the destination.

On RTX 5080 the 512-token run agreed with fresh-process continuations split at token 137 and immediately after token 479, where qzz89 was copied. Every token and copy flag matched the original one-shot E19 implementation. The empty checkpoint occupied 1,516 compressed bytes. A checkpoint with 64 rows of test spellings occupied 5,472 bytes. The boundary check also verifies that the most recent 64 rows survive eviction, the predecessor of the first appended spelling survives restart, both model identities are checked, and an attempted overwrite preserves the existing checkpoint.
