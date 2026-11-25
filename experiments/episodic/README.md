# Ray-addressed episodic byte memory

Each observed byte occupies one angled triangle at its byte value and stream position. Rays fired across a row recover its byte from the hit distance. A ray fired backwards along a byte column finds a previous occurrence. Subsequent rays check older rows and read the successor of the best match. No byte IDs are bound to the query kernel; its only inputs are the acceleration structure and the number of positions to score. The frozen bigram is a separate surface with one triangle per byte pair. A perpendicular ray measures each triangle's height to recover its probability.

Search covers at most the preceding 256 positions, 16 candidate endings and eight matching bytes. Ties favour the most recent ending. If at least two bytes match, the retrieved successor receives 25% of the probability and the frozen bigram supplies the remaining 75%. The one-byte control uses the same weight for the nearest previous occurrence.

| Book | Targets | Frozen bigram | One byte | Longest match | Match coverage | Correct successor given a match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jekyll and Hyde | 143,229 | 3.477 | 3.680 | 3.414 | 61.6% | 31.9% |
| The Time Machine | 184,725 | 3.470 | 3.664 | 3.401 | 62.0% | 31.8% |

Loss is in bits per byte. Results met the preregistered 0.05 gain over both controls on each book. The Python reference independently recovered the identical nearest successor, longest-match successor and match length for every scored position. Flipping all future bytes in short suffixes at three cutoffs per book left predictions through the first changed target identical. The study reused books already examined in E5c.

## Stronger control and limitations

Post hoc, using the *same activation positions* but copying the nearest occurrence raised loss to 3.597 and 3.585 on the two books. Selecting the longer matching occurrence mattered beyond selecting the occasions to copy.

Replacing the frozen bigram with Claude's order-five modified Kneser–Ney predictor (PR #28) gave 2.106 and 2.158 bits/byte on the same positions. Adding the episodic prediction at the preregistered weight worsened these to 2.189 and 2.230. The experiment demonstrates local sequence retrieval that improves a weak long-term model. It does not establish a gain over a competitive long-term predictor.

The benchmark prebuilds the book's acceleration structure for parallel teacher-forced evaluation. The ray bounds prevent the current target and later triangles from being queried. It does not measure online writes, live generation or strict continuous specular paths. The model uses hit-dependent sequences of newly emitted rays. The reference uses ordinary byte search and computes exactly the same predictions. Each report includes the episodic acceleration structure size, build time, number of ray queries and single dispatch/readback wall time on RTX 5080. Times include host overhead and can vary with warmup.

## Reproduction

Use the SlangPy/Vulkan environment from the smoke experiment. Place the six E5c raw files in BOOK_DIRECTORY using the local names and SHA-256 digests in run.py. Source revisions and filenames are listed in the [E5c corpus manifest](https://github.com/SamG-Coder/RTXLLM/blob/1d8cdf705f3d96f6fb461066e9dc6f9a1887325d/experiments/adaptive_cavity/README.md). The first four books train the global prior; their boundaries are reset.

~~~sh
python -B experiments/episodic/run.py BOOK_DIRECTORY jekyll-43 jekyll.json
python -B experiments/episodic/run.py BOOK_DIRECTORY time-35 time.json
~~~
