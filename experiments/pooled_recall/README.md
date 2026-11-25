# Sharing copy probability across tied paths

A backward ray finds each earlier occurrence of the two-byte address. Transverse rays check preceding bytes and read the successor of each occurrence tied for the longest match. Up to 16 candidates in the previous 2,048 bytes contribute. The nearest control uses only the first tied successor. Pooling divides copy probability equally among tied occurrences, so repeated votes for the same byte accumulate. The shader returns the vote count for the target. The host divides by the number of paths and mixes this mass with the GPU-decoded long-term probability using a ray-read match-length weight.

| Book | KN-6 prior | Nearest | Pool at nearest's weights | Pool with own weights |
| --- | ---: | ---: | ---: | ---: |
| Jekyll and Hyde, calibration | 2.03017 | 1.98474 | 1.98010 | 1.97905 |
| The Adventures of Sherlock Holmes, fresh test | 2.03233 | 1.99051 | 1.98712 | 1.98670 |

Units are bits per byte. Of 575,186 test targets, 308,814 had multiple tied paths and 241,517 had different successors among those paths. Pooling improved loss by 0.00381 bits/byte over calibrated nearest, missing the preregistered 0.005 margin. With weights held fixed the improvement was 0.00338. All path lists, selected lengths and target vote fractions matched an independent CPU search at every target in both books. The experiment uses a batched causal evaluation scene. It measures prediction quality and does not establish a hardware speed advantage.

## Reproduction

Build the frozen order-six four-book E9 model as described in [the integrated model instructions](../integrated/README.md). Supply the same pinned Jekyll file and the [pinned Holmes text](https://raw.githubusercontent.com/GITenberg/The-Adventures-of-Sherlock-Holmes_1661/7e5dd2703fbf503c004063e96ab7c61262375e5f/1661.txt) as rtxllm-holmes-1661.txt. The implementation verifies raw SHA-256 digests before removing the Gutenberg wrappers. Calibration fits two gate scenes on Jekyll; test reloads both from the output directory.

~~~sh
python experiments/pooled_recall/run_pool.py calibrate order6.npz BOOK_DIRECTORY OUTPUT_DIRECTORY
python experiments/pooled_recall/run_pool.py test order6.npz BOOK_DIRECTORY OUTPUT_DIRECTORY
~~~
