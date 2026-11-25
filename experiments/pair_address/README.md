# Two-byte addresses for episodic recall

The episodic memory places one angled triangle at each time row. In the original layout its x coordinate is the latest byte. In the paired layout x is 256 times the preceding byte plus the latest byte. A backward ray now finds an occurrence of both bytes at once. A transverse ray measures x and recovers the low byte when checking an earlier byte or reading the successor. Both layouts check at most 16 candidate endings, choose the longest match up to eight bytes, and break ties in favor of the most recent occurrence.

The four-book KN-6 scene and calibrated E9 gate are identical across conditions. The test book was untouched by E10 until the implementation was frozen. All four variants use RTX for suffix and episodic search, copy-gate readout and payload decoding.

| Placement and search window | Jekyll and Hyde, bits/byte | Treasure Island, bits/byte | Treasure Island episodic rays |
| --- | ---: | ---: | ---: |
| No episodic prediction | 2.03017 | 2.17410 | 0 |
| Single byte, 256 | 2.01963 | 2.14994 | 12,725,915 |
| Byte pair, 256 | 2.01908 | 2.14929 | 5,613,058 |
| Single byte, 2,048 | 2.00519 | 2.13762 | 16,736,900 |
| Byte pair, 2,048 | 1.98594 | 2.11371 | 12,362,509 |

At the longer window, the pair placement covered 93.25% of the 376,884 Treasure Island targets with a match of at least two bytes, versus 72.99% for the single-byte placement. Correct successors among those matches rose from 36.90% to 39.71%. Both long-window runs used 376,892 triangles and 22,758,528 bytes of episodic acceleration structures. They took about 26 and 25 ms respectively for search and readback in one run. Lower ray count did not yield a measured wall-time improvement here.

All four sets of results matched the independent CPU substring oracle for nearest successor, selected successor and match length at every position. Changing future bytes at three cutoffs per book preserved predictions through the first changed target. The test met the registered 0.01 bits/byte threshold against both original and matched-window controls. This experiment uses prebuilt book geometry with bounded causal queries. Live writing and autoregressive generation using *pair* placement remain to be tested. The six-gram prior and copy gate were trained earlier. The pair-address layout itself is deterministic.

## Reproduction

Follow the integrated model instructions to build the order-six E9 archive. Keep the existing Jekyll corpus file and save the [pinned Treasure Island text](https://raw.githubusercontent.com/GITenberg/Treasure-Island_120/46f38f7b63dbda34415ce29ef5548023a8ba8dda/120-0.txt) in the book directory as rtxllm-treasure-120.txt. Its SHA-256 is checked before scoring.

~~~sh
python experiments/integrated/model.py build OUTPUT_DIRECTORY/gate.npz order6.npz --e9-books BOOK_DIRECTORY --order 6
python experiments/pair_address/run_index.py validate order6.npz BOOK_DIRECTORY validation.json
python experiments/pair_address/run_index.py test order6.npz BOOK_DIRECTORY test.json
~~~
