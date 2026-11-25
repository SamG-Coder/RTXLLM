# Long-term support for an episodic prediction

On 1,256,564 held-out next bytes from Moby Dick, the four-book six-gram scene used 2.359512 bits/byte. Copying from earlier passages with the length-only gate reduced that to 2.279941. Giving the gate the long-term probability of the nominated successor reduced it to 2.278367. The 0.001574 gain missed the preregistered 0.005 threshold. Jekyll calibration had shown a 0.001319 gain. The 27-triangle gate remains experimental.

Each query traces the same E10 paired-address memory (2,048 bytes, at most 16 candidate endings, eight history bytes). RTX suffix hits supply the six-gram probability of the nominee. The host maps it to one of three intervals with boundaries at 1/16 and 1/2. Interval and match length select a physical gate triangle. A final ray measures its height, which supplies the interpolation weight. Length-only uses nine triangles and the same E9 fitting grid. The candidate, its prior support and the gate address are determined before consulting the target. An absent nominee has zero copy weight.

The runner compares every nomination with a CPU substring search, every target and nominee probability with independent float64 Kneser–Ney, and every readout with the gate's stored height. It changes future bytes at three cutoffs and checks that the nomination, support and gate address through the first changed target remain identical. The JSON archives record the support and weight for each row, ordered by probability interval then length (0–8). This batch evaluation uses prebuilt book geometry with causal search bounds. Both conditions use the same floating-point GPU prior and the same episode matches.

## Reproduce

Build the order-six archive from the four pinned E9 books as described in the [E10 instructions](../pair_address/README.md). Use the same Jekyll calibration file. Save the [pinned Moby Dick file](https://raw.githubusercontent.com/GITenberg/Moby-Dick--Or-The-Whale_2701/bdf1948e6cd00963730971e5624e764a35f238c3/2701-0.txt) as rtxllm-moby-2701.txt in the book directory. The runner checks its digest. With the repository SlangPy environment on RTX, run both commands from the checkout root.

~~~sh
python experiments/agreement_gate/run.py fit order6.npz BOOK_DIRECTORY OUTPUT_DIRECTORY
python experiments/agreement_gate/run.py test order6.npz BOOK_DIRECTORY OUTPUT_DIRECTORY
~~~

The first command writes both fitted gate geometries. The second reloads them and scores the whole wrapper-stripped book starting at byte eight. The candidate model and validation report were frozen at 4245de7 before the test.
