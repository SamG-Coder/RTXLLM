# How much to trust a recent match

A repeated two-byte sequence is a weak hint about the next byte. A repeated eight-byte sequence can reveal a local name or phrase. E9 fits a separate copy weight for each match length. Each fitted weight is stored as the height of a triangle. The E7 episodic search traces the last 256 bytes on RTX; its resulting match length selects the ray that measures the weight. Prediction interpolates the copied successor with a frozen CPU Kneser–Ney prior. The CPU reference independently checks every episodic search result.

Four books train the fixed five-gram prior. Jekyll and Hyde calibrates the mixture weights. For matches of length two through eight, calibration selected 1%, 2.5%, 5%, 10%, 20%, 35% and 35%. Shorter matches get zero. The best constant weight was 2.5%. The code and gate checkpoint were frozen on issue #37 before the fresh-book test.

| Source | Targets | KN-5 alone | Fixed 25% | Calibrated constant | Learned by length | Reversed heights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jekyll and Hyde, calibration | 143,229 | 2.10591 | 2.18945 | 2.10216 | 2.09504 | 2.25305 |
| The War of the Worlds, test | 346,826 | 2.21867 | 2.28897 | 2.21007 | 2.20109 | 2.35319 |

Loss is measured in bits per byte. The registered improvement threshold over KN-5 was 0.01, with a further requirement to beat the constant-weight control. Reversing only the heights changed 204,600 test target probabilities.

In post-hoc checks, the KN-5 improvement was positive in all 32 contiguous, approximately equal-size parts of the test book (0.0063 to 0.0452 bits/byte). The unchanged gate also improved order-seven KN from 2.10825 to 2.09312. These checks reused the same test book. The prior runs on CPU in this experiment; the learned gate and episodic search run on RTX 5080. A CPU implementation of both optical operations can produce equivalent predictions.

## Reproduction

Use the repository's SlangPy/Vulkan environment. Place the first four training books and Jekyll and Hyde in BOOK_DIRECTORY using the names in the [E5c corpus manifest](https://github.com/SamG-Coder/RTXLLM/blob/1d8cdf705f3d96f6fb461066e9dc6f9a1887325d/experiments/adaptive_cavity/README.md). Save the [pinned War of the Worlds text](https://raw.githubusercontent.com/GITenberg/The-War-of-the-Worlds_36/8b08fa3a7cbef117c80b2b437adc46e75c37dbf0/36-0.txt) there as rtxllm-war-36.txt. The evaluator verifies raw file digests and removes the Gutenberg wrapper.

~~~sh
python experiments/copy_confidence/experiment.py calibrate BOOK_DIRECTORY OUTPUT_DIRECTORY
python experiments/copy_confidence/experiment.py test BOOK_DIRECTORY OUTPUT_DIRECTORY
~~~

Calibration writes gate.npz; test reloads it. Reports contain controls, intervention counts, match lengths, and post-hoc diagnostics.
