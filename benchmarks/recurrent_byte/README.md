# Four-book recurrent byte baseline

This standalone comparator trains a single-layer GRU on the four books used by the integrated RTX byte predictor. Both models see the same raw training documents and scored test targets. Each test book begins with eight unscored bytes. Their memory capacity and training algorithms differ.

| Book | Scored bytes | Stateful GRU, bits/byte | Six-gram alone | Six-gram + geometric episodic memory | GRU minus combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| The War of the Worlds | 346,826 | 2.15217 | 2.13000 | 2.11409 | +0.03808 |
| Moby Dick | 1,256,564 | 2.44197 | 2.35951 | 2.27994 | +0.16203 |

Lower loss is better. On both books the six-gram alone already outscored this GRU. Adding traced episodic memory contributed 0.01590 and 0.07957 bits/byte respectively relative to that prior. War uses the 256-byte memory and gate measured in [the integrated model](https://github.com/SamG-Coder/RTXLLM/pull/41). Moby Dick uses the 2,048-byte paired memory and length-only gate measured in [the copy-gate study](https://github.com/SamG-Coder/RTXLLM/pull/56). The spatial results were public before this retrospective comparison.

The GRU has 214,400 parameters (857,600 float32 bytes), 64-dimensional embeddings, 192 recurrent units and a 256-way output. The order-six long-term RTX archive occupies 4,712,722 compressed bytes; live episodic geometry and acceleration structures require additional device memory. Parameter bytes and compressed model bytes describe different representations. Training shuffles 128-target chunks. Validation and primary test scoring carry the hidden state from the start of each book. Resetting at each training-length boundary during testing gave 2.20019 on War and 2.48684 on Moby. The last partial chunk contributes every real target. The six-gram uses five history bytes and the episodic search can reach much further back.

The GRU trained for 47 epochs on 2,153,260 transitions. Epoch 39 had the lowest loss on 143,229 Jekyll and Hyde targets (2.04459 bits/byte). Training stops after eight epochs without a cumulative improvement of 0.001 bits/byte, with a cap of 100. Stateful scoring in blocks of 1,024 and 1,664 inputs agreed within 0.000001 bits/byte. The result covers one model size and seed. The earlier six-epoch, periodically reset baseline overstated the gap.

## Reproduce

Use a CUDA-enabled PyTorch environment (recorded with PyTorch 2.11.0+cu128 on RTX 5080) and NumPy. Place the pinned training and Jekyll files from the [cross-book experiment](https://github.com/SamG-Coder/RTXLLM/pull/27) in BOOK_DIRECTORY. Add the War of the Worlds and Moby Dick editions used by the linked spatial studies. The runner checks all seven SHA-256 digests. The output directory holds the selected checkpoint and both reports.

~~~sh
python benchmarks/recurrent_byte/run.py fit BOOK_DIRECTORY OUTPUT_DIRECTORY
python benchmarks/recurrent_byte/run.py test BOOK_DIRECTORY OUTPUT_DIRECTORY
~~~

The spatial predictors have no dependency on this script or its checkpoint.
