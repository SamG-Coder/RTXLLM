# Adaptive cavity detector experiment

Four training books (Alice, Pride and Prejudice, Frankenstein and Dracula) supply 2,153,232 next-byte targets. Each book starts a new context. The benchmark checks each raw text against its SHA-256 and strips the Gutenberg wrapper. Jekyll and Hyde is the validation book. The Time Machine is the sealed test book. Issue #26 records the criterion and the code freeze before test.

The final byte selects a physical detector page. Either one or seven preceding bytes steer the ray through the shared cavity. Training partitions the landing positions into rectangular regions and writes next-byte masses to triangle strips. A Vulkan ray query selects the region. The benchmark compares each GPU region assignment with a CPU bounds check. All variants have a cap of 23,320 learned triangles plus the shared cavity.

| Held-out book (bits/byte, lower is better) | Fixed one | Fixed seven | Adaptive one | Adaptive seven | Adaptive seven, shuffled prefix | Order-six count control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jekyll and Hyde | 2.9454 | 3.1049 | 3.0803 | 3.0763 | 3.3265 | 2.1599 |
| The Time Machine | 2.9486 | 3.1073 | 3.0815 | 3.0886 | 3.3122 | 2.2182 |

The seven-return adaptive model missed the required 0.05 bits/byte gain over each of adaptive one and fixed seven. Its nonempty-row coverage exceeded 99.99% on test, and all traced paths completed. Recovering coverage alone did not recover predictive accuracy. The JSON files alongside this document include all count baselines, serialized sizes and triangle counts.

## Reproduction

Use the repository SlangPy/Vulkan environment on an RTX GPU. Place the six raw GITenberg files in a single directory using the names listed below. Run from the checkout root. Validation writes the trained scenes. Test reloads them unchanged.

~~~sh
python experiments/adaptive_cavity/benchmark.py validate BOOK_DIRECTORY OUTPUT_DIRECTORY
python experiments/adaptive_cavity/benchmark.py test BOOK_DIRECTORY OUTPUT_DIRECTORY
~~~

| Local filename | Pinned raw source |
| --- | --- |
| rtxllm-alice-11.txt | https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/7f18748e114b994d5d9454d5793581bc0b9cf9d7/11-0.txt |
| rtxllm-pride-1342.txt | https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt |
| rtxllm-frankenstein-84.txt | https://raw.githubusercontent.com/GITenberg/Frankenstein_84/c98a483bd1f34e7366126b81bcedd1a6faee636e/84-0.txt |
| rtxllm-dracula-345.txt | https://raw.githubusercontent.com/GITenberg/Dracula_345/5216d6ea255fbf8966426a160b68251b18f16907/345.txt |
| rtxllm-jekyll-43.txt | https://raw.githubusercontent.com/GITenberg/The-Strange-Case-of-Dr.-Jekyll-and-Mr.-Hyde_43/534f1e1d3193ffaf351570cf808e0ee6db70efe9/43-0.txt |
| rtxllm-time-35.txt | https://raw.githubusercontent.com/GITenberg/The-Time-Machine_35/154a5c628743bd7d35d1af0fc4bcc017f0e111e1/35-0.txt |
