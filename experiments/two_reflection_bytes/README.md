# Two-reflection byte prediction

The first input byte chooses a launch station. The second input byte chooses a direction. On a written route, a mirror reflects that ray into a receiver for the observed pair. The second emission starts at that receiver. Its direction selects one of 32 quantile mirrors fitted from the training continuation counts. Another reflection reaches a byte receiver. A missing pair falls back to the one-reflection byte model. Both checkpoints contain geometric primitives. The generator reloads them without a corpus or count table.

## Reproduce

Use the Python 3.12 / SlangPy 0.43.1 environment described in the Vulkan smoke-test README. From the repository root:

```powershell
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
$book = "$env:TEMP/rtxllm-pride-1342.txt"
$results = "$env:TEMP/rtx-e2b"
$python = "$env:TEMP/rtxllm-vk-env/Scripts/python"
Invoke-WebRequest https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt -OutFile $book
& $python -B experiments/two_reflection_bytes/experiment.py $book $results
& $python -B experiments/two_reflection_bytes/experiment.py $book $results --split test
& $python -B experiments/two_reflection_bytes/generate.py $results Elizabeth --count 64 --seed 0
```

The runner checks the corpus SHA256, strips the Gutenberg marker lines, splits raw bytes at 70% and 85%, and resets context at each boundary. Training saw 493,508 bytes and wrote 1,176 distinct pair stations. Each reserved split contains 105,750 scored bytes. Programs and scene writers were frozen before the test run. The shuffled control permutes first-stage destinations while preserving the pair stations and their continuation mirrors.

| Model | Validation bits/byte | Test bits/byte |
| --- | ---: | ---: |
| Traced one-reflection bigram | 3.6847 | 3.6763 |
| Traced two-reflection model | 2.8829 | 2.8655 |
| Matched finite-vote CPU oracle | 2.8829 | 2.8655 |
| Exact smoothed trigram | 2.7585 | 2.7414 |
| Shuffled intermediate routing | 5.3626 | 5.3674 |

On an RTX 5080 (driver 616.64), all 38,816 tested paths per scene matched their expected intermediate and final receiver or miss. Test scoring reuses a distribution for repeated contexts. The clean two-reflection checkpoint occupied 2,440,580 bytes, its acceleration structures 2,442,240 bytes, and its build/update scratch 165,376 bytes. Across all clean batches, measured dispatches totaled 1.04 ms and traversal plus transfers and synchronization took 1.39 seconds. These costs exclude the separate one-reflection backoff model. Live generation traces both models anew at each output byte.

A second input byte and a second physical reflection improve this model by 0.8108 test bits/byte against the one-byte control. The model implements quantized trigram composition. The matched oracle reproduces it exactly. The experiment does not isolate a quality advantage for ray tracing over a conventional trigram or for reflection alone at matched context capacity. Previously observed pairs account for 99.83% of test targets; evidence of generalisation from shared geometry requires another experiment.
