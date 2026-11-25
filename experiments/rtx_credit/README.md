# Learning shared mirror routes from RTX prediction error

A launch mirror for each input byte directs a ray onto a plane of eight adjacent receiver strips. From the hit strip, 32 rays reflect from a shared set of continuation mirrors into next-byte receivers. The checkpoint stores mirror and receiver triangles. Rebuilding its acceleration structures and tracing it reproduces the same routing and probabilities.

Training proposes a continuous horizontal aim point for every launch mirror, refits the Vulkan acceleration structures, and traces every two-reflection path. For each observed input byte, the optimizer keeps a rotation when the resulting receiver histogram reduces next-byte loss on its training transitions. It retraces the accepted geometry and credits training observations to the actual intermediate hits. Those pooled observations set the next orientations of the 32 output mirrors in each bank. The 1% uniform escape keeps every byte possible. There are 24 rounds and 74 training dispatches per run.

## Reproduction

Use the Python 3.12 / SlangPy 0.43.1 environment from the Vulkan smoke experiment. The corpus SHA256 and contiguous 70/15/15 split are enforced in the runner. Context resets at each boundary. Settings and success criteria were registered in #13 before implementation. The writer and held-out scoring code were frozen at commit 988dd5d before running the test split.

```powershell
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
$book = "$env:TEMP/rtxllm-alice-11.txt"
$results = "$env:TEMP/rtxllm-credit"
$python = "$env:TEMP/rtxllm-vk-env/Scripts/python"
Invoke-WebRequest https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/7f18748e114b994d5d9454d5793581bc0b9cf9d7/11-0.txt -OutFile $book
& $python -B experiments/rtx_credit/learn.py $book $results
& $python -B experiments/rtx_credit/learn.py $book $results --split test
```

## Held-out loss

The training stream contained 107,700 bytes from *Alice’s Adventures in Wonderland*. The test split contained 23,078 next-byte targets. The same seed initializes the feedback, frozen-route, shuffled-credit, and CPU assignment models.

| Seed | RTX feedback | Frozen routes | Shuffled credit | CPU assignment | Exact bigram |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.9632 | 5.1127 | 5.1848 | 4.9690 | 3.3253 |
| 1 | 4.6280 | 5.0389 | 5.3207 | 4.7634 | 3.3253 |
| 2 | 4.7297 | 5.0129 | 5.1076 | 4.7600 | 3.3253 |

The units are bits per byte. Both control gaps exceeded 0.1 in every seed. Shuffled credit permutes the accept/reject decisions among observed input bytes while using the same proposed rotations. CPU assignment uses the same eight banks and quantized emissions, exhaustively chooses a bank for each byte on each round, and can converge to a different local optimum. A CPU implementation could replay this optical model's decisions by computing the strip intersections. The results establish that feedback helps train these mirrors, while relative algorithmic and hardware efficiency remain open.

Each saved scene occupied 46,324 bytes. On RTX 5080 / 616.64 with validation enabled, GPU dispatch time summed to 0.75–0.77 ms per feedback run. Observed wall time for the full feedback fit and verification was 0.33–0.46 s, including host proposal construction, geometry refits, transfers, CPU comparison and checkpoint reload. The wall times are single measurements for a small fixture.

## Post-hoc capacity checks

On the same held-out targets, the exact unigram achieved 4.7637 bits/byte. One of the three trained scenes scored worse than that context-free predictor. Holding the learned routes fixed and replacing the 32 output votes with the full empirical distribution of each bank yielded 4.0281, 3.9527 and 3.9682 bits/byte. Giving each byte its own 32-vote distribution yielded 3.9985. Both bank sharing and vote quantization cost quality. These diagnostics used the frozen models after test scoring and did not select training settings.
