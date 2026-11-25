# Conventional neural byte baseline

This isolated baseline learns byte embeddings, concatenates the last few embeddings, applies one tanh hidden layer, and predicts the next byte with a softmax. Widths 1 and 2 match the context length of the E4 and E2b optical models respectively. Width 8 measures the benefit a small conventional model can obtain from longer context. The optical experiments have no dependency on this module.

The settings in issue #15 were fixed before training. Every model uses embedding dimension 12, hidden dimension 24, batch size 512, Adam learning rate 0.003, seed 20261125, and 12 training epochs. Select the lowest validation loss among the 12 checkpoints, then score the test split. Each corpus digest, split boundary, and first scored byte matches its optical experiment. Padding is local to a split. The runner first checks the gradient of each parameter tensor at its largest-magnitude component with a float64 central difference, and verifies early-context padding and target alignment.

## Reproduce

With Python 3.12 and NumPy 2.5.5, from the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$bookA = "$env:TEMP/rtxllm-alice-11.txt"
$bookP = "$env:TEMP/rtxllm-pride-1342.txt"
Invoke-WebRequest https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/7f18748e114b994d5d9454d5793581bc0b9cf9d7/11-0.txt -OutFile $bookA
Invoke-WebRequest https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt -OutFile $bookP
python benchmarks/neural_byte/train.py alice $bookA "$env:TEMP/rtx-neural-alice.json"
python benchmarks/neural_byte/train.py pride $bookP "$env:TEMP/rtx-neural-pride.json"
```

## Recorded results

Windows, AMD Ryzen 7 9800X3D. Parameter bytes measure float32 weights and biases only. Training wall time includes validation after each epoch. The companion JSON files contain the complete validation histories and scoring times.

| Corpus | Context bytes | Scored test bytes | Parameters | Parameter bytes | Validation-selected epoch | Test bits/byte | Test perplexity | Train + validation seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alice | 1 | 23,078 | 9,796 | 39,184 | 12 | 3.3417 | 10.138 | 1.76 |
| Alice | 8 | 23,078 | 11,812 | 47,248 | 12 | 2.6731 | 6.378 | 2.71 |
| Pride | 2 | 105,750 | 10,084 | 40,336 | 12 | 2.8691 | 7.306 | 7.62 |
| Pride | 8 | 105,750 | 11,812 | 47,248 | 12 | 2.6100 | 6.105 | 12.08 |

At matched context on *Alice*, the eight-bank RTX model scored 4.6280–4.9632 and the exact bigram scored 3.3253. On *Pride*, the two-reflection RTX model scored 2.8655 and the exact trigram scored 2.7414. The 0.0036 bits/byte difference between the two-byte neural comparator and the optical model on *Pride* is small for a single corpus and one neural seed. Both eight-byte neural models achieved lower test loss. Every neural run selected the last available epoch. These results characterize the fixed training budget; longer training and hyperparameter tuning were outside the protocol.
