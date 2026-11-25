# Byte-conditioned optical cavity

E5 found that repeated physical returns lost the identity of the latest byte. This follow-up retains that byte as one coordinate of the learned detector. The preceding seven bytes steer the E5 reflector, and the final continuous landing determines the other coordinate. Each observed detector row contains byte-labelled triangles whose widths encode the training continuation frequencies. Absent rows are empty space. The RTX reader can probe any computed address directly, including an address absent during training.

The evaluator batches identical requests derived from query positions. It reads the learned distribution with 256 stratified rays per requested row. The checkpoint contains the two meshes and incidence directions; training counts and the list of occupied addresses are unnecessary for prediction. The live generator rebuilds the two acceleration structures once, traces the prefix and probes the resulting row each step. Short prompts use the physically read one-byte backoff until they fill the requested prefix.

## Fresh-book result

The input contained 604,405 training bytes from *Dracula*. Every split resets context and begins scoring at offset eight. The curvature, grid, incidence map, 80/20 interpolation, 1% uniform escape and 256 readout samples were fixed before the run. Both checkpoints were saved before test scoring. Validation and test each have 129,508 targets.

| Model | Validation bits/byte | Test bits/byte |
| --- | ---: | ---: |
| Last byte + seven cavity returns | 2.7225 | 2.7949 |
| Last byte + one cavity return | 2.8162 | 2.8661 |
| First six prefix bytes shuffled across rows, latest two retained | 3.4435 | 3.5116 |
| Interpolated trigram | 2.8133 | 2.8582 |
| Interpolated fourth-order counts | 2.2816 | 2.3541 |
| Interpolated fifth-order counts | 2.0732 | 2.1669 |
| Interpolated eighth-order counts | 2.5093 | 2.6210 |

The seven-return model passed both preregistered 0.05 bits/byte margins. All physical cavity paths completed. Its trained readout contained 7,107 occupied conditional rows and 78,130 readout triangles. The compressed checkpoint occupied 592,090 bytes. The one-return model used 1,547 rows, 23,320 triangles and 337,027 bytes. Respective test coverage of written rows was 99.57% and 99.84%. The seven-return model uses extra memory and earlier input bytes. This comparison therefore measures that complete configuration. The fourth-, fifth- and eighth-order count controls all achieved lower test loss.

Both learned models mix a directly probed last-byte distribution with the probed higher-order row; an unwritten row uses the lower-order distribution in full. The count controls use exact training frequencies and recursively interpolate each observed order with the lower order. The accompanying JSON files include every evaluated count order.

## Reproduce

On Windows with an RTX 5080, NVIDIA driver 616.64, Python 3.12, NumPy 2.5.5 and SlangPy 0.43.1:

```powershell
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
$python = "$env:TEMP/rtxllm-vk-env/Scripts/python"
$book = "$env:TEMP/rtxllm-dracula-345.txt"
$out = "$env:TEMP/rtxllm-conditioned"
Invoke-WebRequest https://raw.githubusercontent.com/GITenberg/Dracula_345/5216d6ea255fbf8966426a160b68251b18f16907/345.txt -OutFile $book
& $python -B experiments/conditioned_cavity/experiment.py validate $book $out
& $python -B experiments/conditioned_cavity/experiment.py test $book $out
& $python -B experiments/conditioned_cavity/generate.py "$out/conditioned-7.npz" "The Count " --count 96 --seed 0
```

The first command after download checks the corpus digest and fits only the training prefix. The test command reloads both saved models. The generation command requires only the geometry checkpoint.
