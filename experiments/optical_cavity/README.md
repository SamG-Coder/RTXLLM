# E5 optical cavity

One curved reflector serves every byte. Each byte selects a fixed incidence direction. A ray travels from its current point on a plane to the mirror, reflects, and returns to the plane. Its actual intersection point becomes the launch point for the next byte. The shader carries a continuous position between returns. All histories share the same triangulated paraboloid and the same detector plane.

Training assigns each final landing to a spatial cell and measures its next-byte frequencies. The writer records those frequencies as the widths of byte-labelled triangle strips. The checkpoint also contains the mirror, plane, and incidence directions. Scoring reloads the triangles, physically probes each readout row with 256 stratified rays, and reuses the resulting distributions for repeated cells. A separate set of strips supplies the 20% one-byte backoff, followed by 1% uniform escape. The learned probability tables are absent from the checkpoint.

## Result

**The eight-return design failed its preregistered improvement criterion.** On 64,636 held-out next bytes, eight returns scored 3.4770 bits/byte and one return scored 3.4598. Replacing the first seven input bytes with another test row's prefix, preserving the last byte, raised the eight-return loss to 3.9206. Earlier inputs measurably affect the physical state and predictions. That dependence did not improve the comparison with the separately fitted one-return model. All primary and shuffled paths completed.

| Test method | Bits per byte |
| --- | ---: |
| Exact interpolated trigram | 2.8392 |
| Traced one-byte backoff alone | 3.4589 |
| One cavity return | 3.4598 |
| Eight cavity returns | 3.4770 |
| Exact bigram with 20% unigram interpolation | 3.5144 |
| Eight returns with shuffled prefix | 3.9206 |

The traced one-byte backoff applies uniform escape directly to empirical one-byte strip votes. The statistical bigram additionally interpolates the unigram prior, so its higher loss cannot establish a quality advantage for optical storage. The strongest observed comparator was the trigram.

Training used 301,669 bytes of *Frankenstein*. Each split resets history, and all scores begin at offset eight. The validation sweep selected a 128 by 128 detector grid for both models, with retention 0.50 for one return and 0.25 for eight. Both choices were frozen before the test. The eight-return checkpoint occupied 283,310 compressed bytes, with 32,770 cavity triangles (including the plane) and 13,060 learned readout triangles.

A post-hoc validation diagnostic measured how well training cell majorities recover the final input byte. One return occupied 81 cells and recovered 99.96%. Eight returns occupied 291 cells and recovered 74.96%. This identifies loss of final-byte identity under spatial binning as one plausible cause of the failed quality criterion. It does not establish the effect of changing the encoder or grid on fresh text.

## Reproduction

Use Python 3.12 with SlangPy 0.43.1 and NumPy 2.5.5. The recorded host had an RTX 5080 with driver 616.64. The input checksum is checked before training. Validation writes the selected scenes, and test scoring reads those scenes. The code also compares a flat-mirror path with its analytic landing, checks that reversing a three-byte history changes its curved-mirror landing, and verifies a 3:1 detector strip and an empty detector.

```powershell
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
$python = "$env:TEMP/rtxllm-vk-env/Scripts/python"
$book = "$env:TEMP/rtxllm-frankenstein-84.txt"
$out = "$env:TEMP/rtxllm-cavity"
Invoke-WebRequest https://raw.githubusercontent.com/GITenberg/Frankenstein_84/c98a483bd1f34e7366126b81bcedd1a6faee636e/84-0.txt -OutFile $book
& $python -B experiments/optical_cavity/run.py validate $book $out
& $python -B experiments/optical_cavity/run.py test $book $out
& $python -B experiments/optical_cavity/check_aliasing.py $book $out
```

The adjacent JSON files contain the full validation sweep and frozen test scores. They accompany the preregistered protocol in issue #21.
