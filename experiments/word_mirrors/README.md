# Specular paths through word memory

On 23,124 Pride and Prejudice validation targets, adding a first reflection lowered next-token perplexity from 81.455 to 81.420. The second reflection gave 81.422. Direct graph continuation scored 81.326 at the same maximum depth. The registered criterion in issue #85 required an improvement over the graph, so the experiment missed its target.

| Evidence source | First hit | Up to one reflection | Up to two reflections |
| --- | ---: | ---: | ---: |
| Trained mirror scene | 81.455 | 81.420 | 81.422 |
| Shuffled mirror orientations | 81.497 | 81.615 | 81.649 |
| Follow trained partner IDs from the original first hits | 81.455 | 81.334 | 81.326 |
| Same graph, stop wherever its corresponding optical ray missed | 81.455 | 81.36 | 81.38 |

The last row is a post hoc control. It uses optical survival information and measures the effect of routing with the same number of evidence contributions per ray. The remaining gap favors following the partner links. On training-unseen bigrams, adding the first reflection worsened the trained scene's perplexity from 1,959 to 1,972. This is a descriptive comparison on one validation segment and seed.

## Peer-requested validation controls

The table below gives the lowest validation perplexity per depth over mixture weights 0.05, 0.1, 0.2 and 0.3. Each row uses the same 23,124 targets. These are post hoc validation minima. The registered fixed-weight analysis above stands.

| Source | One stage | Two stages | Three stages |
| --- | ---: | ---: | ---: |
| Trained mirrors, annulus | 81.455 | 81.242 | 81.207 |
| Trained mirrors, full disc | 80.503 | 80.459 | 80.555 |
| Shuffled slopes, full disc | 80.524 | 80.692 | 80.784 |
| Graph from full-disc hits | 80.503 | 80.402 | 80.561 |
| Graph with full-disc optical survival | 80.503 | 80.212 | 80.236 |
| Self-inclusive Gaussian, final station distribution | 79.877 | 79.828 | 79.896 |
| Self-inclusive Gaussian, distributions averaged across stages | 79.877 | 79.793 | 79.776 |

The full disc gives the word access to its own mirror. Its first reflection helps by 0.044 perplexity and its second reflection gives back that gain. The dense Gaussian control uses the same station coordinates, successor payloads and evidence counts. Its row-normalized transition matrix weights each destination by its training count times a Gaussian of width 1. Successive matrix applications propagate evidence to the second and third station distributions. The mirrors additionally receive high-dimensional successor information during fitting through their Hellinger partner links. Even with that information, the best full-disc optical score trails a single Gaussian step. The full grid and controls for both launch shapes are in the JSON.

**Learning and traversal (T).** Start with the frozen E14 word positions and 1,685 six-triangle dishes. Use only Pride training successors to select each word’s closest distinct Hellinger neighbour. Fit its mirror normal so that a ray incident along the original E12 direction reflects from the centre toward that neighbour. Shear the dish along the incident axis to preserve its projected area. The slope cap of 4 affected 94 mirrors. Shuffling reassigns the fitted slopes with centres, areas and payloads fixed. Overlapping projected dishes can change first-hit order after a shear.

For inference, launch the original 512-ray annulus per station. Every hit in the same acceleration structure supplies a triangle normal and an intersection point for the next ray. A miss terminates the path. The limit of two reflections bounds returns and cycles. Per-ray evidence is divided equally across reached stations, then pooled with the original Gaussian radial weight. **Readout.** The host uses the selected stations' training successor distributions and the frozen 0.1 mixture with the E12 bigram prior. This pilot evaluates the routing rule with the existing evaluator. The partner IDs enter the graph controls and mirror fitting only.

Of 862,720 rays, 139,633 hit a first mirror, 87,279 reached a second, and 28,730 reached a third. The second hit reached the designed partner 52,508 times and another station 34,771 times. Third hits reached 653 distinct stations, including 2,135 paths that returned to their first station. For each orientation condition the runner verifies all three hit IDs against brute-force CPU triangle intersections for 512 rays sampled across miss and hit depths. RTX 5080 hardware, the layout archive digest, per-depth hit counts and full-precision scores are in [validation.json](validation.json).

Run from the repository root with the E12 pinned Pride edition and the archive built by [E14](../word_emission/README.md).

```sh
python experiments/word_mirrors/pilot.py BOOK_DIRECTORY E14.npz validation.json
```

## Generate from saved mirrors

The generator persists the fitted mirror vertices, station positions and radii, vocabulary and E14 emission triangles in one archive. At each generation step with a dish-owning predecessor it launches a fresh 512-ray annulus. Vulkan traversal selects the stations at up to three successive hits (T). The host pools their votes, and a second Vulkan scene supplies the emission and prior probabilities at their addressed rows and columns (S). The sampler draws from the resulting vocabulary distribution. A predecessor without a dish uses the prior. The build uses the pinned Pride training portion and the corresponding E14 archive. A fresh inference process takes only the fitted archive, prompt, depth and sampling settings.

```sh
python experiments/word_mirrors/live.py build BOOK_DIRECTORY E14.npz fitted-mirrors.npz
python experiments/word_mirrors/live.py generate fitted-mirrors.npz Elizabeth --count 128 --seed 3
python experiments/word_mirrors/check_live.py BOOK_DIRECTORY fitted-mirrors.npz live.json
```

Use a new output filename for the build. Set `--max-hits 1` to disable both reflections or `--max-hits 2` to allow one. Outputs include the exact token bytes in hex and the number of hits at each depth. The registered annular beam and fixed 0.1 mixture make this an executable version of the original pilot.

On RTX 5080 the 1,005,777-byte archive reproduced the same 128 tokens in two fresh processes. The sample began “Elizabeth, that there was a word. he left the evening last week for your proposals”. The 125 station queries produced 18,760 first hits, 13,756 second hits and 6,185 third hits. One warm run took 0.570 seconds for 128 tokens, including host aggregation and sampling. Three generated steps used only the prior. For each of 11 predecessor words, the verifier compared all 3,338 next-token probabilities to a CPU triangle tracer and independently pooled training distributions. The largest absolute error across both depths was 3.21 × 10⁻⁸. Enabling reflections changed nine distributions. These checks establish that reflected paths affect a running generator. The validation comparison above still favors the Gaussian model. [Recorded generation](live.json)
