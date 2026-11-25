# Two-hop RTX spelling recall

The learned predecessor layout recalled 171 of 2,048 unfamiliar spellings on the registered Jekyll validation prefix. The same beam on shuffled stations recalled 185. The pilot misses the success criterion in issue #82.

| Selector | Correct | Top-1 accuracy |
| --- | ---: | ---: |
| Learned layout, occlusion and reflection | 171 | 8.35% |
| Shuffled predecessor layout | 185 | 9.03% |
| Shuffled tilt assignments | 13 | 0.63% |
| Nearest predecessor position | 194 | 9.47% |
| E19-style predecessor weighting | 217 | 10.60% |

Each query is conditional on the target being outside the frozen Pride vocabulary. All methods see the same preceding 64 unfamiliar-word occurrences. Only predecessors with a dish in E14 participate. Of the 2,048 queries, 1,789 had a predecessor station and 1,559 received an optical vote in the learned layout. Missing predictions count as incorrect. The exact comparator gives each episode weight 8 for a matching predecessor and 1 otherwise. All methods break tied spelling votes by recency. The JSON records paired disagreements.

**Traversal (T).** Each episode owns a mirror at its predecessor’s projected position. Age sets its depth. Thirty-two parallel rays from the query station compete for the closest surfaces. Each first hit reflects using its triangle normal and hit point. A second Vulkan ray query intersects the receiver plane at x=2048. Source and receiver ranks differ by a fixed permutation within each 64-row block. For the intended routing all 43,920 first hits reached their assigned receiver. Shuffling the tilt offsets changed which receiver the reflected rays reached. The second segment transports the first-stage selection to its assigned band. **Addressed storage (S).** The host reads the historical spelling of the intersected receiver and sums one vote per ray.

Windows are evaluated in tiles with 64 preceding rows. The first ray starts below the query and has a finite range. A reflected ray to an unwritten or expired receiver casts no vote. Receiver locations depend only on row numbers. The runner checks both intersection IDs for every emitted ray against independent CPU triangle traversal in all three layouts. It repeats the first 93 queries with the scene physically truncated mid-tile and checks the full hit arrays. The corpus and model archive digests identify the input. This experiment has one validation prefix, one reflection and no held-out likelihood result. It gives no reason to replace E19 with this selector.

From the repository root with the RTX Vulkan environment and the frozen E14 archive:

```sh
python experiments/optical_recall/pilot.py E14.npz rtxllm-jekyll-43.txt report.json
```
