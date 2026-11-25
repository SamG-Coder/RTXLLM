# E1 — multi-bounce retrieval of held-out composed paths (CPU reference)

Claude, issue #4. Pre-registered in #1/#4. Scene contract: `docs/decisions/0001-architecture.md`.

## Protocol
World per seed (0, 1, 2): 512 stations on a jittered 8³ lattice (spacing 6, jitter ±1),
6 relations, 70% fact density, 5% two-valued facts (~2,160 atomic facts, all written into
geometry). Paths of length 2 and 3 split into disjoint train / val / test by query key;
per split × length: 300 unique, 100 ambiguous, 100 miss. Only atomic facts are written;
no composed path is stored anywhere. Geometry size chosen on **val** only, then **test**
was run once.

```
python experiments/e1_multihop/run.py sweep                              # val
python experiments/e1_multihop/run.py test --receiver-radius 0.05          # sealed test, primary
python experiments/e1_multihop/run.py test --receiver-radius 0.05 --masked # secondary
```
Host: Windows 11, Python 3.12, NumPy 2.5.5, brute-force CPU (no BVH); ~2 ms/query.

## Validation sweep (unique-path accuracy, mean of 3 seeds; L2 / L3)
`open` = transport legs can hit fact triangles (physical); `masked` = receivers only.

| receiver = fact radius | specular open | specular masked | transport (either) |
|---|---|---|---|
| 0.05 | 0.996 / 0.990 | 0.997 / 0.999 | 1.000 / 1.000 |
| 0.10 | 0.991 / 0.975 | 0.994 / 0.994 | 1.000 / 1.000 |
| 0.20 | 0.934 / 0.897 | 0.958 / 0.939 | 1.000 / 1.000 |
| 0.40 | 0.771 / 0.679 | 0.876 / 0.831 | 1.000 / 1.000 |

Selected: radius 0.05 (best specular-open val accuracy).

## Sealed test (radius 0.05, open)
| seed | mode | L2 unique | L3 unique | 1-hop cap | shuffled | lookup | graph | per-hop agree | crosstalk |
|---|---|---|---|---|---|---|---|---|---|
| 0 | specular | 1.000 | 0.987 | 0.000 / 0.000 | 0.000 / 0.000 | 0 / 0 | 1 / 1 | 0.996 | 0.5% |
| 1 | specular | 0.997 | 0.993 | 0.000 / 0.010 | 0.003 / 0.000 | 0 / 0 | 1 / 1 | 0.996 | 0.4% |
| 2 | specular | 0.990 | 0.983 | 0.000 / 0.000 | 0.000 / 0.000 | 0 / 0 | 1 / 1 | 0.991 | 0.5% |
| 0–2 | transport | 1.000 | 1.000 | ≤0.010 | ≤0.003 | 0 | 1 | 1.000 | 0 |

- Miss paths: abstention accuracy 1.000 (all seeds, both modes, L2 and L3).
- Ambiguous paths: energy on the correct answer set equals the graph ceiling
  (transport exactly; specular within 0.01), total variation ≤ 0.005. Ceiling mass is
  0.75–0.89 because some branches of a two-valued hop end in a missing fact.
- Cycles (unique L3 paths revisiting a station, 0–3 per seed): all answered correctly.
- Energy conservation error 0.0; ~5.1 rays per query.
- **Acceptance: 30/30 checks pass for each mode** (open and masked):
  ≥ 0.95 accuracy and ≥ 0.50 lift over 1-hop cap, whole-query lookup and shuffled geometry,
  ≥ 256 unique paths per seed and length.

## Interpretation (and what this does *not* show)
- Composition works: answers to never-stored multi-hop queries emerge only when rays are
  allowed to continue bouncing; capping at one hop or scrambling the geometry drops
  accuracy to ~0.
- **Specular memory has a measured capacity/occlusion cost** that transport mode does not:
  reflected rays fly through the scene and are intercepted by other stations and mirrors,
  growing steeply with primitive size (0.99 → 0.68 on L3). Transport mode is immune because
  its hop teleports by stored displacement; nothing between stations matters.
- E1 is a mechanism test. A perfect tracer equals the graph ceiling by construction; this is
  not evidence of generalisation, learning or language ability. RTX parity is #5 (Codex).
