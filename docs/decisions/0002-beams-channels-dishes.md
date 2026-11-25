# 0002 — Beams, relation channels and focusing dishes (shared geometry)

Status: **proposed by Claude** (issue #11), pending Codex review. Extends ADR 0001; E1
scenes and semantics are unchanged.

## Context
Under ADR 0001 every query reaches only geometry written for exactly that query, so
retrieval is lookup-equivalent (E1, #8, #12). GOAL.md asks whether *spatial* structure
can carry information beyond what was written. That requires a query to be able to meet
geometry that **neighbouring** information wrote, and a layout in which neighbourhood
means something.

## Decision

### Scene additions
- `tri_relation i32[T]`: relation channel of each fact triangle (−1 for receivers). It is
  a material property — which "wavelength" a mirror reflects — not a destination id.
  Loaders treat a missing array as all −1 (ADR 0001 scenes stay valid).
- **Focusing dish** (`GeometryConfig.dish_radius > 0`, specular only): a fact is a disk of
  small equilateral facets (circumradius `fact_radius`) laid out on a triangular tiling
  perpendicular to `d_r`, lifted along `d_r` onto the plane through `m` normal to the
  bisector. Each facet's normal reflects `d_r` from **its own centroid** onto `p_o`. The
  central facet's centroid is exactly `m`, so the pencil ray never hits an edge. This
  decouples a memory's capture area (`dish_radius`) from receiver size (landing error
  ≤ facet size), so a beam can be caught without enlarging receivers (which causes
  occlusion, ADR 0001 / E1 sweep).

### Beam propagation (hop-synchronous)
For hop `h` with relation `r = program[h]`, each station `x` holding energy `E_x`:
1. Emits one beam along `d = relation_dir[r]`: a **core** ray from `p_x − span·d` with
   energy `core·E_x`, plus `rays` **halo** rays from `p_x − span·d + o_i`, where `o_i` are
   sunflower points (`r_i = width·sqrt((i+½)/rays)`, `θ_i = i·π(3−√5)`) in the plane ⟂ `d`
   using basis `t1 = normalize(d × a)`, `t2 = d × t1`, `a = x̂` if `|d_x| < 0.9` else `ŷ`;
   each halo ray carries `(1 − core)·E_x / rays`. `span = width`; `width = 0` means core only.
2. **Emission** rays see only fact triangles with `tri_relation == r`
   (RTX: one instance per relation, mask bit `1 << (1 + r)`), `tmax = fact_distance + 2·span + 0.5`.
3. On a fact hit: reflect about the facet normal (facts are single-valued here, `R = 1`);
   the **transport** ray sees only receivers (mask `0x1`), `tmax = transport_range`.
4. A receiver hit deposits energy at its station for hop `h + 1`; misses are lost.
5. Energy arriving at the same station in one hop is **summed** and re-emitted as one
   beam next hop (a station re-radiates what it received). Station energies below
   `energy_floor` (1e-5) are dropped to `lost`.
6. After the last hop the station energies are the answer distribution;
   `Σ answer + lost = 1`. Decode: argmax, abstain only if no energy arrived.

Closest hit, `t_min`, float32 and tie rules as ADR 0001. The CPU reference
(`rtxllm.cpu.beam.BeamTracer`) has an all-pairs mode and an exact broadphase mode that
are tested to give identical results.

### Placement
Station positions are part of the learned representation. `rtxllm.learning.placement`:
`random_lattice` (control), `force_layout` (springs between stations sharing ≥2
*written* (relation, object) pairs + inverse-square repulsion; uses only written facts),
`oracle_layout` (latent labels; upper bound only). All are rescaled to median
nearest-neighbour distance 1.

## Consequences
- With `width > 0` a query can be answered from neighbours' memories: generalisation is a
  physical consequence of layout + beam footprint. It is also a spatial kernel vote, so
  E3 must compare with kNN on the same layout (vector-retrieval control).
- Sharing costs memorisation: a written exception can be out-voted by its neighbours; the
  `core` fraction is the knob. Both are measured in E3.
- Hop-synchronous merging keeps rays per hop proportional to occupied stations
  (no `rays^hops` blow-up), which maps to one dispatch per hop on RTX.
