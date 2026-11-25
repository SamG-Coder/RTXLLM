# 0001 — Spatial associative memory: scene contract and bounce semantics

Status: **accepted by Claude and Codex** for E1 (issue #1, CPU PR #9 and RTX PR #6).
Authors: Claude (draft), Codex (requirements in #1).

## Context
GOAL.md requires information to exist as persistent geometry with stable
identity, queried by genuine RTX intersections over multiple bounces. Issue #1
agreed: Vulkan inline ray queries via SlangPy (pinned `slangpy==0.43.1`), a
brute-force NumPy CPU reference, held-out composed multi-hop paths with
atomic facts visible, three seeds, sealed test split.

## Decision

### Representation
| Concept | Geometry | Identity / payload |
|---|---|---|
| Station (entity / token) | closed icosahedron receiver (20 tris), circumradius `receiver_radius`, centred on `station_pos[s]` | `tri_station = s` on each receiver triangle — the only id in the scene |
| Relation `r` | unit emission direction `relation_dir[r]` (≤6: one vertex of each antipodal icosahedron pair, rotated by a fixed seed so they avoid lattice axes) | none |
| Fact `(s, r, o)` | one equilateral triangle, circumradius `fact_radius`, centroid `m = p_s + (fact_distance + i·stack_gap)·d_r` | `tri_station = -1`; `tri_reflectance = R`; **specular**: normal `normalize(normalize(p_o − m) − d_r)`, no destination stored; **transport**: normal `−d_r`, `tri_displacement = (p_o − 3·receiver_radius·u) − m`, `u = normalize(p_o − m)` |
| Multi-valued `(s, r) → {o_0..o_{n-1}}` | stack along `d_r`, `R_i = 1/(n − i)` (equal energy split) | — |

Facts are single triangles because the emission ray passes exactly through
`m`; with a two-triangle quad the hit lies on the shared diagonal and
non-watertight intersection dropped ~1% of hits (observed in #4 development).

### Scene file (`scene.npz`)
`tri_vertices f32[T,3,3]`, `tri_kind u32[T]` (0 receiver, 1 fact),
`tri_station i32[T]`, `tri_reflectance f32[T]`, `tri_displacement f32[T,3]`,
`station_pos f32[S,3]`, `relation_dir f32[R,3]`, `params` (JSON string of
`TraceParams`). Receivers occupy tri ids `[0, 20·S)` in station order, facts
follow. Queries: `start i32[Q]`, `program i32[Q,K]` padded with −1.

### Bounce semantics (both backends)
1. **Emission.** Hop `h` of query `q` from station `x`: origin
   `station_pos[x] + emit_offset·d`, `d = relation_dir[program[q,h]]`,
   `tmax = emit_range`, `t_min` as below. `emit_offset = 1.5·receiver_radius`
   (> circumradius), so an emission ray starts outside its own convex receiver
   moving away from it and cannot self-hit. `emit_range = fact_distance +
   stack_gap + 2·fact_radius` — a station's finite reach; an absent fact is a
   miss, not a hit on a distant station.
2. **Closest hit**, two-sided triangles, `t_min < t < tmax`, `t_min = 1e-3`
   on every leg. New origins are `o + t·d` in float32 with **no** normal
   offset; `t_min` alone prevents re-hitting the triangle just left.
3. **Masks.** Instance mask bit `0x1` = receivers, `0x2` = facts (one BLAS
   per kind is fine; report tri id as receiver/fact offset + primitive index).
   Emission legs use cull mask `0x3`. Transport and arrival legs use `0x3`
   when `transport_hits_facts`, else `0x1`.
4. **Miss** → lineage energy added to `miss[q]`.
5. **Receiver hit (any leg)** → hop complete at station `x = tri_station`.
   If `h + 1 < len(program)`: re-emit hop `h+1` from `x` (step 1; the
   incoming direction is discarded). Else add energy to `answer[q, x]`.
   A reflected ray may hit its own subject's receiver; that is a genuine
   occlusion and counts as arrival there.
6. **Fact hit** with energy `e`, reflectance `R`:
   - follow branch `R·e`: **specular** — reflect `d' = normalize(d − 2(d·n)n)`,
     `n = normalize(cross(v1 − v0, v2 − v0))`, from the hit point, leg =
     transport, `tmax = transport_range`; **transport** — origin `hit + disp`,
     direction `normalize(disp)`, leg = arrival, `tmax = arrival_range =
     6·receiver_radius`.
   - transmit branch `(1 − R)·e`: same direction from the hit point, same
     leg, `tmax_remaining = tmax − t`.
   - A branch with energy exactly 0 is not created. A branch with
     `0 < energy < energy_floor` is not traced and its energy goes to `floor[q]`.
   - Fact hits on non-emission legs are counted as crosstalk.
7. **Event cap.** Each lineage carries its intersection count (inherited by
   both children). An intersection that would make it exceed `max_events`
   (16) drops the lineage's energy into `overflow[q]`.
8. **Order.** Branch processing order is unspecified; all outputs are sums.
9. **Conservation.** `Σ_x answer[q,x] + miss[q] + floor[q] + overflow[q] = 1`
   (tested to 1e-5).

### Decoding and ties
Abstain when `Σ_x answer[q,x] < 0.5`, else `argmax_x answer[q,x]`, ties to the
lowest station id. CPU closest-hit ties pick the lowest tri id; the GPU tie
rule is unspecified, so **parity compares per-hop receiver station ids and
final decoded answers**, not tri ids (Codex's point in #1).

### Evaluation (E1)
Methods: `graph` (symbolic ceiling, the only method allowed to read facts by
index), `lookup` (whole-query table of training queries incl. atomic facts),
`trace`, `trace_1hop` (program capped at first relation), `trace_shuffled`
(stations moved onto each other's locations after writing; fact geometry
untouched). Buckets per split × length {2, 3}: 300 unique, 100 ambiguous,
100 miss. Acceptance per seed and length on unique test paths: accuracy ≥ 0.95
and lift ≥ 0.50 over each of 1-hop, lookup and shuffled. Geometry size chosen
on `val`; `test` evaluated once.

## Consequences
- Specular mode makes the flight path physical, so occlusion and crosstalk are
  real costs; transport mode is immune to flight-path occlusion by
  construction. This is the trade-off the stress tests in #1 measure.
- The scene stores no fact table, so the RTX backend cannot "cheat" by graph
  lookup; the CPU reference has no BVH, so its correctness is auditable.
- E1 establishes mechanism, not generalisation: by construction a perfect
  tracer matches the graph ceiling. Generalisation (beam spread over learned
  placement) and language prediction are separate experiments.
