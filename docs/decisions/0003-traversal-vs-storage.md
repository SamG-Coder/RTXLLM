# ADR 0003 — When does ray traversal do the work? (proposed)

Status: proposed by Claude (#79); needs Codex review and owner sign-off only if it
weakens a safeguard (it does not; it tightens reporting).

## Context
AGENTS.md requires hardware ray traversal and multi-bounce propagation to be *central*, "not a
visual gimmick or a hidden vector search". Since #36 several components have used RT cores as
**addressed storage**: an integer key (row, column, byte position) becomes an axis-aligned ray
origin that can hit exactly one pre-known triangle, and the hit distance returns a stored number.
Examples: the KN layered dish read (#36, equal to KN by construction), E14 emission rows (#62), E15/E16
pair stations (#64/#66), E17 character table read once at load (#70), E18 byte rows (#72). These
are correct and useful plumbing, but they are array indexing with extra steps. Meanwhile #74 found
that the geometric two-word stations lose to ordinary KN-trigram interpolation with the same
calibration access, and E12/E3 found beams on par with (not better than) kNN/class models.

## Decision
Every RTX **stage** is labelled in its PR and README as one of the classes below. Mixed systems are
classified stage by stage: for example, E14's first-stage dish beam is T and its emission read is S.

- **T — traversal does the work.** Which primitive is hit (or in what order) depends on continuous
  geometry that was learned or composed, so the BVH search itself selects the evidence: beams over a
  learned layout, nearest-hit occlusion, multi-hop specular/transport composition, geometric suffix
  or episodic search where the match is *found* by traversal.
- **S — addressed storage.** The ray is aimed at a cell the host already knows from an integer key;
  a CPU array lookup would return the same value. Allowed as infrastructure (persistence, keeping
  data on device), never counted as evidence for the mission.

Three distinct kinds of evidence, each needed for a different claim:

- **(a) Parity — correctness.** Every RTX stage must agree with an exact CPU replay of the same
  intersections. Parity shows the hardware path is implemented correctly. It says nothing about
  value: a table baked from the same hits would also agree, by construction.
- **(b) Causal intervention — "the geometry matters".** Remove or randomise the geometry of a
  **T** stage, holding the candidate set, payloads and capacity fixed: random or shuffled layout,
  shuffled credit, permuted hit-to-payload routing, or a pencil instead of a beam. The result must
  get worse. Only a T stage that passes (b) supports a claim that traversal is doing useful work.
- **(c) Matched non-geometric model — "better or faster than conventional".** Any comparative
  quality or speed claim needs a non-geometric model with the **same** training data, calibration
  data and tuning grid (e.g. #74's KN trigram interpolated with λ tuned on the same Jekyll targets),
  scored on the same targets. Speed claims compare end-to-end time on the same hardware.

Negative results remain first-class and are reported with the same care.

## Consequences
- Existing merged and open work stays valid. Most earlier READMEs describe their stages without an
  explicit T/S label. Labels are added when a README is next touched; reviews of #62/#64/#66/#70/#72
  have already identified their S stages.
- Research priority shifts toward T-class mechanisms that could beat matched baselines: learned
  layouts whose traversal composes evidence across hops (E1/E3 style), occlusion as competitive
  selection, and geometry that generalises to unseen contexts — rather than more S-class tables.
