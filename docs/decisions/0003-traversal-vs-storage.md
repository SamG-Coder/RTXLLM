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
Every RTX call path is labelled in its PR and README as one of:

- **T — traversal does the work.** Which primitive is hit (or in what order) depends on continuous
  geometry that was learned or composed, so the BVH search itself selects the evidence: beams over a
  learned layout, nearest-hit occlusion, multi-hop specular/transport composition, geometric suffix
  or episodic search where the match is *found* by traversal.
- **S — addressed storage.** The ray is aimed at a cell the host already knows from an integer key;
  a CPU array lookup would return the same value. Allowed as infrastructure (persistence, keeping
  data on device), never counted as evidence for the mission.

Rules:
1. A result may be described as an "RTX/geometric model" improvement only if a **T** path is on its
   critical path and an ablation that replaces that path with its S/CPU equivalent loses.
2. Comparators get **matched access**: the same training data, calibration data and tuning grid as
   the geometric model (e.g. #74's interpolated KN trigram tuned on the same Jekyll targets).
3. Negative results remain first-class and are reported with the same care.

## Consequences
- Existing merged and open work stays valid; READMEs gain a one-line T/S label (done in reviews
  for #62/#64/#66/#70/#72).
- Research priority shifts toward T-class mechanisms that could beat matched baselines: learned
  layouts whose traversal composes evidence across hops (E1/E3 style), occlusion as competitive
  selection, and geometry that generalises to unseen contexts — rather than more S-class tables.
