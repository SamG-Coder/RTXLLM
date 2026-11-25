# RTXLLM results scoreboard

Maintained by both agents. Each row says what the RTX stage does (ADR 0003: **T** = traversal
selects the evidence from continuous geometry; **S** = addressed storage, equivalent to an array
read), whether the geometry was causally tested (**b**: random/shuffled geometry with candidates
and capacity fixed), and how it compares with a **matched** non-geometric model (**c**: same data,
calibration and tuning). Every RTX path listed here matched an exact CPU replay unless noted.
Numbers are test-split results; lower perplexity or bits/byte is better. "Open" means the PR is
not merged yet.

_Last updated: 2026-11-26 (Claude)._

## Bottom line
- **The mechanisms work and are verified.** Multi-bounce composition, learned layouts, beams,
  episodic search, persistent scenes and on-device generation all run on RTX with CPU parity.
- **Traversal is causally necessary where it is T.** Scrambling the layout, credit or routing
  consistently hurts (E3, E3b, E12, E12b, E12c).
- **No matched advantage over conventional methods yet.** Every T-class component so far is on par
  with, or behind, a matched kNN, kernel, class or interpolated n-gram model. The strongest
  language results come from conventional estimators (Kneser–Ney counts, exact caches, interpolation)
  stored or searched through RTX (S-class), plus T-class neighbour smoothing that a dense kernel
  matches.
- **No end-to-end speed advantage** has been shown against an equally optimised CPU path (M6:
  ~1.4× over NumPy, dominated by host costs).

## Facts and composition (synthetic)
| Exp | RTX stage | Result | (b) intervention | (c) matched comparator |
|---|---|---|---|---|
| E1 multi-hop (#6/#9) | T: specular/transport hops | 2–3-hop written facts 98–100% | hop cap / scrambled geometry → ~0 | — (written facts only) |
| E3 beams (#43) | T: beam over learned layout | never-written facts 0.70–0.73 | random layout 0.02, pencil 0.016 | **kNN on same layout better** (0.77–0.79) |
| E3b ray feedback (#48) | T: layout learned from ray error | 0.61–0.65 from random start | shuffled credit 0.02 | below CPU force layout |

## Bytes
| Exp | RTX stage | Test result | (b) | (c) |
|---|---|---|---|---|
| E4 mirror routes (#13) | T: learned two-reflection routing | beats frozen/shuffled routes on Alice | yes | conventional bigram better |
| E5 optical cavity (#25) | T/S: cavity returns + detector rows | Dracula 2.795 bpb | one return worse (2.866) | KN-6 2.017 (much better) |
| E6/E8 layered KN (#36) | S: KN tables as dish layers | equals KN-6 (Pride 1.737, Dracula 2.017) | — | identical to KN by construction |
| E9 episodic + copy gate (#37/#41) | T: backward search finds matches; S: gate heights | War 2.201 (KN-5 2.219) | reversed heights worse | equivalent to a CPU suffix search (same predictions) |
| E10 pair addresses (#44, open) | T: paired-address search | Treasure 2.114, more coverage | matched-window control | same as CPU substring oracle |
| E11 pooled recall (#54, open) | T: tied paths | +0.0038 vs nearest (missed 0.005) | — | negative |
| E13 agreement gate (#56, open) | S: gate by prior mass | +0.0016 (missed 0.005) | — | negative |
| E5c adaptive regions (#27) | T: detector regions | negative | — | — |
| GRU comparator (#60, open) | none (conventional) | stateful GRU War 2.152 / Moby 2.442 (Claude rerun 2.146 / 2.446; cuDNN spread ~±0.006) | — | six-gram + geometric recall (2.114 / 2.280) ahead by ~0.03 / ~0.16 bpb. The GRU is small (214k parameters); a larger recurrent model may cross over |

## Words (Pride, 70/15/15; held-out authors where noted)
| Exp | RTX stage | Test result | (b) | (c) |
|---|---|---|---|---|
| E12 annular beam (#57) | T: beam over learned layout | 80.31 ppl (KN 82.26) | random layout 84.92 | class bigram 79.45 better |
| E12b ray-feedback layout (#81, open) | T | 82.86, unseen −9.6% | shuffled 85.19 | worse than KN overall |
| E12c full-disc beam (#86, open) | T | 79.58 | random +7.0%, shuffled payload +15.5% | **self-inclusive kernel 78.93 better**; on par with class/kNN. Post-hoc val: nearest-hit occlusion costs 0.5 ppl vs all-hit transmission |
| E14 persistent emission (#62, open) | T beam + S emission rows | reproduces E12 exactly | shuffled rows worse | — |
| E15/E16 pair stations (#64/#66/#68, open) | S | E16 beats E14 by 1.0–2.3% on 4 books | shuffled older word worse | **E14 ⊕ KN trigram better on all 4** (#77) |
| E17 spelling (#70, open) | S: char table read once | open-vocabulary spelling | shared-row control | char bigram on host |
| E18/E19 word cache (#72/#75, open) | S: byte rows | −0.51 to −0.67 bits/token; +0.03–0.05 with predecessor | displaced window worse | exact cache |

## What would count as progress
A T-class stage that passes (b) **and** beats its matched comparator (c) on held-out data, or an
end-to-end speed win against an equally optimised conventional implementation at equal quality.
Open directions under test: multi-reflection word continuation (#85), competitive two-hop
retrieval (#82).
