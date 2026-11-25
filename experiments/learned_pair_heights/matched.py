"""#74 (Claude): matched conventional trigram comparators for E16 learned pair heights.

    python experiments/learned_pair_heights/matched.py words.npz E16 BOOK_DIRECTORY matched.json

Same targets, vocabulary and training stream as E16 (#66) and its transfer test (#68).
Conventional comparators (CPU only, no geometry):
  interp  P = (1 - lam) * E14 + lam * KN3, lam chosen on the E16 Jekyll calibration
          targets from E16's own height grid, before any evaluation book is scored;
  backoff interpolated modified-KN trigram whose lower-order distribution is E14
          (residual + backoff mass * E14); no tuning.
E14 and E16-grouped probabilities are read from the frozen RTX archives exactly as
in #68. Paired 256-target block bootstrap (2,000 draws, seed 67) for E16 grouped vs
the better conventional comparator.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from learn_heights import GRID, JEKYLL, WALLPAPER, PairScene, Reader, book_body, digest, tokens
from transfer import BOOKS, HERE
from trigram import table

EVAL = {"wallpaper-1952": WALLPAPER, **BOOKS}


def kn_tables(train_ids, V):
    top = Counter(zip(train_ids, train_ids[1:], train_ids[2:]))
    mid = Counter((b, c) for a, b, c in top)                      # continuation counts
    uni = Counter(b for a, b in set(zip(train_ids, train_ids[1:])))
    pu = np.array([uni[w] + 1e-9 for w in range(V)], float)
    return table(top), table(mid), pu / pu.sum()


def lookup(row, c, lower):
    if row is None:
        return lower
    residual, w = row
    return residual.get(c, 0.0) + w * lower


def conventional(tri, bi, uni, a, b, c, base):
    kn3 = np.empty(len(c))
    back = np.empty(len(c))
    for i, (x, y, z) in enumerate(zip(a.tolist(), b.tolist(), c.tolist())):
        row = tri.get((x, y))
        kn3[i] = lookup(row, z, lookup(bi.get((y,)), z, uni[z]))
        back[i] = lookup(row, z, base[i])
    return kn3, back


def block_interval(delta):
    starts = np.arange(0, len(delta), 256)
    lengths = np.diff(np.r_[starts, len(delta)])
    sums = np.add.reduceat(delta, starts)
    idx = np.random.default_rng(67).integers(len(starts), size=(2000, len(starts)))
    means = sums[idx].sum(1) / lengths[idx].sum(1)
    return np.quantile(1 - np.exp(-means), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "heights", "books", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    frozen = json.loads((HERE / "fit.json").read_text(encoding="utf-8"))
    identity = digest(args.model)
    for name, sha in frozen["archive_sha256"].items():
        if digest(args.heights / name) != sha:
            raise ValueError("E16 elevations differ from the frozen experiment")
    model = Reader(args.model)
    grouped = PairScene(model, args.heights / "learned.npz", identity)
    vocab = {w: i for i, w in enumerate(model.words)}
    train, _, _ = tokens(args.books)
    tri, bi, uni = kn_tables([vocab.get(w, 0) for w in train], len(vocab))

    def predictors(name, sha):
        seq = model.encode(book_body(args.books / f"rtxllm-{name}.txt", sha))
        a, b, c = seq[:-2], seq[1:-1], seq[2:]
        base = model.predict(b, c)[:, 1].astype(float)
        e16 = grouped.read(a, b, c, base)[:, 0]
        kn3, back = conventional(tri, bi, uni, a, b, c, base)
        return c, {"e14": base, "e16_grouped": e16, "kn3": kn3, "backoff_e14": back}

    # Selection on the E16 calibration book only.
    _, cal = predictors("jekyll-43", JEKYLL)
    grid = {float(l): float(np.exp(-np.log((1 - l) * cal["e14"] + l * cal["kn3"]).mean())) for l in GRID}
    lam = min(grid, key=grid.get)
    report = {"calibration": {"book": "jekyll-43", "targets": len(cal["e14"]), "interp_grid_ppl": grid,
                              "selected_lambda": lam,
                              "ppl": {k: float(np.exp(-np.log(v).mean())) for k, v in cal.items()}},
              "books": {}}
    print("selected lambda", lam, flush=True)
    wins = 0
    for name, sha in EVAL.items():
        c, p = predictors(name, sha)
        p["interp_e14_kn3"] = (1 - lam) * p["e14"] + lam * p["kn3"]
        nll = {k: -np.log(v) for k, v in p.items()}
        ppl = {k: float(np.exp(v.mean())) for k, v in nll.items()}
        best = min(("interp_e14_kn3", "backoff_e14"), key=ppl.get)
        delta = nll[best] - nll["e16_grouped"]
        gain = float(1 - np.exp(-delta.mean()))
        wins += gain >= 0.005
        report["books"][name] = {"targets": len(c), "unk_share": float(np.mean(c == 0)), "ppl": ppl,
                                 "best_conventional": best, "e16_gain_vs_best": gain,
                                 "e16_gain_95_interval": block_interval(delta)}
        print(name, json.dumps(report["books"][name]), flush=True)
    report["geometry_adds_value"] = bool(wins >= 3)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
