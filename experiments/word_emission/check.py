"""Test the restored E14 word predictor against the dense E12 evaluator."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scene import WIDTH, Reader, Model, tokens
from run import beam_q


def metrics(p, unseen):
    lp = -np.log(p.astype(float))
    return {
        "ppl": float(np.exp(lp.mean())),
        "unseen_ppl": float(np.exp(lp[unseen].mean())),
        "seen_ppl": float(np.exp(lp[~unseen].mean())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("books", "archive", "report"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    tr, _, te = tokens(args.books)
    model = Model(tr)
    ev = model.enc(te)
    prev, nxt = ev[:-1], ev[1:]
    unseen = model.B[prev, nxt] == 0
    reader = Reader(args.archive)
    with np.load(args.archive, allow_pickle=False) as data:
        pos, ids, radii = [data[k] for k in ("positions", "ids", "radii")]
    q, has, _, _ = beam_q(model, ids, pos, radii, reader.first.hits, 1.0, WIDTH)
    prior = model.kn(prev, nxt)
    row = np.full(model.V, -1)
    row[ids] = np.arange(len(ids))
    r = row[prev]
    use = (r >= 0) & has[np.maximum(r, 0)]
    ref = np.where(use, 0.9 * prior + 0.1 * q[np.maximum(r, 0), nxt], prior)
    t0 = perf_counter()
    p = reader.predict(prev, nxt)
    wall = perf_counter() - t0
    delta = np.max(np.abs(p - np.column_stack((prior, ref))), axis=0)
    if np.any(delta > 2e-6):
        raise AssertionError(f"Target probability parity error {delta}")
    vocab = np.arange(model.V)
    for w in (0, ids[0], ids[-1], prev[0], prev[len(prev) // 2]):
        dist = reader.predict(np.full(model.V, w), vocab)
        base = model.kn(np.full(model.V, w), vocab)
        expected = 0.9 * base + 0.1 * q[row[w]] if row[w] >= 0 and has[row[w]] else base
        np.testing.assert_allclose(dist[:, 1], expected, rtol=0, atol=2e-6)
        np.testing.assert_allclose(
            dist.sum(axis=0, dtype=np.float64), 1, rtol=0, atol=2e-6
        )
    shuffled = Reader(args.archive, shuffle=True).predict(prev, nxt)
    np.testing.assert_array_equal(p[:, 0], shuffled[:, 0])
    report = {
        "targets": len(prev),
        "unseen_targets": int(unseen.sum()),
        **reader.sizes,
        "archive_bytes": args.archive.stat().st_size,
        "prior": metrics(p[:, 0], unseen),
        "beam": metrics(p[:, 1], unseen),
        "shuffled": metrics(shuffled[:, 1], unseen),
        "max_abs_prior_error": float(delta[0]),
        "max_abs_beam_error": float(delta[1]),
        "target_readout_s": wall,
    }
    original = (
        Path(__file__).resolve().parents[1]
        / "e12_word_layout/results/test_s1.0_l0.1_k10.json"
    )
    recorded = json.loads(original.read_text(encoding="utf-8"))
    for key, refkey in (("prior", "kn"), ("beam", "beam_force_s1.0_l0.1")):
        for metric, original_metric in (("ppl", "ppl"), ("unseen_ppl", "ppl_unseen")):
            np.testing.assert_allclose(
                report[key][metric], recorded[refkey][original_metric], rtol=0.001
            )
    s = json.dumps(report, indent=2)
    print(s)
    args.report.write_text(s + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
