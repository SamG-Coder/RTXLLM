"""Frozen transfer test of E16, using the protocol in issue 67."""

import argparse
import json
from pathlib import Path
import numpy as np
from learn_heights import (
    PairScene,
    Reader,
    book_body,
    cpu_lookups,
    digest,
    emissions,
    tokens,
)

HERE = Path(__file__).resolve().parent
BOOKS = {
    "holmes-1661": "2a0ce5568625a304f54fbb6bfdeeadf527a1437254b5f0e06b5a9e233d4784a2",
    "war-36": "0c9bea907452abd6e9a31d9152188ef5007f0e2d3e4a52d7a6a627a37616249f",
    "moby-2701": "baf9b04fcf6e0008f30eaaa6697277751448c1070b20175e0082f28cbb161fd3",
}


def paired_interval(delta):
    starts = np.arange(0, len(delta), 256)
    lengths = np.diff(np.r_[starts, len(delta)])
    sums = np.add.reduceat(delta, starts)
    rng = np.random.default_rng(67)
    idx = rng.integers(len(starts), size=(2000, len(starts)))
    means = sums[idx].sum(1) / lengths[idx].sum(1)
    return np.quantile(1 - np.exp(-means), [0.025, 0.975]).tolist()


def score(model, scenes, rows, heights, seq):
    a, b, c = seq[:-2], seq[1:-1], seq[2:]
    base = model.predict(b, c)[:, 1].astype(float)
    pred = {"one_word": base}
    groups, mass, fixed = cpu_lookups(rows, a, b, c)
    seen = groups >= 0
    want = {
        "fixed": fixed,
        "global": seen * heights[0],
        "grouped": np.where(seen, heights[1:][groups], 0),
    }
    for label, scene in scenes.items():
        out = scene.read(a, b, c, base)
        w = want[label]
        np.testing.assert_allclose(out[:, 2], w, rtol=0, atol=2e-6)
        np.testing.assert_allclose(
            out[:, 0], (1 - w) * base + w * mass, rtol=0, atol=2e-6
        )
        pred[label] = out[:, 0]
    older = np.random.default_rng(67).permutation(a)
    g, m, _ = cpu_lookups(rows, older, b, c)
    w = np.where(g >= 0, heights[1:][g], 0)
    out = scenes["grouped"].read(older, b, c, base)
    np.testing.assert_allclose(out[:, 2], w, rtol=0, atol=2e-6)
    np.testing.assert_allclose(out[:, 0], (1 - w) * base + w * m, rtol=0, atol=2e-6)
    pred["shuffled"] = out[:, 0]
    nll = {k: -np.log(p) for k, p in pred.items()}
    delta = nll["one_word"] - nll["grouped"]
    return {
        "targets": len(c),
        "unk_targets": int(sum(c == 0)),
        "unk_share": float(np.mean(c == 0)),
        "support": np.bincount(groups + 1, minlength=7).tolist(),
        "ppl": {
            k: {
                t: float(np.exp(v[mask].mean()))
                for t, mask in (
                    ("all", np.ones(len(c), bool)),
                    ("known", c != 0),
                    ("unknown", c == 0),
                )
            }
            for k, v in nll.items()
        },
        "gain": float(1 - np.exp(-delta.mean())),
        "gain_95_interval": paired_interval(delta),
    }, float(delta.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "fixed", "heights", "books", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    frozen = json.loads((HERE / "fit.json").read_text())
    prior = json.loads((HERE.parent / "word_pairs" / "fit.json").read_text())
    paths = {
        "fixed": args.fixed,
        "global": args.heights / "global.npz",
        "grouped": args.heights / "learned.npz",
    }
    identity = digest(args.model)
    if identity != prior["model_sha256"] or digest(args.fixed) != prior["pair_sha256"]:
        raise ValueError("E14/E15 archives differ from frozen experiment")
    for name, sha in frozen["archive_sha256"].items():
        if digest(args.heights / name) != sha:
            raise ValueError("E16 elevations differ from frozen experiment")
    model = Reader(args.model)
    scenes = {k: PairScene(model, path, identity) for k, path in paths.items()}
    vocab = {w: i for i, w in enumerate(model.words)}
    train, _, _ = tokens(args.books)
    rows = emissions([vocab.get(w, 0) for w in train])
    report = {"books": {}}
    total_nats = 0
    for book, sha in BOOKS.items():
        seq = model.encode(book_body(args.books / f"rtxllm-{book}.txt", sha))
        result, delta = score(model, scenes, rows, np.array(frozen["heights"]), seq)
        report["books"][book] = result
        total_nats += delta
        print(
            book,
            result["targets"],
            result["gain"],
            result["gain_95_interval"],
            flush=True,
        )
    report["targets"] = sum(v["targets"] for v in report["books"].values())
    report["pooled_gain"] = float(1 - np.exp(-total_nats / report["targets"]))
    report["passes"] = all(v["gain"] >= 0.01 for v in report["books"].values())
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
