"""Post-hoc conventional word trigram comparator on E16's frozen targets."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from learn_heights import WALLPAPER, book_body, scores, tokens


def discounts(counts):
    numbers = Counter(counts)
    n1, n2, n3, n4 = (numbers[i] for i in range(1, 5))
    y = n1 / (n1 + 2 * n2)
    return (0, 1 - 2 * y * n2 / n1, 2 - 3 * y * n3 / n2, 3 - 4 * y * n4 / n3)


def table(counts):
    d = discounts(counts.values())
    rows = defaultdict(dict)
    for key, count in counts.items():
        rows[key[:-1]][key[-1]] = count
    model = {}
    for ctx, row in rows.items():
        n = sum(row.values())
        residual = {word: (c - d[min(c, 3)]) / n for word, c in row.items()}
        backoff = sum(d[min(c, 3)] for c in row.values()) / n
        np.testing.assert_allclose(
            sum(residual.values()) + backoff, 1, atol=1e-12, rtol=0
        )
        if min(residual.values()) < 0 or backoff < 0:
            raise ArithmeticError("Invalid discount")
        model[ctx] = (residual, backoff)
    return model


def mix(row, target, prior):
    if row is None:
        return prior
    residual, w = row
    return residual.get(target, 0) + w * prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "books", "report"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    with np.load(args.model, allow_pickle=False) as archive:
        words = [b"<UNK>", *archive["vocabulary"].tolist()]
    vocab = {w: i for i, w in enumerate(words)}
    tr, _, _ = tokens(args.books)
    s = [vocab.get(w, 0) for w in tr]
    top = Counter(zip(s, s[1:], s[2:]))
    # Lower-order counts are distinct left extensions observed in training.
    mid = Counter((b, c) for a, b, c in top)
    unigram = Counter(b for a, b in set(zip(s, s[1:])))
    uni = np.array([unigram[w] + 1e-9 for w in range(len(vocab))], float)
    uni /= uni.sum()
    bigrams, trigrams = table(mid), table(top)
    body = book_body(args.books / "rtxllm-wallpaper-1952.txt", WALLPAPER)
    seq = [vocab.get(w, 0) for w in re.findall(rb"[a-z0-9]+|[^\sa-z0-9]", body.lower())]
    pred, seen = [], []
    for a, b, c in zip(seq, seq[1:], seq[2:]):
        prior = mix(bigrams.get((b,)), c, uni[c])
        row = trigrams.get((a, b))
        pred.append(mix(row, c, prior))
        seen.append(row is not None)
    report = {
        "targets": len(pred),
        "modified_kn_trigram": scores(np.array(pred), np.array(seen)),
    }
    result = json.dumps(report, indent=2)
    args.report.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
