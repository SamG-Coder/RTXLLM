"""Post-hoc validation comparison under the one-return readout triangle budget."""

import argparse
from pathlib import Path

import numpy as np
from experiment import (
    BACKOFF,
    ESCAPE,
    GRID,
    TriangleScene,
    addresses,
    evaluate,
    open_kernel,
    pairs,
    read,
    save,
    splits,
    write_strips,
)

BUDGET = 23320


def select_rows(train, width, source, output, device, kernel):
    h, y = pairs(train)
    mirror, _, _, _, retention = read(source)
    scene = TriangleScene(device, mirror)
    at = addresses(device, kernel, scene, h, width)
    strips, kept = budgeted_strips(at, h, y)
    save(output, mirror, strips, width, GRID, retention)
    return kept, len(strips["tri_kind"])


def budgeted_strips(at, h, y):
    if np.any(at < 0):
        raise ValueError("Unexpected invalid training return")
    keys, inverse = np.unique(at, return_inverse=True)
    n = len(keys)
    cutoff = 9 * len(y) // 10
    # Allocate using a held-back suffix of training only. The counts used to
    # estimate the benefit of a row exclude those observations.
    full = np.zeros((n + 257, 256), np.int32)
    fit = np.zeros_like(full)
    for rows, end in ((fit, cutoff), (full, len(y))):
        np.add.at(rows, (inverse[:end], y[:end]), 1)
        np.add.at(rows, (n + h[:end, -1].astype(int), y[:end]), 1)
        rows[-1] = np.bincount(y[:end], minlength=256)
    row = inverse[cutoff:]
    target = y[cutoff:]
    byte = h[cutoff:, -1].astype(int)
    total = fit.sum(1)
    unigram = fit[-1, target] / total[-1]
    back = np.divide(
        fit[n + byte, target],
        total[n + byte],
        out=unigram.copy(),
        where=total[n + byte] > 0,
    )
    high = np.divide(
        fit[row, target], total[row], out=back.copy(), where=total[row] > 0
    )
    benefit = np.bincount(
        row,
        weights=np.log2(
            (0.99 * (0.8 * high + 0.2 * back) + ESCAPE) / (0.99 * back + ESCAPE)
        ),
        minlength=n,
    )
    triangles = 2 * np.count_nonzero(full[:n], axis=1)
    remaining = BUDGET - 2 * np.count_nonzero(full[n:])
    keep = []
    for i in np.argsort(-benefit / triangles, kind="stable"):
        if benefit[i] <= 0:
            break
        if triangles[i] <= remaining:
            keep.append(i)
            remaining -= triangles[i]
    keep = np.array(keep, dtype=int)
    ids = np.r_[keys[keep], BACKOFF + np.arange(257)]
    strips = write_strips(np.vstack((full[keep], full[n:])), ids)
    assert len(strips["tri_kind"]) <= BUDGET
    return strips, len(keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    train, val, _ = splits(args.book)
    h, y = pairs(val)
    device, kernel = open_kernel("cavity.slang")
    reader = device.create_compute_kernel(
        device.load_program("cavity_readout.slang", ["trace"])
    )
    for w in (1, 7):
        output = args.out / f"budget-{w}.npz"
        rows, triangles = select_rows(
            train, w, args.out / f"conditioned-{w}.npz", output, device, kernel
        )
        result = evaluate(device, kernel, reader, h, y, output)
        print(w, rows, triangles, result)


if __name__ == "__main__":
    main()
