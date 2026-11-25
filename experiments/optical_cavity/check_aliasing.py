"""Validation-only test of whether a detector cell retains the latest byte."""

import argparse
from pathlib import Path

import numpy as np
from run import TriangleScene, cell_ids, corpus, land, open_kernel, pairs, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    train, val, _ = corpus(args.book)
    th, _ = pairs(train)
    vh, _ = pairs(val)
    device, kernel = open_kernel("cavity.slang")
    for w in (1, 8):
        mirror, _, width, grid, _ = read(args.out / f"width-{w}.npz")
        scene = TriangleScene(device, mirror)
        tr = cell_ids(land(device, kernel, scene, th[:, -width:]), grid)
        vr = cell_ids(land(device, kernel, scene, vh[:, -width:]), grid)
        hist = np.zeros((grid**2, 256), np.int32)
        np.add.at(hist, (tr[tr >= 0], th[tr >= 0, -1]), 1)
        # For every validation landing, predict its own final input byte
        # using only the majority byte among training landings in that cell.
        valid = (vr >= 0) & (hist.sum(1)[vr] > 0)
        pred = np.where(valid, hist.argmax(1)[vr], -1)
        print(
            width,
            "training cells",
            int((hist.sum(1) > 0).sum()),
            "validation final-byte recovery",
            float(np.mean(pred == vh[:, -1])),
        )


if __name__ == "__main__":
    main()
