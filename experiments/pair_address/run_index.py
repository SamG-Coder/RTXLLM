"""E10 paired versus single-byte ray columns at matched memory and search budget."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/integrated"))
from model import Predictor, book
from online import TILT
from run import TriangleScene, dispatch, geometry

TREASURE = "0fee2fef1b2ffde22e2c962bd8bc2b75ea1ae9b02b3fe85ef08d44f4cc1bb2de"


def place(predictor, raw, paired):
    column = np.frombuffer(raw, np.uint8).astype(np.int32)
    if paired:
        column[1:] += 256 * column[:-1]
    centers = np.column_stack((column, np.arange(len(raw)), np.zeros(len(raw))))
    return TriangleScene(predictor.device, geometry(centers, TILT))


def search(model, kernel, scene, n, window, paired):
    return dispatch(
        model.device,
        kernel,
        scene,
        np.empty((n, 4), np.int32),
        "outcomes",
        targets=n,
        windowSize=window,
        pairAddress=int(paired),
    )


def oracle(raw, window, paired):
    width = 2 if paired else 1
    result = np.empty((len(raw) - 8, 3), np.int32)
    for t in range(8, len(raw)):
        lower = max(0, t - window)
        key = raw[t - width : t]
        cursor = t - 1
        first = pred = -1
        best = 0
        for _ in range(16):
            begin = raw.rfind(key, lower, cursor)
            if begin < 0:
                break
            end = begin + width - 1
            length = width
            while (
                length < 8
                and end - length >= lower
                and raw[end - length] == raw[t - 1 - length]
            ):
                length += 1
            if length > best:
                best = length
                pred = raw[end + 1]
                if first < 0:
                    first = pred
            if best == 8:
                break
            cursor = end
        result[t - 8] = [first, pred, best]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "test"))
    parser.add_argument("model", type=Path)
    parser.add_argument("books", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    name = "jekyll-43" if args.phase == "validate" else "treasure-120"
    raw = book(args.books, name, None if args.phase == "validate" else TREASURE)
    model = Predictor(args.model)
    if model.layered.k != 5:
        raise ValueError("Protocol requires the order-six long-term scene")
    kernel = model.kernel("episodic.slang", "indexedTrace")
    data = np.frombuffer(raw, np.uint8)
    idx = np.arange(8, len(data))
    dishes, _ = model.layered.trace(
        model.suffix_kernel, model.layered.points(data, idx)
    )
    blank = np.zeros((len(idx), 4), np.int32)
    base = model.decode(dishes, blank, data[8:])
    report = {
        "book": name,
        "targets": len(idx),
        "model_sha256": model.identity,
        "no_copy_bpb": float(-np.log2(base).mean()),
        "conditions": {},
    }
    for paired in (False, True):
        scene = place(model, raw, paired)
        for window in (256, 2048):
            start = perf_counter()
            hits = search(model, kernel, scene, len(idx), window, paired)
            wall = perf_counter() - start
            np.testing.assert_array_equal(hits[:, :3], oracle(raw, window, paired))
            if paired:
                for cut in (8, 257, 4096):
                    altered = raw[:cut] + bytes(b ^ 255 for b in raw[cut : cut + 512])
                    changed = search(
                        model,
                        kernel,
                        place(model, altered, True),
                        len(altered) - 8,
                        window,
                        True,
                    )
                    np.testing.assert_array_equal(changed[: cut - 7], hits[: cut - 7])
            p = model.decode(dishes, hits, data[8:])
            active = hits[:, 2] >= 2
            report["conditions"][f"{'pair' if paired else 'one'}_{window}"] = {
                "bpb": float(-np.log2(p).mean()),
                "rays": int(hits[:, 3].sum()),
                "coverage": float(active.mean()),
                "copy_accuracy": float((hits[active, 1] == data[8:][active]).mean()),
                "triangles": len(raw),
                "as_bytes": scene.metrics["as_bytes"],
                "search_wall_s": wall,
            }
    scores = report["conditions"]
    report["passes"] = all(
        scores[k]["bpb"] - scores["pair_2048"]["bpb"] >= 0.01
        for k in ("one_256", "one_2048")
    )
    print(json.dumps(report, indent=2))
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
