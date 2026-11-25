"""E15: conditional continuation stations addressed by earlier-word beam hits."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import slangpy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/word_emission"))
from scene import Reader, WordRTX, WordScene, gpu_buffer, tokens

GRID = (0.0, 0.05, 0.1, 0.2, 0.35)
JEKYLL = "31df0a9af1bce24ccbb5ab3ccb659603353accde5b7e40f7c5b887fb88173d48"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def emissions(words):
    rows = defaultdict(Counter)
    for a, b, c in zip(words, words[1:], words[2:]):
        rows[a, b][c] += 1
    return rows


def write(path, rows, size, identity):
    station, mass = [], []
    for i, ((a, b), counts) in enumerate(sorted(rows.items())):
        n = counts.total()
        station.append((a, b, n / (n + 5)))
        mass.extend((c, size + i, k / n) for c, k in counts.items())
    centers = np.array(station + mass, np.float32)
    corners = np.array(
        [[-0.375, -0.25, 0], [0.375, -0.25, 0], [0, 0.375, 0]], np.float32
    )
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        tmp = Path(folder) / "pairs.npz"
        np.savez_compressed(tmp, triangles=centers[:, None] + corners, model=identity)
        os.link(tmp, path)


class PairScene:
    def __init__(self, model, path, identity):
        self.model = model
        self.kernel = model.device.create_compute_kernel(
            model.device.load_program("word_pairs.slang", ["trace"])
        )
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["model"]) != identity:
                raise ValueError("Pair geometry belongs to a different word archive")
            self.scene = WordRTX(
                model.device, self.kernel, WordScene(saved["triangles"])
            )

    def read(self, a, b, c, base):
        device = self.model.device
        out = gpu_buffer(
            device, np.empty((len(a), 4), np.float32), sp.BufferUsage.unordered_access
        )
        self.kernel.dispatch(
            thread_count=[len(a), 1, 1],
            vars={
                "queries": len(a),
                "vocabSize": len(self.model.words),
                "scene": self.scene.tlas.structure,
                "older": gpu_buffer(device, a.astype(np.uint32)),
                "recent": gpu_buffer(device, b.astype(np.uint32)),
                "target": gpu_buffer(device, c.astype(np.uint32)),
                "backoff": gpu_buffer(device, base.astype(np.float32)),
                "readout": out,
                **self.model.buffers,
            },
        )
        return out.to_numpy().view(np.float32).reshape(-1, 4).astype(float)


def cpu_check(model, rows, a, b, c, base, actual):
    # The common first-stage RTX hit list is fixed. Check both later ray stages
    # independently with original training counts, without loading their triangles.
    offsets = model.buffers["offsets"].to_numpy().view(np.uint32)
    ids = model.buffers["neighbours"].to_numpy().view(np.uint32)
    energy = model.buffers["energy"].to_numpy().view(np.float32)
    neighbourhood = {}

    def routes(x, y):
        if (x, y) not in neighbourhood:
            r = range(offsets[x], offsets[x + 1])
            neighbourhood[x, y] = [
                (rows[ids[j], y], float(energy[j])) for j in r if (ids[j], y) in rows
            ]
        return neighbourhood[x, y]

    for i, (x, y, z) in enumerate(zip(a, b, c)):
        entry = rows.get((x, y), {})
        n = sum(entry.values())
        exact = (5 * base[i] + entry.get(z, 0)) / (5 + n)
        donors = routes(x, y)
        total = sum(w for _, w in donors)
        local = (
            sum(w * counts[z] / counts.total() for counts, w in donors) / total
            if total
            else exact
        )
        if max(abs(actual[i] - np.array([exact, local, n / (n + 5), total]))) > 2e-6:
            raise AssertionError(f"RTX pair routing diverged at {i}")


def scores(p, seen):
    return {
        label: float(np.exp(-np.log(p[mask]).mean()))
        for label, mask in (
            ("all", np.ones(len(p), bool)),
            ("unseen_pair", ~seen),
            ("seen_pair", seen),
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test", "generate"))
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--books", type=Path)
    parser.add_argument("--prompt", default="Elizabeth was")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    model = Reader(args.model)
    pairpath = args.out / "pairs.npz"
    identity = digest(args.model)
    if args.phase != "generate":
        if args.books is None:
            parser.error("fit and test require --books")
        train, val, _ = tokens(args.books)
        ids = {w: i for i, w in enumerate(model.words)}
        rows = emissions([ids.get(t, 0) for t in train])
        if args.phase == "fit":
            write(pairpath, rows, len(ids), identity)
            seq = np.array([ids.get(t, 0) for t in val])
        else:
            raw = (args.books / "rtxllm-jekyll-43.txt").read_bytes()
            if hashlib.sha256(raw).hexdigest() != JEKYLL:
                raise ValueError("Jekyll digest mismatch")
            body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[
                1
            ]
            seq = model.encode(body)
    pair = PairScene(model, pairpath, identity)
    if args.phase != "fit":
        fit = json.loads((args.out / "fit.json").read_text(encoding="utf-8"))
        if fit["pair_sha256"] != digest(pairpath):
            raise ValueError("Calibration and pair geometry differ")
        weight = fit["selected"]
    if args.phase == "generate":
        seq = model.encode(args.prompt.encode())
        if len(seq) < 2 or args.count < 0:
            parser.error(
                "generation requires two prompt tokens and a nonnegative count"
            )
        x, y = seq[-2:]
        rng = np.random.default_rng(args.seed)
        dest = np.arange(len(model.words))
        output = []
        for _ in range(args.count):
            a, b = np.full(len(dest), x), np.full(len(dest), y)
            r = pair.read(a, b, dest, model.predict(b, dest)[:, 1])
            p = (1 - weight) * r[:, 0] + weight * r[:, 1]
            if abs(p.sum() - 1) > 2e-6 or min(p) < 0:
                raise AssertionError("invalid distribution")
            x, y = y, rng.choice(dest, p=p / p.sum())
            output.append(model.words[y].hex())
        print(json.dumps(output))
        return
    a, b, c = seq[:-2], seq[1:-1], seq[2:]
    base = model.predict(b, c)[:, 1].astype(float)
    r = pair.read(a, b, c, base)
    cpu_check(model, rows, a, b, c, base, r)
    seen = r[:, 2] > 0

    def mix(r, w):
        return (1 - w) * r[:, 0] + w * r[:, 1]

    report = {
        "model_sha256": identity,
        "pair_sha256": digest(pairpath),
        "targets": len(c),
        "unseen_pairs": int((~seen).sum()),
        "donor_coverage": float((r[:, 3] > 0).mean()),
        "pair_stations": len(rows),
        "pair_as_bytes": pair.scene.as_bytes,
        "one_word": scores(base, seen),
        "exact": scores(r[:, 0], seen),
    }
    # Diagnostic control: Witten-Bell backs off according to the number of
    # distinct successors. It uses the same E14 lower-order distribution.
    wb = base.copy()
    for i, (x, y, z) in enumerate(zip(a, b, c)):
        counts = rows.get((x, y))
        if counts:
            types = len(counts)
            wb[i] = (counts[z] + types * base[i]) / (counts.total() + types)
    report["witten_bell_pair"] = scores(wb, seen)
    if args.phase == "fit":
        losses = [scores(mix(r, w), seen)["all"] for w in GRID]
        weight = GRID[np.argmin(losses)]
        report["validation_grid"] = dict(zip(GRID, losses))
    report.update(selected=weight, donors=scores(mix(r, weight), seen))
    shuffled = np.random.default_rng(63).permutation(a)
    sr = pair.read(shuffled, b, c, base)
    cpu_check(model, rows, shuffled, b, c, base, sr)
    report["shuffled_older"] = scores(mix(sr, weight), seen)
    report["relative_gain"] = 1 - report["donors"]["all"] / report["exact"]["all"]
    report["meets_threshold"] = report["relative_gain"] >= 0.01
    s = json.dumps(report, indent=2)
    (args.out / f"{args.phase}.json").write_text(s + "\n", encoding="utf-8")
    print(s)


if __name__ == "__main__":
    main()
