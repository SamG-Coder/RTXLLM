"""E16: train the elevation of observed two-word stations using RTX scores."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/word_pairs"))
from run_pairs import JEKYLL, PairScene, Reader, digest, emissions, scores, tokens
import re

GRID = np.array([0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
WALLPAPER = "fdd24cf8a952177d32321079a373ddbf4f98663aec8a8a0009bdff0e1ebbb622"


def encoded_book(model, path, expected):
    if digest(path) != expected:
        raise ValueError("Book digest mismatch")
    body = re.search(
        rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", path.read_bytes()
    )[1]
    return model.encode(body)


def group(pair, n):
    return int(n >= 2) + int(n >= 5) + 3 * int(0 in pair)


def cpu_lookups(rows, a, b, c):
    groups = np.full(len(a), -1)
    mass = np.zeros(len(a))
    strength = np.zeros(len(a))
    for i, (x, y, z) in enumerate(zip(a, b, c)):
        row = rows.get((x, y))
        if row:
            n = row.total()
            groups[i] = group((x, y), n)
            mass[i] = row[z] / n
            strength[i] = n / (n + 5)
    return groups, mass, strength


def scene_file(path, original, groups, heights, identity, compressed=True):
    triangles = original.copy()
    triangles[: len(groups), :, 2] = heights[groups, None]
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        stage = Path(tmp) / "scene.npz"
        writer = np.savez_compressed if compressed else np.savez
        writer(stage, triangles=triangles, model=identity)
        os.link(stage, path)


def fit(model, source, folder, identity, rows, a, b, c, base, groups):
    with np.load(source, allow_pickle=False) as f:
        if str(f["model"]) != identity:
            raise ValueError("Pair archive mismatch")
        original = f["triangles"]
    pairs = sorted(rows)
    # The third vertex is at the column centre and the first lies 0.25 below the row.
    np.testing.assert_array_equal(original[: len(pairs), 2, 0], [p[0] for p in pairs])
    np.testing.assert_array_equal(
        original[: len(pairs), 0, 1] + 0.25, [p[1] for p in pairs]
    )
    station_groups = np.array([group(p, rows[p].total()) for p in pairs])
    losses = []
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        for i, w in enumerate(GRID):
            path = Path(tmp) / f"trial-{i}.npz"
            scene_file(path, original, station_groups, np.full(6, w), identity, False)
            prob = PairScene(model, path, identity).read(a, b, c, base)[:, 0]
            bits = -np.log2(prob)
            losses.append([bits.sum(), *(bits[groups == g].sum() for g in range(6))])
    best = GRID[np.argmin(losses, axis=0)]
    scene_file(
        folder / "global.npz", original, station_groups, np.full(6, best[0]), identity
    )
    scene_file(folder / "learned.npz", original, station_groups, best[1:], identity)
    return best


def generate(model, scene, prompt, count, seed):
    seq = model.encode(prompt.encode())
    if len(seq) < 2 or count < 0:
        raise ValueError("Two prompt tokens and a nonnegative count are required")
    rng = np.random.default_rng(seed)
    x, y = seq[-2:]
    dest = np.arange(len(model.words))
    output = []
    for _ in range(count):
        a, b = np.full(len(dest), x), np.full(len(dest), y)
        p = scene.read(a, b, dest, model.predict(b, dest)[:, 1])[:, 0]
        if min(p) < 0 or abs(p.sum() - 1) > 2e-6:
            raise AssertionError("Unnormalized pair distribution")
        x, y = y, rng.choice(dest, p=p / p.sum())
        output.append(model.words[y].hex())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test", "generate"))
    parser.add_argument("model", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--books", type=Path)
    parser.add_argument("--prompt", default="Elizabeth was")
    parser.add_argument("--count", default=32, type=int)
    parser.add_argument("--seed", default=65, type=int)
    args = parser.parse_args()
    start = perf_counter()
    identity = digest(args.model)
    model = Reader(args.model)
    args.folder.mkdir(exist_ok=True, parents=True)
    if args.phase == "generate":
        scene = PairScene(model, args.folder / "learned.npz", identity)
        print(json.dumps(generate(model, scene, args.prompt, args.count, args.seed)))
        return
    if args.source is None or args.books is None:
        parser.error("fit and test require --source and --books")
    tr, _, _ = tokens(args.books)
    ids = {w: i for i, w in enumerate(model.words)}
    rows = emissions([ids.get(t, 0) for t in tr])
    name, sha = (
        ("jekyll-43", JEKYLL) if args.phase == "fit" else ("wallpaper-1952", WALLPAPER)
    )
    seq = encoded_book(model, args.books / f"rtxllm-{name}.txt", sha)
    a, b, c = seq[:-2], seq[1:-1], seq[2:]
    base = model.predict(b, c)[:, 1].astype(float)
    groups, mass, strength = cpu_lookups(rows, a, b, c)
    seen = groups >= 0
    if args.phase == "fit":
        best = fit(
            model, args.source, args.folder, identity, rows, a, b, c, base, groups
        )
    else:
        fitted = json.loads((args.folder / "fit.json").read_text(encoding="utf-8"))
        best = np.array(fitted["heights"])
        for file, checksum in fitted["archive_sha256"].items():
            if digest(args.folder / file) != checksum:
                raise ValueError("Changed gate archive")
    report = {
        "book": name,
        "targets": len(a),
        "group_support": np.bincount(groups + 1, minlength=7).tolist(),
        "heights": best.tolist(),
        "one_word": scores(base, seen),
        "archive_sha256": {},
    }
    for label, path, w in (
        ("fixed", args.source, strength),
        ("global", args.folder / "global.npz", seen * best[0]),
        ("learned", args.folder / "learned.npz", np.where(seen, best[1:][groups], 0)),
    ):
        scene = PairScene(model, path, identity)
        actual = scene.read(a, b, c, base)
        np.testing.assert_allclose(actual[:, 2], w, atol=2e-6, rtol=0)
        np.testing.assert_allclose(
            actual[:, 0], (1 - w) * base + w * mass, atol=2e-6, rtol=0
        )
        report[label] = scores(actual[:, 0], seen)
        if label != "fixed":
            report["archive_sha256"][path.name] = digest(path)
    shuffled = np.random.default_rng(65).permutation(a)
    g, m, _ = cpu_lookups(rows, shuffled, b, c)
    w = np.where(g >= 0, best[1:][g], 0)
    p = scene.read(shuffled, b, c, base)[:, 0]
    np.testing.assert_allclose(p, (1 - w) * base + w * m, atol=2e-6, rtol=0)
    report["shuffled"] = scores(p, seen)
    report["relative_gain"] = 1 - report["learned"]["all"] / report["one_word"]["all"]
    report["passes"] = report["relative_gain"] >= 0.01
    report["wall_seconds"] = perf_counter() - start
    s = json.dumps(report, indent=2)
    (args.folder / f"{args.phase}.json").write_text(s + "\n", encoding="utf-8")
    print(s)


if __name__ == "__main__":
    main()
