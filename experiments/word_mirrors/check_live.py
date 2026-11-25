"""Verify whole-vocabulary probabilities and fresh-process mirror generation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from live import LiveReader
from pilot import D, RAYS, WIDTH, Model, WordScene, beam_origins, cpu, tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("books", "archive", "report"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    train, _, _ = tokens(a.books)
    model = Model(train)
    v = np.arange(model.V)
    with np.load(a.archive, allow_pickle=False) as archive:
        ids, pos, radii = (archive[k] for k in ("ids", "positions", "radii"))
        triangles = WordScene(archive["mirrors"])
    missing = np.setdiff1d(v[1:], ids)
    words = [
        0,
        missing[0],
        *ids[np.linspace(0, len(ids) - 1, 8, dtype=int)],
        ids[np.argmax(radii)],
    ]
    readers = {n: LiveReader(a.archive, n) for n in (1, 3)}
    changed = 0
    err = {1: 0.0, 3: 0.0}
    for w in np.unique(words):
        prev = np.full(model.V, w)
        prior = model.kn(prev, v)
        loc = np.flatnonzero(ids == w)
        if len(loc):
            radius = radii[loc[0]]
            origins = beam_origins(pos, [w], [radius], D, WIDTH, RAYS, back=WIDTH)
            paths = cpu(triangles, origins)
            delta = origins.astype(float) - (pos[w] - WIDTH * D)
            energy = np.exp(-0.5 * (delta**2).sum(1))
        preds = []
        for n, reader in readers.items():
            expected = prior
            if len(loc):
                masses = np.zeros(model.V)
                for ray, weight in zip(paths[:, :n], energy):
                    observed = ray[ray >= 0]
                    if len(observed):
                        masses += weight * model.Pml[ids[observed]].mean(axis=0)
                if masses.sum():
                    expected = 0.9 * prior + 0.1 * masses / masses.sum()
            pred = reader.predict(prev, v)[:, 1]
            np.testing.assert_allclose(pred, expected, rtol=0, atol=2e-6)
            np.testing.assert_allclose(pred.sum(dtype=float), 1, rtol=0, atol=2e-6)
            err[n] = max(err[n], float(np.max(abs(pred - expected))))
            preds.append(pred)
        changed += int(not np.array_equal(*preds))
    assert changed > 0
    reader = readers[3]
    start = perf_counter()
    seq = reader.generate(b"Elizabeth", 128, 3)
    elapsed = perf_counter() - start
    command = [
        sys.executable,
        str(Path(__file__).with_name("live.py")),
        "generate",
        str(a.archive),
        "Elizabeth",
        "--count",
        "128",
        "--seed",
        "3",
    ]

    def fresh():
        return json.loads(subprocess.check_output(command, text=True).splitlines()[-1])

    result = fresh()
    assert result == fresh() and seq == result["tokens_hex"]
    assert result["hits"][2] > 0
    output = dict(
        gpu=reader.device.info.adapter_name,
        archive_bytes=a.archive.stat().st_size,
        full_distributions_per_depth=len(np.unique(words)),
        max_abs_errors=err,
        distributions_changed=changed,
        warm_128_seconds=elapsed,
        **result,
    )
    a.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
