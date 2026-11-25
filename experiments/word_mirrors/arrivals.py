"""Fit an orientation update from measured RTX arrivals and compare frozen depths."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pilot import (
    D,
    FACT_DISTANCE,
    RAYS,
    WIDTH,
    Model,
    WordScene,
    beam_origins,
    evaluate,
    fit,
    mixture_grid,
    open_kernel,
    score,
    tokens,
    trace,
)

BLENDS = (0.0, 0.25, 0.5, 1.0)


def slopes(normals):
    dot = normals @ D
    normals = np.where(dot[:, None] < 0, -normals, normals)
    dot = abs(dot)[:, None]
    tangent = normals - dot * D
    return -tangent / np.maximum(
        np.maximum(dot, np.linalg.norm(tangent, axis=1, keepdims=True) / 4), 1e-15
    )


def update(paths, weights, normals, desired, shuffle):
    incoming = np.zeros_like(normals)
    first = paths[:, :, 0]
    second = paths[:, :, 1]
    for hop, station in enumerate((first, second)):
        i, j = np.nonzero(station >= 0)
        direction = np.broadcast_to(D, (len(i), 3))
        if hop:
            n = normals[first[i, j]]
            direction = D - 2 * (n @ D)[:, None] * n
            if shuffle:
                direction = np.random.default_rng(92).permutation(direction)
        np.add.at(incoming, station[i, j], weights[i, j, None] * direction)
    product = incoming[:, :, None] * desired[:, None, :]
    _, eigvec = np.linalg.eigh(product + product.transpose(0, 2, 1))
    fitted = slopes(eigvec[:, :, 0])
    return np.where(
        np.linalg.norm(incoming, axis=1, keepdims=True) > 1e-12, fitted, slopes(normals)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=("val", "test"))
    for arg in ("books", "archive", "validation"):
        p.add_argument(arg, type=Path)
    p.add_argument("--test-report", type=Path)
    a = p.parse_args()
    if (a.split == "test") != bool(a.test_report):
        p.error("Supply --test-report only for frozen test evaluation")
    frozen = (
        None if a.split == "val" else json.loads(a.validation.read_text())["selected"]
    )
    train, val, test = tokens(a.books)
    m = Model(train)
    ev = m.enc(val if frozen is None else test)
    prev, nxt = ev[:-1], ev[1:]
    with np.load(a.archive, allow_pickle=False) as archive:
        ids, pos, triangles = (archive[k] for k in ("ids", "positions", "dishes"))
        np.testing.assert_array_equal(m.vocab, archive["vocabulary"])
    partner, make_scene, _ = fit(pos, ids, m, triangles)
    base = make_scene(np.arange(len(ids)))
    normals = np.cross(base.e1[::6], base.e2[::6]).astype(float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    centres = pos[ids] + FACT_DISTANCE * D
    target = centres[partner] - centres
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    device, kernel = open_kernel("word_mirrors.slang")
    origins = beam_origins(pos, ids, np.zeros(len(ids)), D, WIDTH, RAYS, back=WIDTH)
    radial = np.broadcast_to(
        np.exp(-0.5 * WIDTH**2 * (np.arange(RAYS) + 0.5) / RAYS), (len(ids), RAYS)
    )
    initial, verified = trace(device, kernel, base, origins)
    report = dict(
        split=a.split,
        targets=len(prev),
        gpu=device.info.adapter_name,
        archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),
        optics={},
    )
    candidates = {}

    def settings(name, blend):
        if frozen is None:
            return None
        return {
            int(k): v["lambda"] for k, v in frozen[name].items() if v["blend"] == blend
        }

    def record(name, blend, curves):
        for depth, grid in enumerate(curves, 1):
            for lam, metrics in grid.items():
                candidates.setdefault(name, {}).setdefault(str(depth), []).append(
                    dict(blend=blend, **{"lambda": float(lam)}, **metrics)
                )

    old = slopes(normals)
    original = triangles.reshape(-1, 6, 3, 3).astype(float)
    for name in ("unchanged", "feedback", "shuffled"):
        new = update(
            initial, radial * m.rowtot[ids, None], normals, target, name == "shuffled"
        )
        for blend in (0.0,) if name == "unchanged" else BLENDS:
            selected = settings(name, blend)
            if selected == {}:
                continue
            if blend == 0:
                paths, checked = initial, verified
            else:
                slope = (1 - blend) * old + blend * new
                height = np.einsum(
                    "wfvc,wc->wfv", original - centres[:, None, None], slope
                )
                scene = WordScene((original + height[..., None] * D).reshape(-1, 3, 3))
                paths, checked = trace(device, kernel, scene, origins)
            report["optics"][f"{name}_{blend}"] = dict(
                hits=(paths >= 0).sum(axis=(0, 1)).tolist(), cpu_checked=checked
            )
            record(name, blend, evaluate(m, ids, paths, radial, prev, nxt, selected))

    graph = initial.copy()
    for depth in (1, 2):
        source = graph[:, :, depth - 1]
        graph[:, :, depth] = np.where(source >= 0, partner[source], -1)
    record(
        "graph", 0.0, evaluate(m, ids, graph, radial, prev, nxt, settings("graph", 0.0))
    )
    diff = centres[:, None] - centres[None]
    transition = np.exp(-0.5 * (diff**2).sum(2)) * m.rowtot[ids][None]
    transition /= transition.sum(1, keepdims=True)
    q = m.Pml[ids]
    pooled = np.zeros_like(q)
    has = np.ones(len(ids), bool)
    curves = []
    for depth in (1, 2, 3):
        q = transition @ q
        pooled += q
        if frozen is None:
            curves.append(mixture_grid(m, ids, pooled / depth, has, prev, nxt))
        else:
            lam = frozen["gaussian"][str(depth)]["lambda"]
            curves.append(
                {str(lam): score(m, ids, pooled / depth, has, lam, prev, nxt)}
            )
    record("gaussian", 0.0, curves)
    report["selected"] = {
        name: {d: min(options, key=lambda x: x["ppl"]) for d, options in depths.items()}
        for name, depths in candidates.items()
    }
    if frozen is None:
        report["grid"] = candidates
    output = a.validation if frozen is None else a.test_report
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
