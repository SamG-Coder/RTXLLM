"""Fit word mirrors on Pride training counts and evaluate paths on validation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import slangpy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e12_word_layout"))
from run import D, FACT_DISTANCE, RAYS, Model, score, tokens
from word_beam import (
    WordRTX,
    WordScene,
    beam_origins,
    gpu_buffer,
    mt_closest,
    open_kernel,
)

WIDTH = 2.5
FIRST_RANGE = 2 * WIDTH + FACT_DISTANCE + 0.5


def fit(pos, ids, model, triangles):
    features = np.sqrt(model.Pml[ids])
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    partner = np.argmax(similarity, axis=1)
    centres = pos[ids] + FACT_DISTANCE * D
    aim = centres[partner] - centres
    aim /= np.linalg.norm(aim, axis=1, keepdims=True)
    normal = D - aim
    # z = slope . xy in the plane perpendicular to the incident direction.
    dot = normal @ D
    slope = -(normal - dot[:, None] * D) / np.maximum(dot[:, None], 1e-12)
    magnitude = np.linalg.norm(slope, axis=1)
    slope /= np.maximum(1, magnitude / 4)[:, None]

    def scene(order):
        tri = triangles.reshape(-1, 6, 3, 3).astype(np.float64)
        height = np.einsum("wfvc,wc->wfv", tri - centres[:, None, None], slope[order])
        return WordScene((tri + height[..., None] * D).reshape(-1, 3, 3))

    return partner, scene, int((magnitude > 4).sum())


def cpu(scene, origins):
    dirs = np.broadcast_to(D, origins.shape).astype(np.float32).copy()
    o = origins.copy()
    paths = np.full((len(o), 3), -1, np.int32)
    live = np.arange(len(o))
    for depth, limit in enumerate((FIRST_RANGE, 100, 100)):
        t, tri = mt_closest(
            scene.v0,
            scene.e1,
            scene.e2,
            o[live],
            dirs[live],
            np.full(len(live), limit, np.float32),
            0.001,
            chunk=16,
        )
        ok = tri >= 0
        live, tri = live[ok], tri[ok]
        paths[live, depth] = tri // 6
        n = np.cross(scene.e1[tri], scene.e2[tri])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        o[live] += t[ok, None] * dirs[live]
        dirs[live] -= 2 * (dirs[live] * n).sum(axis=1, keepdims=True) * n
    return paths


def trace(device, kernel, scene, origins):
    gpu = WordRTX(device, kernel, scene)
    output = gpu_buffer(
        device,
        np.full((len(origins), 3), -1, np.int32),
        sp.BufferUsage.unordered_access,
    )
    kernel.dispatch(
        thread_count=[len(origins), 1, 1],
        vars=dict(
            scene=gpu.tlas.structure,
            origins=gpu_buffer(device, origins),
            vertices=gpu.vbuf,
            paths=output,
            count=len(origins),
            maxHits=3,
            incident=sp.float3(*D),
            firstRange=FIRST_RANGE,
        ),
    )
    paths = output.to_numpy().view(np.int32).reshape(-1, 3)
    # Fixed random samples from each survival stratum exercise second reflections
    # even when most incident rays miss.
    rng = np.random.default_rng(85)
    samples = []
    for live in range(4):
        indices = np.flatnonzero((paths >= 0).sum(axis=1) == live)
        samples.extend(rng.choice(indices, min(128, len(indices)), replace=False))
    np.testing.assert_array_equal(cpu(scene, origins[samples]), paths[samples])
    return paths.reshape(-1, RAYS, 3), len(samples)


def mixture_grid(model, ids, q, has, prev, nxt):
    return {
        str(lam): score(model, ids, q, has, lam, prev, nxt)
        for lam in (0.05, 0.1, 0.2, 0.3)
    }


def evaluate(model, ids, paths, radial, prev, nxt):
    rows = np.broadcast_to(np.arange(len(ids))[:, None], radial.shape)
    scores = []
    for depth in range(1, 4):
        observed = paths[:, :, :depth]
        energy = radial / np.maximum(1, (observed >= 0).sum(2))
        weight = np.zeros((len(ids), len(ids)))
        for hit in observed.transpose(2, 0, 1):
            ok = hit >= 0
            np.add.at(weight, (rows[ok], hit[ok]), energy[ok])
        total = weight.sum(1)
        weight /= np.where(total > 0, total, 1)[:, None]
        q = weight @ model.Pml[ids]
        scores.append(mixture_grid(model, ids, q, total > 0, prev, nxt))
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("books", "archive", "report"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    train, val, _ = tokens(args.books)
    model = Model(train)
    ev = model.enc(val)
    with np.load(args.archive, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["vocabulary"], model.vocab)
        ids, pos, radii, triangles = [
            archive[k] for k in ("ids", "positions", "radii", "dishes")
        ]
    partner, scene, clipped = fit(pos, ids, model, triangles)
    device, kernel = open_kernel("word_mirrors.slang")
    report = dict(
        targets=len(ev) - 1,
        rays=len(ids) * RAYS,
        clipped_normals=clipped,
        gpu=device.info.adapter_name,
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
    )
    for label, launch_radii in (
        ("annular", radii),
        ("full_disc", np.zeros_like(radii)),
    ):
        report[label] = beam_conditions(
            model, ev, ids, pos, launch_radii, device, kernel, scene, partner
        )
    coords = pos[ids]
    d2 = np.sum((coords[:, None] - coords[None]) ** 2, axis=2)
    transition = np.exp(-0.5 * d2) * model.rowtot[ids][None, :]
    transition /= transition.sum(axis=1, keepdims=True)
    q = model.Pml[ids]
    pooled = np.zeros_like(q)
    report["gaussian"] = []
    has = np.ones(len(ids), dtype=bool)
    for step in range(1, 4):
        q = transition @ q
        pooled += q
        report["gaussian"].append(
            {
                "endpoint": mixture_grid(model, ids, q, has, ev[:-1], ev[1:]),
                "pooled": mixture_grid(model, ids, pooled / step, has, ev[:-1], ev[1:]),
            }
        )
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")


def beam_conditions(model, ev, ids, pos, radii, device, kernel, scene, partner):
    report = {}
    origins = beam_origins(pos, ids, radii, D, WIDTH, RAYS, back=WIDTH)
    inner2 = (1.05 * radii[:, None]) ** 2
    outer2 = np.maximum(WIDTH, 1.1 * radii[:, None]) ** 2
    radial = np.exp(
        -0.5 * (inner2 + (outer2 - inner2) * (np.arange(RAYS) + 0.5) / RAYS)
    )
    for name, order in (
        ("trained", np.arange(len(ids))),
        ("shuffled", np.random.default_rng(0).permutation(len(ids))),
    ):
        paths, checked = trace(device, kernel, scene(order), origins)
        hits = paths >= 0
        entry = dict(
            cpu_checked_rays=checked,
            hits_by_depth=hits.sum(axis=(0, 1)).tolist(),
            distinct_stations=[
                int(np.unique(layer[layer >= 0]).size)
                for layer in paths.transpose(2, 0, 1)
            ],
            returned_to_first=int(
                ((paths[:, :, 2] == paths[:, :, 0]) & hits[:, :, 2]).sum()
            ),
            intended_partner_second=int(
                ((paths[:, :, 1] == partner[paths[:, :, 0]]) & hits[:, :, 1]).sum()
            ),
            scores=evaluate(model, ids, paths, radial, ev[:-1], ev[1:]),
        )
        if name == "trained":
            graph = paths.copy()
            for depth in (1, 2):
                source = graph[:, :, depth - 1]
                graph[:, :, depth] = np.where(source >= 0, partner[source], -1)
            entry["graph_scores"] = evaluate(model, ids, graph, radial, ev[:-1], ev[1:])
            entry["graph_matched_survival"] = evaluate(
                model, ids, np.where(hits, graph, -1), radial, ev[:-1], ev[1:]
            )
        report[name] = entry
    return report


if __name__ == "__main__":
    main()
