"""Competitive episode mirrors on a pinned Jekyll validation prefix (#82)."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import slangpy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e12_word_layout"))
from run import D
from word_beam import (
    WordRTX,
    WordScene,
    annulus,
    basis,
    gpu_buffer,
    mt_closest,
    open_kernel,
)

WINDOW, RAYS = 64, 32
PERM = np.random.default_rng(0).permutation(WINDOW)
TWIST = np.random.default_rng(1).permutation(WINDOW)
OFFSETS = annulus(RAYS, 0, 1)


def quads(center, u, v):
    corners = np.stack(
        [center - u - v, center + u - v, center + u + v, center - u + v], axis=1
    )
    return WordScene(corners[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3, 3))


def scenes(xy, start, twist):
    row = np.arange(len(xy))
    slot = (row + start) % WINDOW
    z = 8.0 * row
    band = z + 8 * (PERM[slot] - slot)
    offset = PERM - np.arange(WINDOW)
    aim = 8 * offset[TWIST[slot] if twist else slot]
    # For incident -Z the tangent to the mirror bisects +X travel and the
    # incoming direction. The receiver lies at x=2048, z=z_source+aim.
    angle = np.arctan2(aim, 2048 - xy[:, 0])
    slope = np.tan((angle - np.pi / 2) / 2)
    mirror = quads(
        np.column_stack((xy, z)),
        np.column_stack((np.ones(len(z)), np.zeros(len(z)), slope)),
        np.array([0, 1, 0]),
    )
    receiver = quads(
        np.column_stack((np.full(len(z), 2048), np.zeros(len(z)), band)),
        np.array([0, 128, 0]),
        np.array([0, 0, 3]),
    )
    return mirror, receiver


def cpu_trace(mirror, receiver, o):
    d = np.zeros_like(o)
    d[:, 2] = -1

    def closest(scene, origins, directions, far):
        return mt_closest(
            scene.v0,
            scene.e1,
            scene.e2,
            origins,
            directions,
            np.full(len(origins), far, np.float32),
            0.001,
        )

    t, tri = closest(mirror, o, d, 512)
    ok = tri >= 0
    result = np.full((len(o), 2), -1, np.int32)
    result[ok, 0] = tri[ok] // 2
    n = np.cross(mirror.e1[tri[ok]], mirror.e2[tri[ok]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    reflected = d[ok] - 2 * n * (d[ok] * n).sum(1, keepdims=True)
    _, hit = closest(receiver, o[ok] + t[ok, None] * d[ok], reflected, 4096)
    result[ok, 1] = hit // 2
    return result


def transport(device, kernel, xy, eligible, twist=False):
    result = np.full((len(xy), RAYS, 2), -1, np.int32)
    for block in range(0, len(xy), WINDOW):
        start, end = max(0, block - WINDOW), min(len(xy), block + WINDOW)
        rows = np.arange(block, end)[eligible[block:end]]
        if not len(rows):
            continue
        mirror, receiver = scenes(xy[start:end], start, twist)
        o = np.empty((len(rows), RAYS, 3), np.float32)
        o[:, :, :2] = xy[rows, None] + OFFSETS
        o[:, :, 2] = 8 * (rows[:, None] - start) - 4
        o = o.reshape(-1, 3)
        first, second = [WordRTX(device, kernel, s) for s in (mirror, receiver)]
        buf = gpu_buffer(
            device, np.zeros((len(o), 2), np.int32), sp.BufferUsage.unordered_access
        )
        kernel.dispatch(
            thread_count=[len(o), 1, 1],
            vars=dict(
                mirrors=first.tlas.structure,
                receivers=second.tlas.structure,
                vertices=first.vbuf,
                origins=gpu_buffer(device, o),
                hits=buf,
                rayCount=len(o),
            ),
        )
        gpu = buf.to_numpy().view(np.int32).reshape(-1, 2)
        np.testing.assert_array_equal(gpu, cpu_trace(mirror, receiver, o))
        absolute = np.where(gpu >= 0, gpu + start, -1).reshape(len(rows), RAYS, 2)
        source = absolute[:, :, 0]
        assert np.all(
            (source == -1)
            | ((source < rows[:, None]) & (source >= rows[:, None] - WINDOW))
        )
        # A misdirected ray may reach an unwritten or expired receiver. Its
        # payload is unavailable at prediction time.
        target = absolute[:, :, 1]
        target[(target >= rows[:, None]) | (target < rows[:, None] - WINDOW)] = -1
        result[rows] = absolute
    return result


def vote(words, rows, weights):
    totals, latest = Counter(), {}
    for row, w in zip(rows, weights):
        if row >= 0:
            token = words[row]
            totals[token] += w
            latest[token] = max(latest.get(token, -1), row)
    return max(totals, key=lambda token: (totals[token], latest[token]), default=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("jekyll", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    raw = args.jekyll.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "31df0a9af1bce24ccbb5ab3ccb659603353accde5b7e40f7c5b887fb88173d48"
    ):
        raise ValueError("Pinned Jekyll edition required")
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1]
    tokens = re.findall(rb"[a-z0-9]+|[^\sa-z0-9]", body.lower())
    with np.load(args.archive, allow_pickle=False) as archive:
        vocab = {word: i + 1 for i, word in enumerate(archive["vocabulary"].tolist())}
        ids, pos = archive["ids"], archive["positions"] @ np.stack(basis(D)).T
    events = [
        (vocab.get(a, 0), b) for a, b in zip(tokens, tokens[1:]) if b not in vocab
    ][:2048]
    context, words = zip(*events)
    context = np.array(context)
    eligible = np.isin(context, ids)
    xy = pos[context].copy()
    xy[~eligible] = [-1000, -1000]
    device, kernel = open_kernel("episode_mirror.slang")
    report = dict(
        queries=len(events),
        with_station=int(eligible.sum()),
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        gpu=device.info.adapter_name,
    )
    positive = {}
    for mode in ("learned", "shuffled_layout", "shuffled_tilt"):
        coords = xy.copy()
        if mode == "shuffled_layout":
            shuffled = pos.copy()
            shuffled[ids] = pos[np.random.default_rng(0).permutation(ids)]
            coords[eligible] = shuffled[context[eligible]]
        hits = transport(device, kernel, coords, eligible, mode == "shuffled_tilt")
        prefix = transport(
            device, kernel, coords[:93], eligible[:93], mode == "shuffled_tilt"
        )
        np.testing.assert_array_equal(hits[:93], prefix)
        received = hits[:, :, 1]
        if mode != "shuffled_tilt":
            np.testing.assert_array_equal(received, hits[:, :, 0])
        predictions = [vote(words, row, np.ones(RAYS, int)) for row in received]
        positive[mode] = np.array([a == b for a, b in zip(predictions, words)])
        report[mode] = dict(
            correct=int(positive[mode].sum()),
            first_hits=int((hits[:, :, 0] >= 0).sum()),
            eligible_receiver_hits=int((received >= 0).sum()),
            queries_with_receiver=int((received >= 0).any(axis=1).sum()),
        )
    for mode in ("nearest", "exact_weighted"):
        results = []
        for i in range(len(words)):
            rows = np.arange(max(0, i - WINDOW), i)[::-1]
            rows = rows[eligible[rows]] if eligible[i] else rows[:0]
            if mode == "nearest" and len(rows):
                rows = rows[[np.argmin(np.linalg.norm(xy[rows] - xy[i], axis=1))]]
            weights = np.where(context[rows] == context[i], 8, 1)
            results.append(vote(words, rows, weights) == words[i])
        positive[mode] = np.array(results)
        report[mode] = dict(correct=int(sum(results)))
    for mode, success in positive.items():
        report[mode]["accuracy"] = float(success.mean())
        report[mode]["learned_only_correct"] = int(
            (positive["learned"] & ~success).sum()
        )
        report[mode]["comparator_only_correct"] = int(
            (success & ~positive["learned"]).sum()
        )
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
