"""Fit a budgeted spatial partition, then intersect its leaf rectangles on RTX."""

import heapq
from dataclasses import dataclass

import numpy as np
import slangpy as sp
from optics import TriangleScene, gpu_buffer, mesh, triangles_from_corners, write_strips


@dataclass
class Leaf:
    byte: int
    bounds: np.ndarray  # integer lattice, inclusive lower / exclusive upper
    samples: np.ndarray
    counts: np.ndarray
    root: bool


def entropy_cost(counts):
    positive = counts[counts > 0]
    return (
        positive.sum() * np.log2(positive.sum()) - positive @ np.log2(positive)
        if len(positive)
        else 0.0
    )


def propose(leaf, xy, y):
    if len(leaf.samples) < 64:
        return None
    best = None
    parent_cost = entropy_cost(leaf.counts)
    for axis in range(2):
        feature = xy[leaf.samples, axis]
        cut = int(np.median(feature))
        mask = feature < cut
        if min(mask.sum(), (~mask).sum()) < 32:
            continue
        a = np.bincount(y[leaf.samples[mask]], minlength=256)
        b = leaf.counts - a
        gain = parent_cost - entropy_cost(a) - entropy_cost(b)
        # Root emission rows are shared with the permanently resident backoff.
        parent_rows = 0 if leaf.root else np.count_nonzero(leaf.counts)
        cost = 2 + 2 * (np.count_nonzero(a) + np.count_nonzero(b) - parent_rows)
        priority = gain / cost
        if gain > 0 and (best is None or priority > best[0]):
            best = (priority, int(cost), axis, cut)
    return best


def tile(byte):
    return np.stack((byte % 16, byte // 16), axis=-1) * 16


def make_partition(xy, byte, y, budget):
    back = np.zeros((257, 256), np.int64)
    np.add.at(back, (byte, y), 1)
    back[256] = np.bincount(y, minlength=256)
    used = 2 * np.count_nonzero(back) + 512
    if used > budget:
        raise ValueError("Backoff distribution alone exceeds the budget")
    leaves = {}
    heap = []
    next_id = 0

    def add(leaf):
        nonlocal next_id
        idx = next_id
        next_id += 1
        leaves[idx] = leaf
        split = propose(leaf, xy, y)
        if split:
            p, c, axis, cut = split
            heapq.heappush(heap, (-p, idx, c, axis, cut))

    for b in range(256):
        add(
            Leaf(
                b,
                np.array([[0, 0], [4096, 4096]]),
                np.flatnonzero(byte == b),
                back[b],
                True,
            )
        )
    while heap:
        _, idx, extra, axis, cut = heapq.heappop(heap)
        if used + extra > budget:
            continue
        used += extra
        leaf = leaves.pop(idx)
        left = xy[leaf.samples, axis] < cut
        for side, mask in enumerate((left, ~left)):
            sample = leaf.samples[mask]
            bounds = leaf.bounds.copy()
            bounds[1 - side, axis] = cut
            add(
                Leaf(
                    leaf.byte,
                    bounds,
                    sample,
                    np.bincount(y[sample], minlength=256),
                    False,
                )
            )
    distributions = list(back)
    corners, labels = [], []
    for leaf in leaves.values():
        row = leaf.byte
        if not leaf.root:
            row = len(distributions)
            distributions.append(leaf.counts)
        lo, hi = leaf.bounds / 256 + tile(np.asarray(leaf.byte))
        corners.append(
            [[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]]
        )
        labels.extend([row, row])
    verts = triangles_from_corners(np.asarray(corners))
    part = mesh(verts, np.zeros(len(verts)), labels, np.array([[0.0, 0.0, -1.0]]))
    part["station_pos"] = np.zeros((len(distributions), 3), np.float32)
    strips = write_strips(np.asarray(distributions))
    if len(verts) + len(strips["tri_kind"]) != used:
        raise AssertionError("Physical allocation differs from the split budget")
    return part, strips


def lattice(positions):
    if not (np.abs(positions) < 8).all():
        raise ValueError("Invalid cavity landing")
    return np.floor((positions + 8) * 256).astype(np.int32)


def detect(device, kernel, part, xy, byte):
    points = ((xy + 0.5) / 256 + tile(byte.astype(np.int32))).astype(np.float32)
    scene = TriangleScene(device, part)
    out = gpu_buffer(
        device, np.empty(len(byte), np.int32), sp.BufferUsage.unordered_access
    )
    kernel.dispatch(
        thread_count=[len(byte), 1, 1],
        vars={
            "scene": scene.tlas.structure,
            "owners": scene.bindings["owners"],
            "xy": gpu_buffer(device, points),
            "count": len(byte),
            "rows": out,
        },
    )
    rows = out.to_numpy().view(np.int32).copy()
    # Independently check rectangle membership. Pixel centres stay strictly
    # inside or outside integer-lattice split boundaries.
    cpu = np.full(len(byte), -1, np.int32)
    groups = [np.flatnonzero(byte == b) for b in range(256)]
    for vertices, owner in zip(
        part["tri_vertices"].reshape(-1, 6, 3), part["tri_station"][::2]
    ):
        lo, hi = vertices[:, :2].min(0), vertices[:, :2].max(0)
        bx, by = (lo // 16).astype(int)
        ii = groups[16 * by + bx]
        p = points[ii]
        inside = (
            (p[:, 0] >= lo[0])
            & (p[:, 0] < hi[0])
            & (p[:, 1] >= lo[1])
            & (p[:, 1] < hi[1])
        )
        cpu[ii[inside]] = owner
    np.testing.assert_array_equal(rows, cpu)
    if np.any(rows < 0):
        raise ValueError("Uncovered detector point")
    return rows
