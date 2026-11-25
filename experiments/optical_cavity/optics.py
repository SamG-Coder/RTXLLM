"""Shared optics and a persistent, ray-readable conditional distribution."""

import sys
from pathlib import Path

import numpy as np
import slangpy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "rtx"))
from run_specular import TriangleScene, gpu_buffer, open_kernel

__all__ = [
    "TriangleScene",
    "cell_ids",
    "check_optics",
    "land",
    "open_kernel",
    "read_strips",
    "reflector",
    "write_strips",
]

EXTENT = 8.0


def triangles_from_corners(corners):
    return corners[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3, 3).astype(np.float32)


def mesh(vertices, kinds, owners, directions):
    return {
        "tri_vertices": vertices,
        "tri_kind": np.asarray(kinds, np.uint32),
        "tri_station": np.asarray(owners, np.int32),
        "tri_reflectance": np.ones(len(vertices), np.float32),
        "station_pos": np.zeros((256, 3), np.float32),
        "relation_dir": directions.astype(np.float32),
        "emit_offset": np.float32(0),
        "t_min": np.float32(0.0001),
    }


def reflector(retention):
    axis = np.linspace(-EXTENT, EXTENT, 129)
    x, y = np.meshgrid(axis, axis)
    z = 10 - (1 - retention) * (x * x + y * y) / 40
    pts = np.stack([x, y, z], axis=-1)
    tiles = np.stack(
        [pts[:-1, :-1], pts[:-1, 1:], pts[1:, 1:], pts[1:, :-1]], axis=2
    ).reshape(-1, 4, 3)
    mirror = triangles_from_corners(tiles)
    plane = triangles_from_corners(
        np.array([[[-8, -8, 0], [8, -8, 0], [8, 8, 0], [-8, 8, 0]]])
    )
    rng = np.random.default_rng(20261125)
    slopes = rng.uniform(-0.07, 0.07, (256, 2))
    directions = np.column_stack((slopes, np.ones(256)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return mesh(
        np.concatenate((plane, mirror)),
        np.repeat([0, 1], [2, len(mirror)]),
        np.repeat([0, -1], [2, len(mirror)]),
        directions,
    )


def land(device, kernel, scene, history):
    n, width = history.shape
    buffer = gpu_buffer(
        device, np.empty((n, 2), np.float32), sp.BufferUsage.unordered_access
    )
    kernel.dispatch(
        thread_count=[n, 1, 1],
        vars={
            "scene": scene.tlas.structure,
            "vertices": scene.vertices,
            "directions": scene.bindings["directions"],
            "history": gpu_buffer(device, history.astype(np.uint32)),
            "queryCount": n,
            "width": width,
            "landings": buffer,
        },
    )
    return buffer.to_numpy().view(np.float32).reshape(n, 2).copy()


def cell_ids(points, grid):
    ok = (np.abs(points) < EXTENT).all(1)
    xy = np.floor((points + EXTENT) * (grid / (2 * EXTENT))).astype(np.int64)
    return np.where(ok, xy[:, 0] + grid * xy[:, 1], -1)


def write_strips(counts):
    # Nonempty byte outcomes partition each row's unit-width detector. Only
    # triangle extents and stable output-byte IDs carry the learned masses.
    rr, token = np.nonzero(counts)
    cdf = counts.cumsum(1)
    total = cdf[rr, -1]
    left = rr % 128 + (cdf[rr, token] - counts[rr, token]) / total
    right = rr % 128 + cdf[rr, token] / total
    y = rr // 128
    z = np.zeros(len(y))
    corners = np.stack(
        [
            np.column_stack((left, y, z)),
            np.column_stack((right, y, z)),
            np.column_stack((right, y + 1, z)),
            np.column_stack((left, y + 1, z)),
        ],
        axis=1,
    )
    v = triangles_from_corners(corners)
    return mesh(
        v,
        np.zeros(len(v), np.uint32),
        np.repeat(token, 2),
        np.array([[0, 0, -1]], np.float32),
    )


def read_strips(device, kernel, scene_data, row_count):
    scene = TriangleScene(device, scene_data)
    output = gpu_buffer(
        device, np.empty((row_count, 256), np.int32), sp.BufferUsage.unordered_access
    )
    kernel.dispatch(
        thread_count=[row_count * 256, 1, 1],
        vars={
            "scene": scene.tlas.structure,
            "owners": scene.bindings["owners"],
            "votes": output,
            "rowCount": row_count,
        },
    )
    hits = output.to_numpy().view(np.int32).reshape(row_count, 256)
    populated = (hits >= 0).sum(1)
    if np.any((populated != 0) & (populated != 256)):
        raise RuntimeError("Partly uncovered readout strip")
    counts = np.zeros((row_count, 256), np.int32)
    row, slot = np.nonzero(hits >= 0)
    np.add.at(counts, (row, hits[row, slot]), 1)
    return counts


def check_optics(device, kernel):
    data = reflector(1.0)
    scene = TriangleScene(device, data)
    history = np.array([[2, 17, 5], [5, 17, 2], [7, 7, 7], [0, 0, 0]], np.uint32)
    d = data["relation_dir"][history]
    np.testing.assert_allclose(
        land(device, kernel, scene, history),
        20 * (d[:, :, :2] / d[:, :, 2:]).sum(1),
        atol=2e-5,
    )
    curved = TriangleScene(device, reflector(0.5))
    returns = land(device, kernel, curved, history)
    if np.linalg.norm(returns[0] - returns[1]) < 0.01:
        raise AssertionError("Curved mirror failed the order intervention")
    print(
        "Flat-mirror analytic check and curved-mirror order intervention passed",
        flush=True,
    )
