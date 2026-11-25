"""Exercise two physical reflections, padding, misses and workgroup boundaries."""

import numpy as np
from run_specular import open_kernel, trace


def main():
    # First mirror reflects +Z into +X. The translated pair repeats that
    # interaction after the intermediate station emits the second relation.
    pair = np.array(
        [
            [[-1, -1, -1], [1, -1, 1], [0, 1, 0]],
            [[2, -1, -1], [2, 1, -1], [2, 0, 1]],
        ],
        np.float32,
    )
    scene = {
        "tri_vertices": np.concatenate([pair, pair + np.array([2, 0, 2], np.float32)]),
        "tri_kind": np.array([1, 0, 1, 0], np.uint32),
        "tri_station": np.array([-1, 1, -1, 2], np.int32),
        "tri_reflectance": np.ones(4, np.float32),
        "station_pos": np.array([[0, 0, -2], [2, 0, 0], [4, 0, 2]], np.float32),
        "relation_dir": np.array([[0, 0, 1], [0, -1, 0]], np.float32),
        "emit_offset": np.float32(0.02),
        "t_min": np.float32(0.001),
    }
    queries = {
        "start": np.tile(np.array([0, 0, 1, 0], np.int32), 33),
        "program": np.tile(
            np.array([[0, 0], [1, -1], [0, -1], [0, 1]], np.int32), (33, 1)
        ),
    }
    device, kernel = open_kernel()
    receivers, metrics = trace(device, kernel, scene, queries)
    expected = np.tile([[1, 2], [-1, -1], [2, -1], [1, -1]], (33, 1))
    np.testing.assert_array_equal(receivers, expected)
    print(f"Matched all {len(expected)} fixture rows", metrics)


if __name__ == "__main__":
    main()
