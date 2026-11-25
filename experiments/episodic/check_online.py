"""Check live queries, eviction and exact sampling resume on real RTX."""

import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from online import OnlineMemory
from run import open_kernel, reference


def main():
    dev, _ = open_kernel("episodic.slang")
    rng = np.random.default_rng(53)
    raw = (
        bytes(range(256))
        + b"abacabadabracadabra " * 18
        + rng.integers(0, 20, 320, dtype=np.uint8).tobytes()
        + b"\xff\x00" * 60
    )
    oracle = reference(raw + b"\x00")
    memory = OnlineMemory.train(dev, [b"abracadabra " * 53, bytes(range(256)) * 3], 53)
    size = memory.scene.metrics["as_bytes"]
    start = perf_counter()
    for t, byte in enumerate(raw, 1):
        memory.observe(byte)
        hit = memory.query()
        assert hit[3] == byte
        np.testing.assert_array_equal(hit[:3], oracle[t - 8] if t >= 8 else [-1, -1, 0])
    elapsed = perf_counter() - start
    assert size == memory.scene.metrics["as_bytes"]
    with tempfile.TemporaryDirectory() as folder:
        ck = Path(folder) / "live.npz"
        memory.save(ck)
        restored = OnlineMemory.load(dev, ck)
        start = perf_counter()
        sample = bytes(memory.sample() for _ in range(128))
        generate_time = perf_counter() - start
        assert sample == bytes(restored.sample() for _ in range(128))
        np.testing.assert_array_equal(memory.vertices, restored.vertices)
        print(
            json.dumps(
                {
                    "device": str(dev.info.adapter_name),
                    "checked_positions": len(raw),
                    "ring_wraps": len(raw) // 256,
                    "ring_triangles": len(memory.vertices),
                    "ring_as_bytes": size,
                    "checkpoint_bytes": ck.stat().st_size,
                    "observe_and_query_per_s": len(raw) / elapsed,
                    "generated_bytes_per_s": len(sample) / generate_time,
                    "resumed_identical_bytes": len(sample),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
