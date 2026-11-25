"""Exercise train/update/score/generate as separate processes on an RTX adapter."""

import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

CLI = Path(__file__).resolve().with_name("byte_model.py")


def invoke(*args, expect_error=False):
    result = subprocess.run(
        [sys.executable, "-B", str(CLI), *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
    )
    if expect_error:
        if result.returncode == 0:
            raise AssertionError("Expected failure")
        return result.stderr
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_score(scene, files):
    import byte_model

    report = json.loads(invoke("score", scene, *files).splitlines()[-1])
    transitions = Counter()
    total_bytes = 0
    for file in files:
        raw = file.read_bytes()
        total_bytes += len(raw)
        transitions.update(itertools.pairwise(raw))
    assert report["input_bytes"] == total_bytes
    assert report["scored_bytes"] == transitions.total()
    assert report["files"] == len(files)
    learn = byte_model.learn
    device, kernel = learn.bigram.open_kernel()
    geometry = learn.bigram.SpecularScene(device, byte_model.read_geometry(scene))
    programs = np.column_stack(
        (np.full(learn.N, learn.N, np.int32), np.arange(learn.N, dtype=np.int32))
    )
    bits = 0.0
    for prev in sorted({a for a, _ in transitions}):
        paths, _ = geometry.trace(
            kernel,
            {
                "start": np.full(learn.N, 256 + prev, np.int32),
                "program": programs,
            },
        )
        p = learn.vote_distribution(paths[:, 1][None, :])[0]
        bits -= sum(n * np.log2(p[b]) for (a, b), n in transitions.items() if a == prev)
    np.testing.assert_allclose(
        report["bits_per_byte"], bits / transitions.total(), atol=1e-12, rtol=0
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Include a chunk boundary and non-UTF-8 bytes; boundary transitions
        # must appear exactly once whether the input arrives in one or two files.
        a = (b"lantern\x00\xff " * 120000) + b"!"
        b = b"?violet " * 50
        first, second = root / "a.raw", root / "b.raw"
        first.write_bytes(a)
        second.write_bytes(b)
        initial, continued, independent = (
            root / "first",
            root / "continued",
            root / "independent",
        )
        invoke("train", first, initial, "--seed", 7, "--rounds", 2)
        original = digest(initial / "scene.npz"), digest(initial / "training.npz")
        invoke("update", initial, second, continued, "--continuation", "--rounds", 2)
        invoke("update", initial, second, independent, "--rounds", 2)
        for directory, linked in ((continued, True), (independent, False)):
            with np.load(directory / "training.npz", allow_pickle=False) as state:
                data = np.frombuffer(a + b, np.uint8).astype(np.int32)
                expected = np.bincount(
                    data[:-1] * 256 + data[1:], minlength=65536
                ).reshape(256, 256)
                if not linked:
                    expected[a[-1], b[0]] -= 1
                np.testing.assert_array_equal(expected, state["counts"])
                np.testing.assert_array_equal(
                    np.bincount(data, minlength=256), state["frequency"]
                )
                meta = json.loads(str(state["metadata"]))
                assert meta["rounds"] == 4 and meta["total_bytes"] == len(a) + len(b)
                assert meta["scene_sha256"] == digest(directory / "scene.npz")
            with (
                np.load(initial / "scene.npz") as prior,
                np.load(directory / "scene.npz") as current,
            ):
                for key in ("station_pos", "tri_station", "tri_kind", "relation_dir"):
                    np.testing.assert_array_equal(prior[key], current[key])
        assert original == (
            digest(initial / "scene.npz"),
            digest(initial / "training.npz"),
        )
        # Inference is exercised with only the exported geometry on its path.
        export = root / "export"
        export.mkdir()
        scene = export / "only-geometry.npz"
        shutil.copyfile(continued / "scene.npz", scene)
        for name in ("sample1", "sample2"):
            invoke(
                "generate",
                scene,
                "Hi",
                "--seed",
                9,
                "--count",
                32,
                "--out",
                export / name,
            )
        raw = (export / "sample1").read_bytes()
        assert (
            raw.startswith(b"Hi")
            and len(raw) == 34
            and raw == (export / "sample2").read_bytes()
        )
        assert "FileExistsError" in invoke("train", second, initial, expect_error=True)
        empty = root / "empty"
        empty.write_bytes(b"")
        single = root / "single"
        single.write_bytes(b"!")
        verify_score(scene, [empty, first, single, second])
        assert "two or more bytes" in invoke(
            "score", scene, empty, single, expect_error=True
        )
        assert "at least one byte" in invoke(
            "train", empty, root / "failed", expect_error=True
        )
        assert not (root / "failed").exists()
        # An interrupted copy or mixed pair must fail before publishing a child.
        shutil.copyfile(initial / "training.npz", export / "training.npz")
        shutil.copyfile(continued / "scene.npz", export / "scene.npz")
        assert "same checkpoint" in invoke(
            "update", export, second, root / "failed", expect_error=True
        )
        assert not (root / "failed").exists()
        print(
            "Verified chunking, both update boundary modes, stable identities, geometry-only scoring and sampling and publication failures"
        )


if __name__ == "__main__":
    main()
