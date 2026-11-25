"""Exercise train/update/generate as separate processes on an RTX adapter."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
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
            "Verified chunking, both update boundary modes, stable identities, geometry-only sampling and publication failures"
        )


if __name__ == "__main__":
    main()
