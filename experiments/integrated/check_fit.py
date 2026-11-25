"""Exercise local fitting from binary files and inference in a clean process."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CLI = Path(__file__).with_name("model.py").resolve()


def invoke(*args, fail=False):
    result = subprocess.run(
        [sys.executable, "-B", str(CLI), *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
    )
    if fail:
        assert result.returncode and "nine or more bytes" in result.stderr
        return
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout.splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        training = root / "training.bin"
        training.write_bytes(bytes(range(256)) * 9 + b"earth and moon " * 70)
        a, b, short = (root / n for n in ("a", "b", "short"))
        a.write_bytes((b"moonlight\xff " * 39 + b"tab\x00" * 30) * 2)
        b.write_bytes(bytes(range(255, -1, -1)) * 3)
        short.write_bytes(b"12345678")
        model = root / "model.npz"
        invoke("fit", model, "--documents", training, "--calibration", short, fail=True)
        assert not model.exists()
        fitted = invoke(
            "fit",
            model,
            "--layout",
            "pair",
            "--documents",
            training,
            "--calibration",
            a,
            b,
            short,
        )
        assert (
            fitted["calibration_targets"]
            == len(a.read_bytes()) + len(b.read_bytes()) - 16
        )
        scores = [invoke("score", model, p) for p in (a, b)]
        reconstructed = (
            sum(s["targets"] * s["bits_per_byte"] for s in scores)
            / fitted["calibration_targets"]
        )
        np.testing.assert_allclose(
            fitted["calibration_bpb"], reconstructed, rtol=0, atol=1e-6
        )
        for p in (training, a, b, short):
            p.unlink()
        prompt = root / "prompt"
        prompt.write_bytes(b"\xff\x00")
        sample = invoke(
            "generate", model, root / "session.npz", "--prompt", prompt, "--count", 20
        )
        assert len(bytes.fromhex(sample["sample_hex"])) == 20
        with np.load(root / "session.npz") as state:
            assert str(state["layout"]) == "pair"
        print(
            json.dumps(
                {
                    "fitted": fitted,
                    "archive_only_calibration_bpb": reconstructed,
                    "generated_bytes": 20,
                }
            )
        )


if __name__ == "__main__":
    main()
