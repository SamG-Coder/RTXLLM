"""Word-session publication and cross-process restart checks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from session import Reader, Session, Speller, checksum, restore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("spell", type=Path)
    args = p.parse_args()
    model, spell = args.model.resolve(), args.spell.resolve()
    script = Path(__file__).resolve().with_name("session.py")
    mid, sid = checksum(model), checksum(spell)
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)

        def run(name, *argv):
            data = subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    str(script),
                    str(model),
                    str(spell),
                    name,
                    *argv,
                ],
                cwd=cwd,
            )
            return json.loads(data.splitlines()[-1])

        prompt = "the qzz89 and qzz23 the qzz89 and qzz23 the"
        expected = run("all.npz", "--prompt", prompt, "--count", "512")
        first_copy = next(i + 1 for i, row in enumerate(expected) if row["copied"])
        for split in (137, first_copy):
            left = run(f"a{split}.npz", "--prompt", prompt, "--count", str(split))
            right = run(
                f"b{split}.npz",
                "--resume",
                f"a{split}.npz",
                "--count",
                str(512 - split),
            )
            assert left + right == expected
        assert run("empty.npz", "--prompt", "Elizabeth", "--count", "0") == []
        with np.load(cwd / "empty.npz", allow_pickle=False) as blank:
            assert blank["vertices"].shape == (0, 3, 3)
        assert run("from-empty.npz", "--resume", "empty.npz", "--count", "8") == run(
            "empty-control.npz", "--prompt", "Elizabeth", "--count", "8"
        )
        run("before.npz", "--prompt", " ".join(f"qzz{i}" for i in range(100)) + " the")
        (cwd / "observations").write_bytes(b"qzzadded and")
        run("after.npz", "--resume", "before.npz", "--observe", "observations")
        reader = Reader(model)
        session = Session(reader, Speller(reader, spell, mid), 0)
        restore(cwd / "after.npz", session, mid, sid)
        assert [session.mem.token(i)[2:] for i in range(64)] == [
            f"qzz{i}".encode() for i in range(37, 100)
        ] + [b"qzzadded"]
        assert (
            int.from_bytes(session.mem.token(63)[:2], "little") == session.ids[b"the"]
        )
        assert session.prev == session.ids[b"and"]
        for wrong in (("wrong", sid), (mid, "wrong")):
            try:
                restore(cwd / "after.npz", session, *wrong)
            except ValueError as error:
                assert "digest mismatch" in str(error)
            else:
                raise AssertionError("Mismatched archive accepted")
        before = (cwd / "before.npz").read_bytes()
        failed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(script),
                str(model),
                str(spell),
                "before.npz",
                "--prompt",
                "the",
            ],
            cwd=cwd,
            capture_output=True,
        )
        assert failed.returncode and (cwd / "before.npz").read_bytes() == before
        print(
            json.dumps(
                {
                    "sampled": len(expected),
                    "first_copy": first_copy,
                    "copied_hex": expected[first_copy - 1]["hex"],
                    "empty_bytes": (cwd / "empty.npz").stat().st_size,
                    "full_64_bytes": (cwd / "after.npz").stat().st_size,
                }
            )
        )


if __name__ == "__main__":
    main()
