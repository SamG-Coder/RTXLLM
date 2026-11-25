"""Save and resume a live E19 word session across processes."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from route import Memory, Reader, Session, Speller, WINDOW, checksum, tokenize


def save(path, session, model_id, spell_id):
    vertices = (
        session.mem.scene.vbuf.to_numpy().view(np.float32).reshape(-1, 3, 3)
        if session.history
        else np.empty((0, 3, 3), np.float32)
    )
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / "session.npz"
        np.savez_compressed(
            staged,
            version=1,
            model=model_id,
            spell=spell_id,
            previous=int(session.prev),
            rng=json.dumps(session.rng.bit_generator.state),
            vertices=vertices,
        )
        os.link(staged, path)


def restore(path, session, model_id, spell_id):
    with np.load(path, allow_pickle=False) as data:
        if int(data["version"]) != 1 or (str(data["model"]), str(data["spell"])) != (
            model_id,
            spell_id,
        ):
            raise ValueError("Session version or bound model digest mismatch")
        vertices = data["vertices"]
        prev = int(data["previous"])
        rng = json.loads(str(data["rng"]))
    if (
        not 0 <= prev < len(session.ids)
        or vertices.dtype != np.float32
        or vertices.shape[1:] != (3, 3)
        or vertices.ndim != 3
    ):
        raise ValueError("Invalid session metadata")
    mem = Memory(session.reader)
    history = []
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = rng
    if len(vertices):
        # Verify that each record is a contiguous row of canonical triangles.
        centers = vertices[:, 2].copy()
        centers[:, 1] -= 0.375
        corners = np.array(
            [[-0.375, -0.25, 0], [0.375, -0.25, 0], [0, 0.375, 0]], np.float32
        )
        if (
            not np.isfinite(centers).all()
            or not np.array_equal(vertices, centers[:, None] + corners)
            or not np.array_equal(centers, np.floor(centers))
        ):
            raise ValueError("Invalid memory triangle")
        x, y, z = centers.T
        if (
            y.min() != 0
            or y.max() >= WINDOW
            or x.min() != 0
            or z.min() < 1
            or z.max() > 257
        ):
            raise ValueError("Invalid memory dimensions")
        rows = int(y.max()) + 1
        widths = np.bincount(y.astype(int), minlength=rows)
        if (
            min(widths) < 4
            or not np.array_equal(y, np.repeat(np.arange(rows), widths))
            or not np.array_equal(
                x, np.arange(len(x)) - np.repeat(widths.cumsum() - widths, widths)
            )
        ):
            raise ValueError("Invalid memory row")
        mem.install(vertices, int(widths.max()))
        history = [mem.token(i) for i in range(rows)]
        for record in history:
            word = record[2:]
            if (
                int.from_bytes(record[:2], "little") >= len(session.ids)
                or tokenize(word) != [word]
                or word in session.ids
            ):
                raise ValueError("Invalid contextual spelling")
    session.mem, session.history = mem, history
    session.prev, session.rng = prev, restored_rng


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "spell", "output"):
        p.add_argument(key, type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--resume", type=Path)
    p.add_argument(
        "--observe", type=Path, help="File containing further complete tokens"
    )
    p.add_argument("--seed", type=int)
    p.add_argument("--count", type=int, default=0)
    args = p.parse_args()
    if (
        args.output.exists()
        or args.count < 0
        or (args.resume and args.seed is not None)
    ):
        p.error(
            "Expected a new output, nonnegative count and a seed only with a new prompt"
        )
    mid, sid = checksum(args.model), checksum(args.spell)
    reader = Reader(args.model)
    session = Session(
        reader, Speller(reader, args.spell, mid), 73 if args.seed is None else args.seed
    )
    if args.resume:
        restore(args.resume, session, mid, sid)
    else:
        session.observe(args.prompt.encode())
    if args.observe:
        session.observe(args.observe.read_bytes())
    output = session.sample(args.count)
    save(args.output, session, mid, sid)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
