"""Train, stream into and sample a bounded RTX episodic scene."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from run import (
    TriangleScene,
    dispatch,
    fit_prior,
    geometry,
    mesh,
    open_kernel,
    read_prior,
)

WINDOW = 256
TILT = np.array([(-0.3, 0.3, -0.3), (0.3, -0.3, -0.3), (0, 0, 0.3)], np.float32)


def scene_from_vertices(vertices):
    zeros = np.zeros(len(vertices), np.int32)
    return mesh(vertices, zeros, zeros, np.array([[1, 0, 0]]))


def capacity(paired):
    return 2048 if paired else WINDOW


def empty_ring(paired=False):
    rows = np.arange(2 * capacity(paired))
    centers = np.column_stack(
        (np.full(len(rows), 65537 if paired else 300), rows, np.zeros(len(rows)))
    )
    return geometry(centers, TILT)["tri_vertices"]


def place(device, raw, paired):
    column = np.frombuffer(raw, np.uint8).astype(np.int32)
    if paired:
        column[1:] += 256 * column[:-1]
    centers = np.column_stack((column, np.arange(len(raw)), np.zeros(len(raw))))
    return TriangleScene(device, geometry(centers, TILT))


def search(device, kernel, scene, n, window, paired):
    return dispatch(
        device,
        kernel,
        scene,
        np.empty((n, 4), np.int32),
        "outcomes",
        targets=n,
        windowSize=window,
        pairAddress=int(paired),
    )


class EpisodicRing:
    def __init__(self, device, ring_vertices, cursor, count, paired=False):
        self.paired = paired
        self.window = capacity(paired)
        if not (
            0 <= cursor < self.window
            and 0 <= count <= self.window
            and (count == self.window or cursor == count)
        ):
            raise ValueError("Invalid ring cursor or count")
        if ring_vertices.shape != (2 * self.window, 3, 3):
            raise ValueError("Unexpected ring geometry dimensions")
        self.device = device
        self.vertices = ring_vertices
        self.scene = TriangleScene(device, scene_from_vertices(self.vertices))
        self.kernel = device.create_compute_kernel(
            device.load_program("episodic.slang", ["live"])
        )
        self.cursor, self.count = cursor, count

    def observe(self, byte):
        if not 0 <= byte < 256:
            raise ValueError("Expected a byte")
        column = int(byte)
        if self.paired and self.count:
            # The third vertex is centered in x; the preceding byte is the low digit.
            previous = int(self.vertices[(self.cursor - 1) % self.window, 2, 0]) & 255
            column += previous * 256
        for row in (self.cursor, self.cursor + self.window):
            self.vertices[row] = np.array([column, row, 0], np.float32) + TILT
        self.scene.refit_vertices(self.vertices)
        self.cursor = (self.cursor + 1) % self.window
        self.count = min(self.count + 1, self.window)

    def query(self):
        if not self.count:
            raise ValueError("Observe a prompt byte before sampling")
        return dispatch(
            self.device,
            self.kernel,
            self.scene,
            np.empty((1, 4), np.int32),
            "outcomes",
            ringCursor=self.cursor,
            ringCount=self.count,
            windowSize=self.window,
            pairAddress=int(self.paired),
        )[0]


class OnlineMemory(EpisodicRing):
    def __init__(self, device, prior_vertices, ring_vertices, cursor, count, rng):
        if prior_vertices.shape != (65536, 3, 3):
            raise ValueError("Unexpected prior geometry dimensions")
        super().__init__(device, ring_vertices, cursor, count)
        self.rng = rng
        self.prior_vertices = prior_vertices
        self.probs = read_prior(device, scene_from_vertices(prior_vertices))
        if not np.all(self.probs > 0) or not np.allclose(
            self.probs.sum(1), 1, rtol=1e-5
        ):
            raise ValueError("Global prior geometry has invalid masses")

    @classmethod
    def train(cls, device, documents, seed):
        prior, _, _ = fit_prior(documents)
        return cls(
            device,
            prior["tri_vertices"],
            empty_ring(),
            0,
            0,
            np.random.default_rng(seed),
        )

    def sample(self):
        _, predicted, length, last = self.query()
        if last < 0 or (length >= 2 and predicted < 0):
            raise RuntimeError("Expected an occupied byte station")
        p = self.probs[last].copy()
        if length >= 2:
            p *= 0.75
            p[predicted] += 0.25
        byte = int(self.rng.choice(256, p=p / p.sum()))
        self.observe(byte)
        return byte

    def save(self, path):
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            stage = Path(directory) / "checkpoint.npz"
            np.savez_compressed(
                stage,
                version=1,
                prior=self.prior_vertices,
                ring=self.vertices,
                cursor=self.cursor,
                count=self.count,
                rng=json.dumps(self.rng.bit_generator.state),
            )
            # Publish a complete file atomically and preserve any existing destination.
            os.link(stage, path)

    @classmethod
    def load(cls, device, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("Unknown checkpoint version")
            rng = np.random.default_rng()
            rng.bit_generator.state = json.loads(str(data["rng"]))
            return cls(
                device,
                data["prior"],
                data["ring"],
                int(data["cursor"]),
                int(data["count"]),
                rng,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Write a new prior and empty ring")
    train.add_argument("output", type=Path)
    train.add_argument("documents", type=Path, nargs="+")
    train.add_argument("--seed", type=int, default=0)
    observe = commands.add_parser(
        "observe", help="Append observed bytes to an existing memory"
    )
    observe.add_argument("checkpoint", type=Path)
    observe.add_argument("stream", type=Path)
    observe.add_argument("output", type=Path)
    sample = commands.add_parser(
        "generate", help="Print generated bytes and save the advanced scene"
    )
    sample.add_argument("checkpoint", type=Path)
    sample.add_argument("output", type=Path)
    sample.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new checkpoint path")
    if args.command == "generate" and args.count < 0:
        parser.error("Count must be nonnegative")
    device, _ = open_kernel("episodic.slang")
    if args.command == "train":
        mem = OnlineMemory.train(
            device, (doc.read_bytes() for doc in args.documents), args.seed
        )
    else:
        mem = OnlineMemory.load(device, args.checkpoint)
        if args.command == "observe":
            with args.stream.open("rb") as source:
                while chunk := source.read(65536):
                    for byte in chunk:
                        mem.observe(byte)
        else:
            generated = bytes(mem.sample() for _ in range(args.count))
    mem.save(args.output)
    if args.command == "generate":
        print(repr(generated))


if __name__ == "__main__":
    main()
