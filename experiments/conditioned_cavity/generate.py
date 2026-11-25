"""Sample raw bytes using only the saved conditional cavity geometry."""

import argparse
from pathlib import Path

import numpy as np
from experiment import (
    BACKOFF,
    ESCAPE,
    GRID,
    TriangleScene,
    addresses,
    open_kernel,
    read,
)
from optics import probe_strips


class Generator:
    def __init__(self, path):
        mirror, strips, self.width, grid, _ = read(path)
        if grid != GRID or self.width not in (1, 7):
            raise ValueError("Unexpected conditional cavity checkpoint")
        self.device, self.kernel = open_kernel("cavity.slang")
        self.reader = self.device.create_compute_kernel(
            self.device.load_program("cavity_readout.slang", ["trace"])
        )
        self.cavity = TriangleScene(self.device, mirror)
        self.memory = TriangleScene(self.device, strips)
        prior = self.probe([BACKOFF + 256])[0]
        if prior.sum() != 256:
            raise ValueError("Missing unigram detector")
        self.unigram = prior / 256

    def probe(self, rows):
        return probe_strips(
            self.device, self.reader, self.memory, len(rows), np.asarray(rows, np.int64)
        )

    def predict(self, context):
        last = context[-1]
        rows = [BACKOFF + last]
        if len(context) > self.width:
            history = np.frombuffer(bytes(context[-self.width - 1 :]), np.uint8)[
                None, :
            ]
            at = int(
                addresses(self.device, self.kernel, self.cavity, history, self.width)[0]
            )
            if at >= 0:
                rows.append(at)
        hits = self.probe(rows)
        back = hits[0] / 256 if hits[0].sum() else self.unigram
        high = hits[1] / 256 if len(hits) > 1 and hits[1].sum() else back
        return 0.99 * (0.8 * high + 0.2 * back) + ESCAPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    stream = bytearray(args.prompt.encode("utf-8"))
    if not stream or args.count < 0:
        parser.error("Provide a nonempty prompt and a nonnegative count")
    model = Generator(args.scene)
    rng = np.random.default_rng(args.seed)
    for _ in range(args.count):
        stream.append(int(rng.choice(256, p=model.predict(stream))))
    print(repr(bytes(stream)))


if __name__ == "__main__":
    main()
