"""Generate bytes from a saved mirror scene, tracing the context at every step."""

import argparse
from pathlib import Path

import numpy as np
from run import GLOBAL_VOTES, VOTES, blend, load_scene, open_kernel, trace_slots


def histogram(hits):
    return np.bincount(hits[hits >= 0], minlength=256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("prefix", help="Nonempty UTF-8 prompt")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.prefix or args.count < 0:
        parser.error("Provide a nonempty prefix and a nonnegative byte count")
    device, kernel = open_kernel()
    scene = load_scene(device, args.scene)
    global_hits, _ = trace_slots(scene, kernel, 256 * VOTES, GLOBAL_VOTES)
    prior = histogram(global_hits)
    stream = bytearray(args.prefix.encode("utf-8"))
    rng = np.random.default_rng(args.seed)
    for _ in range(args.count):
        hits, _ = trace_slots(scene, kernel, stream[-1] * VOTES, VOTES)
        conditional, _ = blend(histogram(hits)[None, :], prior)
        stream.append(int(rng.choice(256, p=conditional[0])))
    print(repr(bytes(stream)))


if __name__ == "__main__":
    main()
