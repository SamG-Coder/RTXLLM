"""Generate bytes with fresh two-reflection queries against saved checkpoints."""

import argparse
from pathlib import Path

import numpy as np
from experiment import E1Scene, N, bigram, distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    stream = bytearray(args.prompt.encode("utf-8"))
    if len(stream) < 2 or args.count < 0:
        parser.error("Supply at least two prompt bytes and a nonnegative count")
    device, kernel = bigram.open_kernel()
    low = bigram.load_scene(device, args.directory / "one_hop.npz")
    global_hits, _ = bigram.trace_slots(low, kernel, 256 * N, bigram.GLOBAL_VOTES)
    prior = np.bincount(global_hits[global_hits >= 0], minlength=256)
    with np.load(args.directory / "two_hop.npz", allow_pickle=False) as saved:
        high = E1Scene(device, {k: saved[k] for k in saved.files})
    pair_kernel = device.create_compute_kernel(
        device.load_program("e1.slang", ["trace"])
    )
    rng = np.random.default_rng(args.seed)
    for _ in range(args.count):
        first, second = stream[-2:]
        hits, _ = bigram.trace_slots(low, kernel, second * N, N)
        conditional = np.bincount(hits[hits >= 0], minlength=256)[None, :]
        backoff, _ = bigram.blend(conditional, prior)
        programs = np.column_stack(
            (np.full(N, second, np.int32), np.arange(N, dtype=np.int32))
        )
        traced, _ = high.trace(pair_kernel, np.full(N, 256 + first, np.int32), programs)
        probabilities = distribution(traced["paths"][:, 2][None, :], backoff)[0]
        stream.append(int(rng.choice(256, p=probabilities)))
    print(repr(bytes(stream)))


if __name__ == "__main__":
    main()
