"""Train, update or sample the experimental two-reflection RTX byte learner."""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments" / "rtx_credit")
)
import learn


def ingest(path, counts, frequencies, preceding=None):
    length = 0
    with path.open("rb") as source:
        while raw := source.read(1024 * 1024):
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
            frequencies += np.bincount(data, minlength=256)
            if preceding is not None:
                counts[preceding, data[0]] += 1
            keys = data[:-1] * 256 + data[1:]
            counts += np.bincount(keys, minlength=65536).reshape(256, 256)
            preceding = int(data[-1])
            length += len(data)
    if not length:
        raise ValueError("Input must contain at least one byte")
    return length, preceding


def read_geometry(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def train(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    device, kernel = learn.bigram.open_kernel()
    if args.command == "update":
        previous = args.model / "scene.npz"
        with np.load(args.model / "training.npz", allow_pickle=False) as state:
            meta = json.loads(str(state["metadata"]))
            if (
                meta["version"] != 1
                or hashlib.sha256(previous.read_bytes()).hexdigest()
                != meta["scene_sha256"]
            ):
                raise ValueError(
                    "Training state and geometry must belong to the same checkpoint"
                )
            counts, freq = state["counts"], state["frequency"]
            aim, votes = state["aim"], state["votes"]
            rng = np.random.default_rng()
            rng.bit_generator.state = meta["rng"]
        length, last = ingest(
            args.file, counts, freq, meta["last_byte"] if args.continuation else None
        )
        optical = learn.OpticalTrainer(device, kernel, read_geometry(previous))
        route, _ = optical.read()
    else:
        meta = {"total_bytes": 0, "rounds": 0}
        counts = np.zeros((256, 256), np.int64)
        freq = np.zeros(256, np.int64)
        length, last = ingest(args.file, counts, freq)
        rng = np.random.default_rng(args.seed)
        aim = rng.uniform(-0.49, learn.BANKS - 0.51, 256)
        votes = np.tile(learn.bigram.quantiles(freq, learn.N), (learn.BANKS, 1))
        optical = learn.OpticalTrainer(device, kernel, learn.make_scene(aim, votes))
        route, _ = optical.read()
    default = learn.bigram.quantiles(freq, learn.N)
    votes = learn.bank_emissions(counts, route, default)
    route, p = optical.trial(aim, votes)
    aim, votes, route, p, changes = learn.optimize_routes(
        optical, counts, default, rng, aim, votes, route, p, args.rounds
    )
    # A private staging directory is removed on failure. Publish only the
    # complete pair of files by renaming its child within the same filesystem.
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".rtxllm-build-"
    ) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir()
        checkpoint = stage / "scene.npz"
        np.savez(checkpoint, **learn.make_scene(aim, votes))
        persisted = learn.OpticalTrainer(device, kernel, read_geometry(checkpoint))
        restored_route, restored_p = persisted.read()
        np.testing.assert_array_equal(route, restored_route)
        np.testing.assert_array_equal(p, restored_p)
        metadata = {
            "version": 1,
            "total_bytes": meta["total_bytes"] + length,
            "rounds": meta["rounds"] + args.rounds,
            "last_byte": last,
            "rng": rng.bit_generator.state,
            "scene_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        np.savez_compressed(
            stage / "training.npz",
            counts=counts,
            frequency=freq,
            aim=aim,
            votes=votes,
            metadata=json.dumps(metadata),
        )
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    print(
        json.dumps(
            {
                "model": str(output),
                "bytes_ingested": length,
                "total_bytes": metadata["total_bytes"],
                "transitions": int(counts.sum()),
                "route_changes": changes,
                "training_bpb": float(learn.nll(counts, p).sum() / counts.sum())
                if counts.sum()
                else None,
            }
        )
    )


def sample_bytes(device, kernel, scene, previous, count, rng):
    import slangpy as sp

    if scene.station_count != 520 or scene.relation_count != 33:
        raise ValueError("Expected a shared eight-bank RTX byte checkpoint")
    generated = bytearray()
    for begin in range(0, count, 256):
        length = min(count - begin, 256)
        # One uniform draw per byte matches the reference's seeded choice().
        draws = (rng.random(length) * 25600).astype(np.uint32)
        output = device.create_buffer(
            data=np.full(length, -1, np.int32), usage=sp.BufferUsage.unordered_access
        )
        kernel.dispatch(
            thread_count=[32, 1, 1],
            vars={
                **scene.bindings,
                "draws": device.create_buffer(
                    data=draws, usage=sp.BufferUsage.shader_resource
                ),
                "sampled": output,
                "count": length,
                "startByte": previous,
            },
        )
        tokens = output.to_numpy().view(np.int32)
        if np.any(tokens < 0):
            raise ValueError("A generation ray failed to reach its receiver")
        generated.extend(tokens.astype(np.uint8).tobytes())
        previous = int(tokens[-1])
    return generated


def generate(args):
    stream = bytearray(args.prompt.encode("utf-8"))
    if not stream:
        raise ValueError("Prompt must contain a byte")
    device, kernel = learn.bigram.open_kernel("generate.slang")
    scene = learn.bigram.SpecularScene(device, read_geometry(args.scene))
    stream.extend(
        sample_bytes(
            device,
            kernel,
            scene,
            stream[-1],
            args.count,
            np.random.default_rng(args.seed),
        )
    )
    if args.out is None:
        print(repr(bytes(stream)))
    else:
        with args.out.open("xb") as destination:
            destination.write(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initial = commands.add_parser(
        "train", help="Fit a model to a local raw byte stream"
    )
    initial.add_argument("--seed", type=int, default=0)
    update = commands.add_parser(
        "update", help="Continue from a trainer and geometry pair"
    )
    update.add_argument("model", type=Path)
    update.add_argument(
        "--continuation",
        action="store_true",
        help="Connect the prior final byte to the new first byte",
    )
    for cmd in (initial, update):
        cmd.add_argument("file", type=Path)
        cmd.add_argument("output", type=Path, help="New model directory")
        cmd.add_argument("--rounds", type=int, default=learn.ROUNDS)
    sample = commands.add_parser("generate", help="Sample from scene.npz alone")
    sample.add_argument("scene", type=Path)
    sample.add_argument("prompt")
    sample.add_argument("--count", type=int, default=64)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument(
        "--out", type=Path, help="Write the prompt and sampled raw bytes to a new file"
    )
    args = parser.parse_args()
    if args.command == "generate":
        if args.count < 0:
            parser.error("--count must be nonnegative")
        generate(args)
    else:
        if args.rounds < 0:
            parser.error("--rounds must be nonnegative")
        train(args)


if __name__ == "__main__":
    main()
