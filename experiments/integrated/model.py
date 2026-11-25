"""Build, score and sample the layered and episodic RTX byte model."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "src", ROOT / "src/rtx", ROOT / "experiments/episodic"):
    sys.path.insert(0, str(directory))
from backoff import LayeredScene, write_scene
from online import EpisodicRing, empty_ring, scene_from_vertices
from run import BOOKS, TriangleScene, book, dispatch, gpu_buffer, open_kernel, trace

from rtxllm.baselines.ngram import KneserNey


def publish(path, **arrays):
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        stage = Path(folder) / "archive.npz"
        np.savez_compressed(stage, **arrays)
        os.link(stage, path)


def build(output, documents, gate_path):
    with np.load(gate_path, allow_pickle=False) as gate:
        vertices = gate["tri_vertices"]
    if vertices.shape != (9, 3, 3):
        raise ValueError("Expected nine gate triangles")
    arrays = [np.frombuffer(raw, np.uint8) for raw in documents]
    longest = max(map(len, arrays), default=0)
    if longest == 0:
        raise ValueError("Training requires at least one byte")
    kn = KneserNey(min(5, longest)).fit(arrays)
    publish(output, gate_vertices=vertices, version=1, **write_scene(kn))


class Predictor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["version"]) != 1:
                raise ValueError("Unknown model version")
            gate = saved["gate_vertices"]
            scene = {
                k: saved[k]
                for k in saved.files
                if k not in ("gate_vertices", "version")
            }
        with path.open("rb") as f:
            self.identity = hashlib.file_digest(f, "sha256").hexdigest()
        self.device, self.episode_kernel = open_kernel("episodic.slang")
        self.layered = LayeredScene(self.device, scene)
        self.gate = TriangleScene(self.device, scene_from_vertices(gate))
        self.suffix_kernel = self.kernel("backoff.slang", "trace")
        self.decoder = self.kernel("copy_confidence.slang", "decode")
        self.recent = self.kernel("episodic.slang", "recentBytes")
        self.payload = {
            label: gpu_buffer(self.device, scene[key].astype(dtype))
            for label, key, dtype in (
                ("offsets", "em_ptr", np.uint32),
                ("emissionBytes", "em_byte", np.uint32),
                ("emissionMass", "em_prob", np.float32),
                ("transmission", "dish_gamma", np.float32),
            )
        }

    def kernel(self, shader, entry):
        return self.device.create_compute_kernel(
            self.device.load_program(shader, [entry])
        )

    def decode(self, dishes, episode, targets, full=False):
        p = dispatch(
            self.device,
            self.decoder,
            self.gate,
            np.empty(len(targets), np.float32),
            "weights",
            queries=len(targets),
            layers=dishes.shape[1],
            fullDistribution=int(full),
            dishIds=gpu_buffer(self.device, dishes.astype(np.int32)),
            episode=gpu_buffer(self.device, episode.astype(np.int32)),
            targetBytes=gpu_buffer(self.device, targets.astype(np.uint32)),
            **self.payload,
        )
        if np.any(p < 0) or not np.isfinite(p).all():
            raise RuntimeError("Invalid scene readout")
        return p.astype(np.float64)

    def score(self, raw, chunk=32768):
        if len(raw) <= 8:
            raise ValueError("Scoring needs at least nine bytes")
        scores = []
        for start in range(8, len(raw), chunk):
            low = max(0, start - 256)
            block = raw[low : start + chunk]
            idx = np.arange(start - low, len(block))
            data = np.frombuffer(block, np.uint8)
            episodic, _ = trace(self.device, self.episode_kernel, block)
            dishes, _ = self.layered.trace(
                self.suffix_kernel, self.layered.points(data, idx)
            )
            scores.append(self.decode(dishes, episodic[idx - 8], data[idx]))
        return np.concatenate(scores)

    def distribution(self, ring):
        episode = ring.query()[None, :]
        k = self.layered.k
        n = min(ring.count, k)
        history = np.zeros(k, np.uint8)
        if n:
            recent = dispatch(
                self.device,
                self.recent,
                ring.scene,
                np.empty(n, np.int32),
                "suffix",
                targets=n,
                ringCursor=ring.cursor,
            )
            if np.any((recent < 0) | (recent > 255)):
                raise RuntimeError("Missing recent byte")
            history[-n:] = recent
        point = self.layered.points(history, np.array([k]))
        dishes, _ = self.layered.trace(self.suffix_kernel, point, budget=n + 1)
        p = self.decode(dishes, episode, np.arange(256), full=True)
        if abs(p.sum() - 1) > 2e-5:
            raise RuntimeError("Unnormalized next-byte prediction")
        return p / p.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="action", required=True)
    writer = cmds.add_parser("build")
    writer.add_argument("gate", type=Path)
    writer.add_argument("model", type=Path)
    sources = writer.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--documents", type=Path, nargs="+", help="Independent raw training files"
    )
    sources.add_argument(
        "--e9-books", type=Path, help="Folder with four pinned E9 training books"
    )
    scorer = cmds.add_parser("score")
    scorer.add_argument("model", type=Path)
    scorer.add_argument("stream", type=Path)
    sampler = cmds.add_parser("generate")
    sampler.add_argument("model", type=Path)
    sampler.add_argument("output_state", type=Path)
    src = sampler.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=Path)
    src.add_argument("--resume", type=Path)
    sampler.add_argument("--count", type=int, default=64)
    sampler.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "build":
        docs = (
            (p.read_bytes() for p in args.documents)
            if args.documents
            else (book(args.e9_books, n) for n in list(BOOKS)[:4])
        )
        build(args.model, docs, args.gate)
        return
    model = Predictor(args.model)
    if args.action == "score":
        raw = args.stream.read_bytes()
        start = perf_counter()
        p = model.score(raw)
        print(
            json.dumps(
                {
                    "targets": len(p),
                    "bits_per_byte": float(-np.log2(p).mean()),
                    "score_wall_s": perf_counter() - start,
                }
            )
        )
        return
    if args.count < 0 or (args.resume and args.seed is not None):
        parser.error("Use a nonnegative count and seed only a new session")
    rng = np.random.default_rng(args.seed if args.seed is not None else 0)
    if args.resume:
        with np.load(args.resume, allow_pickle=False) as saved:
            if str(saved["model"]) != model.identity or int(saved["version"]) != 1:
                raise ValueError("Incompatible session or model")
            ring = EpisodicRing(
                model.device, saved["ring"], int(saved["cursor"]), int(saved["count"])
            )
            rng.bit_generator.state = json.loads(str(saved["rng"]))
    else:
        ring = EpisodicRing(model.device, empty_ring(), 0, 0)
        with args.prompt.open("rb") as prompt:
            while chunk := prompt.read(65536):
                for byte in chunk:
                    ring.observe(byte)
        if not ring.count:
            raise ValueError("Prompt must contain a byte")
    start = perf_counter()
    generated = bytearray()
    for _ in range(args.count):
        byte = int(rng.choice(256, p=model.distribution(ring)))
        generated.append(byte)
        ring.observe(byte)
    elapsed = perf_counter() - start
    publish(
        args.output_state,
        version=1,
        model=model.identity,
        ring=ring.vertices,
        cursor=ring.cursor,
        count=ring.count,
        rng=json.dumps(rng.bit_generator.state),
    )
    print(
        json.dumps(
            {
                "sample_hex": generated.hex(),
                "sampled_bytes": len(generated),
                "generation_wall_s": elapsed,
            }
        )
    )


if __name__ == "__main__":
    main()
