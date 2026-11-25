"""Persist E12 and score or sample words through two spatial scenes."""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import slangpy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e12_word_layout"))
from run import D, FACT_DISTANCE, RAYS, Model, force_word_layout, scene_for, tokens
from word_beam import WordRTX, WordScene, beam_origins, gpu_buffer, open_word_kernel

WIDTH = 2.5


def store(model, ids, pos, radii, dishes, path):
    src, dst = np.nonzero(model.B)
    counts = model.B[src, dst]
    discount = model.Dk[np.minimum(counts, 3).astype(int)]
    total = model.rowtot[src]
    gamma = np.divide(
        np.dot(model.Dk[1:], np.array(model.N)),
        model.rowtot,
        out=np.ones(model.V),
        where=model.rowtot > 0,
    )
    words = np.arange(model.V)
    x = np.concatenate((dst, dst, np.full(model.V, model.V), words))
    y = np.concatenate(
        (src, src + model.V, words + model.V, np.full(model.V, 2 * model.V))
    )
    z = np.concatenate(
        (counts / total, np.maximum(counts - discount, 0) / total, gamma, model.pcont)
    )
    centers = np.column_stack((x, y, z))[z > 0].astype(np.float32)
    offsets = np.array(
        [[-0.375, -0.25, 0], [0.375, -0.25, 0], [0, 0.375, 0]], np.float32
    )
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        stage = Path(tmp) / "scene.npz"
        np.savez_compressed(
            stage,
            version=1,
            vocabulary=np.array(model.vocab),
            ids=ids,
            positions=pos,
            radii=radii,
            dishes=dishes.tris,
            emissions=centers[:, None] + offsets,
        )
        os.link(stage, path)


class Reader:
    def __init__(self, path, shuffle=False):
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["version"]) != 1:
                raise ValueError("Unknown word scene version")
            self.words = [b"<UNK>", *archive["vocabulary"].tolist()]
            ids, pos, radii = (archive[k] for k in ("ids", "positions", "radii"))
            first, second = (WordScene(archive[k]) for k in ("dishes", "emissions"))
        self.device, first_kernel = open_word_kernel()
        self.first = WordRTX(self.device, first_kernel, first)
        self.kernel = self.device.create_compute_kernel(
            self.device.load_program("word_emission.slang", ["trace"])
        )
        self.second = WordRTX(self.device, self.kernel, second)
        self.sizes = {
            "dish_triangles": len(first.tris),
            "emission_triangles": len(second.tris),
            "dish_as_bytes": self.first.as_bytes,
            "emission_as_bytes": self.second.as_bytes,
        }
        origins = beam_origins(pos, ids, radii, D, WIDTH, RAYS, back=WIDTH)
        hit, _ = self.first.hits(origins, D, 2 * WIDTH + FACT_DISTANCE + 0.5)
        hit = hit.reshape(len(ids), RAYS)
        if np.any(hit == np.arange(len(ids))[:, None]):
            raise AssertionError("Annular beam read its own dish")
        row_ids = np.random.default_rng(61).permutation(ids) if shuffle else ids
        # Rays hitting the same dish select the same continuation row. Aggregate
        # their energy once per query word; the second intersections run on demand.
        offsets = np.zeros(len(self.words) + 1, np.uint32)
        neighbours, energies = [], []
        for word, radius, hits in zip(ids, radii, hit):
            valid = hits >= 0
            inner, outer = 1.05 * radius, max(WIDTH, 1.1 * radius)
            r2 = inner**2 + (outer**2 - inner**2) * (np.flatnonzero(valid) + 0.5) / RAYS
            weights = np.bincount(
                hits[valid], weights=np.exp(-0.5 * r2), minlength=len(ids)
            )
            used = np.flatnonzero(weights)
            offsets[word + 1] = len(used)
            neighbours.extend(row_ids[used])
            if len(used):
                energies.extend(weights[used] / weights.sum())
        self.buffers = {
            k: gpu_buffer(self.device, a)
            for k, a in (
                ("offsets", offsets.cumsum(dtype=np.uint32)),
                ("neighbours", np.array(neighbours, np.uint32)),
                ("energy", np.array(energies, np.float32)),
            )
        }

    def predict(self, prev, nxt):
        out = gpu_buffer(
            self.device,
            np.zeros((len(prev), 2), np.float32),
            sp.BufferUsage.unordered_access,
        )
        self.kernel.dispatch(
            thread_count=[len(prev), 1, 1],
            vars={
                "scene": self.second.tlas.structure,
                "queryCount": len(prev),
                "vocabSize": len(self.words),
                "previous": gpu_buffer(self.device, prev.astype(np.uint32)),
                "following": gpu_buffer(self.device, nxt.astype(np.uint32)),
                "result": out,
                **self.buffers,
            },
        )
        return out.to_numpy().view(np.float32).reshape(-1, 2)

    def encode(self, raw):
        tokens = re.findall(rb"[a-z0-9]+|[^\sa-z0-9]", raw.lower())
        ids = {word: i for i, word in enumerate(self.words)}
        return np.array([ids.get(token, 0) for token in tokens], np.int32)

    def score(self, raw):
        sequence = self.encode(raw)
        if len(sequence) < 2:
            raise ValueError("Scoring requires two or more tokens")
        n = len(sequence) - 1
        bits = np.zeros(2, np.float64)
        for start in range(0, n, 32768):
            stop = min(start + 32768, n)
            p = self.predict(sequence[start:stop], sequence[start + 1 : stop + 1])
            bits -= np.log2(p.astype(np.float64)).sum(axis=0)
        prior, beam = np.exp2(bits / n)
        return {"targets": n, "prior_ppl": float(prior), "beam_ppl": float(beam)}

    def generate(self, prompt, n, seed):
        sequence = self.encode(prompt)
        if not len(sequence):
            raise ValueError("Prompt needs a token")
        word = sequence[-1]
        rng = np.random.default_rng(seed)
        vocab = np.arange(len(self.words))
        output = []
        for _ in range(n):
            p = self.predict(np.full(len(vocab), word), vocab)[:, 1].astype(float)
            if not np.isfinite(p).all() or min(p) < 0 or abs(p.sum() - 1) > 2e-6:
                raise ArithmeticError("Invalid traced distribution")
            word = rng.choice(vocab, p=p / p.sum())
            output.append(self.words[word].hex())
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="action", required=True)
    build = cmd.add_parser("build")
    build.add_argument("books", type=Path)
    build.add_argument("archive", type=Path)
    score = cmd.add_parser("score")
    score.add_argument("archive", type=Path)
    score.add_argument("document", type=Path)
    gen = cmd.add_parser("generate")
    gen.add_argument("archive", type=Path)
    gen.add_argument("prompt")
    gen.add_argument("--count", type=int, default=32)
    gen.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    if a.action == "build":
        train, _, _ = tokens(a.books)
        m = Model(train)
        ids, x = force_word_layout(m)
        pos, radii, scene = scene_for(m, ids, x)
        store(m, ids, pos, radii, scene, a.archive)
    elif a.action == "score":
        print(json.dumps(Reader(a.archive).score(a.document.read_bytes())))
    else:
        if a.count < 0:
            parser.error("count must be nonnegative")
        print(
            json.dumps(Reader(a.archive).generate(a.prompt.encode(), a.count, a.seed))
        )


if __name__ == "__main__":
    main()
