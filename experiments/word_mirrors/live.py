"""Build a fitted mirror archive or generate words by tracing it at every step."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import slangpy as sp

from pilot import (
    D,
    FIRST_RANGE,
    RAYS,
    ROOT,
    WIDTH,
    Model,
    WordRTX,
    WordScene,
    beam_origins,
    fit,
    gpu_buffer,
    open_kernel,
    tokens,
)

sys.path.insert(0, str(ROOT / "experiments/word_emission"))
from scene import Reader

FORMAT = "rtxllm-word-mirrors-1"


def build(books, original, destination):
    train, _, _ = tokens(books)
    model = Model(train)
    with np.load(original, allow_pickle=False) as archive:
        if int(archive["version"]) != 1:
            raise ValueError("Expected an E14 word archive")
        np.testing.assert_array_equal(archive["vocabulary"], model.vocab)
        ids, pos, radii = (archive[k] for k in ("ids", "positions", "radii"))
        _, make_scene, _ = fit(pos, ids, model, archive["dishes"])
        mirrors = make_scene(np.arange(len(ids)))
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            stage = Path(tmp) / "model.npz"
            np.savez_compressed(
                stage,
                format=FORMAT,
                vocabulary=archive["vocabulary"],
                ids=ids,
                positions=pos,
                radii=radii,
                mirrors=mirrors.tris,
                emissions=archive["emissions"],
            )
            os.link(stage, destination)


class LiveReader(Reader):
    """Reuse E14 geometric emissions, encoding and sampling; trace the current word on demand."""

    def __init__(self, archive, max_hits=3):
        if max_hits not in (1, 2, 3):
            raise ValueError("Choose one to three hits")
        self.max_hits = max_hits
        with np.load(archive, allow_pickle=False) as a:
            if str(a["format"]) != FORMAT:
                raise ValueError("Expected a fitted word mirror archive")
            self.words = [b"<UNK>", *a["vocabulary"].tolist()]
            self.ids, self.pos, self.radii = (
                a[k] for k in ("ids", "positions", "radii")
            )
            mirrors, emissions = (WordScene(a[k]) for k in ("mirrors", "emissions"))
        self.station = {int(w): i for i, w in enumerate(self.ids)}
        self.device, self.transport = open_kernel("word_mirrors.slang")
        self.first = WordRTX(self.device, self.transport, mirrors)
        self.kernel = self.device.create_compute_kernel(
            self.device.load_program("word_emission.slang", ["trace"])
        )
        self.second = WordRTX(self.device, self.kernel, emissions)
        self.path_buffer = gpu_buffer(
            self.device, np.zeros((RAYS, 3), np.int32), sp.BufferUsage.unordered_access
        )
        self.launches = 0
        self.hit_counts = np.zeros(3, np.int64)

    def read_stations(self, word):
        weights = np.zeros(len(self.ids))
        if word in self.station:
            radius = self.radii[self.station[word]]
            origins = beam_origins(
                self.pos, [word], [radius], D, WIDTH, RAYS, back=WIDTH
            )
            self.transport.dispatch(
                thread_count=[RAYS, 1, 1],
                vars=dict(
                    scene=self.first.tlas.structure,
                    vertices=self.first.vbuf,
                    origins=gpu_buffer(self.device, origins),
                    incident=sp.float3(*D),
                    firstRange=FIRST_RANGE,
                    count=RAYS,
                    maxHits=self.max_hits,
                    paths=self.path_buffer,
                ),
            )
            path = self.path_buffer.to_numpy().view(np.int32).reshape(RAYS, 3)
            valid = path >= 0
            self.launches += 1
            self.hit_counts += valid.sum(axis=0)
            inner2 = (1.05 * radius) ** 2
            outer2 = max(WIDTH, 1.1 * radius) ** 2
            radial = np.exp(
                -0.5 * (inner2 + (outer2 - inner2) * (np.arange(RAYS) + 0.5) / RAYS)
            )
            ray, hop = np.where(valid)
            np.add.at(weights, path[ray, hop], radial[ray] / valid.sum(axis=1)[ray])
        used = np.flatnonzero(weights)
        offsets = np.zeros(len(self.words) + 1, np.uint32)
        offsets[word + 1 :] = len(used)
        # Empty GPU buffers are disallowed. Zero offsets keep the sentinel unread.
        rows = self.ids[used] if len(used) else [0]
        energy = weights[used] / weights.sum() if len(used) else [0]
        self.buffers = {
            k: gpu_buffer(self.device, a)
            for k, a in (
                ("offsets", offsets),
                ("neighbours", np.asarray(rows, np.uint32)),
                ("energy", np.asarray(energy, np.float32)),
            )
        }

    def predict(self, prev, nxt):
        out = np.empty((len(prev), 2), np.float32)
        for word in np.unique(prev):
            self.read_stations(int(word))
            mask = prev == word
            out[mask] = super().predict(prev[mask], nxt[mask])
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="action", required=True)
    builder = cmds.add_parser("build")
    for arg in ("books", "original", "destination"):
        builder.add_argument(arg, type=Path)
    gen = cmds.add_parser("generate")
    gen.add_argument("archive", type=Path)
    gen.add_argument("prompt")
    gen.add_argument("--count", type=int, default=32)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--max-hits", type=int, choices=(1, 2, 3), default=3)
    a = p.parse_args()
    if a.action == "build":
        build(a.books, a.original, a.destination)
    else:
        if a.count < 0:
            p.error("count must be nonnegative")
        reader = LiveReader(a.archive, a.max_hits)
        seq = reader.generate(a.prompt.encode(), a.count, a.seed)
        print(
            json.dumps(
                dict(
                    tokens_hex=seq,
                    beam_launches=reader.launches,
                    hits=reader.hit_counts.tolist(),
                )
            )
        )


if __name__ == "__main__":
    main()
