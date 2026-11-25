"""Spell E14 unknown tokens with an RTX-read character model."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import slangpy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/word_emission"))
from scene import Reader, WordRTX, WordScene, gpu_buffer, tokens

START = STOP = 256
SHARED = 257
ALNUM = list(b"abcdefghijklmnopqrstuvwxyz0123456789")
FIRST = [
    b
    for b in range(256)
    if bytes([b]).lower() == bytes([b]) and not bytes([b]).isspace()
]
NEXT = ALNUM + [STOP]
BOOKS = {
    "holmes-1661": "2a0ce5568625a304f54fbb6bfdeeadf527a1437254b5f0e06b5a9e233d4784a2",
    "war-36": "0c9bea907452abd6e9a31d9152188ef5007f0e2d3e4a52d7a6a627a37616249f",
    "moby-2701": "baf9b04fcf6e0008f30eaaa6697277751448c1070b20175e0082f28cbb161fd3",
}


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fit(train, known):
    counts = np.zeros((258, 257))
    counts[START, FIRST] = 0.5
    counts[np.ix_(ALNUM + [SHARED], NEXT)] = 0.5
    for word in train:
        if word in known:
            continue
        counts[START, word[0]] += 1
        if word[0] in ALNUM:
            for a, b in zip(word, [*word[1:], STOP]):
                counts[a, b] += 1
                counts[SHARED, b] += 1
    return counts / np.maximum(counts.sum(1, keepdims=True), 1)


def store(path, probs, identity):
    y, x = np.nonzero(probs)
    centers = np.column_stack((x, y, probs[y, x])).astype(np.float32)
    corners = np.array(
        [[-0.375, -0.25, 0], [0.375, -0.25, 0], [0, 0.375, 0]], np.float32
    )
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        stage = Path(tmp) / "spell.npz"
        np.savez_compressed(stage, triangles=centers[:, None] + corners, model=identity)
        os.link(stage, path)


class Speller:
    def __init__(self, reader, path, identity):
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["model"]) != identity:
                raise ValueError("Vocabulary archive mismatch")
            mesh = WordScene(saved["triangles"])
        dev = reader.device
        kernel = dev.create_compute_kernel(
            dev.load_program("word_spelling.slang", ["trace"])
        )
        scene = WordRTX(dev, kernel, mesh)
        out = gpu_buffer(
            dev, np.empty((258, 257), np.float32), sp.BufferUsage.unordered_access
        )
        kernel.dispatch(
            thread_count=[258 * 257, 1, 1],
            vars={
                "queries": 258 * 257,
                "scene": scene.tlas.structure,
                "probabilities": out,
            },
        )
        self.probs = out.to_numpy().view(np.float32).reshape(258, 257).astype(float)
        active = ALNUM + [START, SHARED]
        np.testing.assert_allclose(self.probs[active].sum(1), 1, atol=2e-6, rtol=0)
        self.probs[active] /= self.probs[active].sum(1, keepdims=True)
        self.known = set(reader.words[1:])
        self.remaining = {}
        for shared in (False, True):
            excluded = math.fsum(2 ** self.logprob(word, shared) for word in self.known)
            if not 0 <= excluded < 1:
                raise ArithmeticError("Invalid excluded vocabulary mass")
            self.remaining[shared] = 1 - excluded
        self.triangles = len(mesh.tris)
        self.as_bytes = scene.as_bytes

    def logprob(self, word, shared):
        bits = math.log2(self.probs[START, word[0]])
        if word[0] in ALNUM:
            for a, b in zip(word, [*word[1:], STOP]):
                bits += math.log2(self.probs[SHARED if shared else a, b])
        return bits

    def draw(self, rng):
        while True:
            word = bytearray([int(rng.choice(257, p=self.probs[START]))])
            if word[0] in ALNUM:
                while (n := rng.choice(257, p=self.probs[word[-1]])) != STOP:
                    word.append(int(n))
            token = bytes(word)
            if token not in self.known:
                return token


def check_exclusion(spell, reference):
    # Independent complement calculation. Stop walking a prefix as soon as it
    # differs from every known token; all continuations from there are allowed.
    trie = {}
    for word in spell.known:
        node = trie
        for byte in word:
            node = node.setdefault(byte, {})
        node[STOP] = True
    for shared in (False, True):
        free = 0.0
        stack = [(trie, START, 1.0)]
        while stack:
            node, last, prefix = stack.pop()
            for byte in FIRST if last == START else NEXT:
                row = SHARED if shared and last != START else last
                mass = prefix * reference[row, byte]
                if byte not in node:
                    free += mass
                elif byte in ALNUM:
                    stack.append((node[byte], byte, mass))
        np.testing.assert_allclose(spell.remaining[shared], free, atol=2e-7, rtol=0)


def evaluate(reader, spell, folder):
    report = {
        "triangles": spell.triangles,
        "as_bytes": spell.as_bytes,
        "remaining": {
            "character": spell.remaining[False],
            "shared": spell.remaining[True],
        },
        "books": {},
    }
    vocab = {w: i for i, w in enumerate(reader.words)}
    for name, sha in BOOKS.items():
        path = folder / f"rtxllm-{name}.txt"
        if checksum(path) != sha:
            raise ValueError(name)
        body = re.search(
            rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", path.read_bytes()
        )[1]
        ts = re.findall(rb"[a-z0-9]+|[^\sa-z0-9]", body.lower())
        seq = np.array([vocab.get(t, 0) for t in ts])
        p = reader.predict(seq[:-1], seq[1:])[:, 1].astype(float)
        closed = -np.log2(p).sum()
        counts = Counter(w for w in ts[1:] if w not in spell.known)
        n = sum(len(w) * cnt for w, cnt in counts.items())
        r = {
            "targets": len(p),
            "unknown_tokens": counts.total(),
            "unknown_token_bytes": n,
        }
        for shared, key in ((False, "character"), (True, "shared")):
            spelling = math.fsum(
                cnt * (math.log2(spell.remaining[shared]) - spell.logprob(w, shared))
                for w, cnt in counts.items()
            )
            r[key] = {
                "spelling_bits_per_unknown_byte": spelling / n,
                "open_bits_per_token": (closed + spelling) / len(p),
            }
        gain = (
            r["shared"]["spelling_bits_per_unknown_byte"]
            - r["character"]["spelling_bits_per_unknown_byte"]
        )
        r["gain_bits_per_unknown_byte"] = gain
        report["books"][name] = r
        print(name, r, flush=True)
    report["passes"] = all(
        r["gain_bits_per_unknown_byte"] >= 0.05 for r in report["books"].values()
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "test", "generate"))
    parser.add_argument("model", type=Path)
    parser.add_argument("spell", type=Path)
    parser.add_argument("--books", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--prompt", default="Elizabeth")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    reader = Reader(args.model)
    identity = checksum(args.model)
    if args.phase != "generate" and args.books is None:
        parser.error("build and test require --books")
    if args.phase == "build":
        tr, _, _ = tokens(args.books)
        ref = fit(tr, set(reader.words[1:]))
        store(args.spell, ref, identity)
        actual = Speller(reader, args.spell, identity)
        np.testing.assert_allclose(actual.probs, ref, atol=2e-6, rtol=0)
        check_exclusion(actual, ref)
        return
    spell = Speller(reader, args.spell, identity)
    if args.phase == "test":
        if args.report is None:
            parser.error("test requires --report")
        result = evaluate(reader, spell, args.books)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        seq = reader.encode(args.prompt.encode())
        if not len(seq) or args.count < 0:
            parser.error("Expected nonempty prompt and nonnegative count")
        w = seq[-1]
        rng = np.random.default_rng(args.seed)
        vocab = np.arange(len(reader.words))
        output = []
        for _ in range(args.count):
            p = reader.predict(np.full(len(vocab), w), vocab)[:, 1].astype(float)
            if abs(p.sum() - 1) > 2e-6:
                raise ArithmeticError("Invalid word distribution")
            w = rng.choice(vocab, p=p / p.sum())
            token = reader.words[w] if w else spell.draw(rng)
            output.append({"hex": token.hex(), "spelled": bool(w == 0)})
        print(json.dumps(output))


if __name__ == "__main__":
    main()
