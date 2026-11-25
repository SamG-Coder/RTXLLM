"""E18: retain a causal distribution over exact unknown-token spellings."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import slangpy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/word_spelling"))
from spell import BOOKS, Reader, Speller, WordRTX, WordScene, checksum, gpu_buffer

WINDOW = 64
WEIGHT = 0.25


def tokenize(raw):
    return re.findall(rb"[a-z0-9]+|[^\sa-z0-9]", raw.lower())


class Memory:
    def __init__(self, reader):
        self.device = reader.device

        def kernel(entry):
            return self.device.create_compute_kernel(
                self.device.load_program("spelling_recall.slang", [entry])
            )

        self.search = kernel("score")
        self.read = kernel("readRow")

    def build(self, words):
        widths = np.array([len(w) + 1 for w in words], np.int32)
        symbols = np.empty(widths.sum(), np.uint16)
        ends = widths.cumsum()
        starts = ends - widths
        for w, lo, hi in zip(words, starts, ends):
            symbols[lo : hi - 1] = np.frombuffer(w, np.uint8)
            symbols[hi - 1] = 256
        x = np.arange(len(symbols)) - np.repeat(starts, widths)
        y = np.repeat(np.arange(len(words)), widths)
        centers = np.column_stack((x, y, symbols + 1)).astype(np.float32)
        offsets = np.array(
            [[-0.375, -0.25, 0], [0.375, -0.25, 0], [0, 0.375, 0]], np.float32
        )
        self.scene = WordRTX(
            self.device, self.search, WordScene(centers[:, None] + offsets)
        )
        self.width = int(widths.max())

    def counts(self, words, delay):
        starts = np.r_[0, np.cumsum([len(w) for w in words])].astype(np.uint32)
        flat = np.frombuffer(b"".join(words), np.uint8).astype(np.uint32)
        out = gpu_buffer(
            self.device,
            np.empty((len(words), 2), np.int32),
            sp.BufferUsage.unordered_access,
        )
        self.search.dispatch(
            thread_count=[len(words), 1, 1],
            vars={
                "scene": self.scene.tlas.structure,
                "queries": len(words),
                "delay": delay,
                "starts": gpu_buffer(self.device, starts),
                "targetBytes": gpu_buffer(self.device, flat),
                "counts": out,
            },
        )
        return out.to_numpy().view(np.int32).reshape(-1, 2)

    def token(self, index):
        out = gpu_buffer(
            self.device, np.empty(self.width, np.int32), sp.BufferUsage.unordered_access
        )
        self.read.dispatch(
            thread_count=[self.width, 1, 1],
            vars={
                "scene": self.scene.tlas.structure,
                "queries": self.width,
                "rowIndex": index,
                "letters": out,
            },
        )
        symbols = out.to_numpy().view(np.int32)
        end = np.flatnonzero(symbols == 256)
        if (
            len(end) != 1
            or np.any(symbols[: end[0]] < 0)
            or np.any(symbols[end[0] + 1 :] != -1)
        ):
            raise RuntimeError("Bad spelling geometry")
        return symbols[: end[0]].astype(np.uint8).tobytes()


def cpu_counts(words, delay):
    rows = []
    for i, w in enumerate(words):
        end = max(0, i - delay)
        history = words[max(0, end - WINDOW) : end]
        rows.append((history.count(w), len(history)))
    return np.array(rows)


def evaluate(reader, spell, folder):
    report = {"books": {}}
    vocab = {w: i for i, w in enumerate(reader.words)}
    mem = Memory(reader)
    for name, sha in BOOKS.items():
        path = folder / f"rtxllm-{name}.txt"
        if checksum(path) != sha:
            raise ValueError("Book digest mismatch")
        body = re.search(
            rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", path.read_bytes()
        )[1]
        ts = tokenize(body)
        seq = np.array([vocab.get(w, 0) for w in ts])
        closed = -np.log2(reader.predict(seq[:-1], seq[1:])[:, 1].astype(float)).sum()
        words = [w for w, i in zip(ts, seq) if i == 0]
        # Token zero is context only but still contributes to later memory.
        start = int(seq[0] == 0)
        scored = words[start:]
        logp = np.array(
            [
                spell.logprob(w, False) - math.log2(spell.remaining[False])
                for w in scored
            ]
        )
        nbytes = sum(map(len, scored))
        basebits = -logp.sum()
        r = {
            "targets": len(ts) - 1,
            "unknown_targets": len(scored),
            "unknown_bytes": nbytes,
            "base_bits_per_token": (closed + basebits) / (len(ts) - 1),
            "base_spelling_bits_per_byte": basebits / nbytes,
        }
        mem.build(words)
        for delay, label in ((0, "recent"), (256, "remote")):
            counts = mem.counts(words, delay)
            np.testing.assert_array_equal(counts, cpu_counts(words, delay))
            match, n = counts[start:].T
            empirical = np.divide(match, n, out=np.zeros(len(n), float), where=n > 0)
            with np.errstate(divide="ignore"):
                mixed = np.logaddexp2(
                    logp + np.log2(1 - np.where(n > 0, WEIGHT, 0)),
                    np.log2(WEIGHT * empirical),
                )
            bits = -mixed.sum()
            r[label] = {
                "coverage": float(np.mean(match > 0)),
                "bits_per_token": (closed + bits) / (len(ts) - 1),
                "spelling_bits_per_byte": bits / nbytes,
                "gain_bits_per_token": (basebits - bits) / (len(ts) - 1),
            }
        # Replace all later geometry with a short arbitrary suffix: early counts must agree.
        prefix = words[:384] + [b"counterfactual"]
        mem.build(prefix)
        np.testing.assert_array_equal(
            mem.counts(prefix, 0)[:384], cpu_counts(words[:384], 0)
        )
        report["books"][name] = r
        print(name, r, flush=True)
    report["passes"] = all(
        r["recent"]["gain_bits_per_token"] >= 0.05 for r in report["books"].values()
    )
    return report


def generate(reader, spell, prompt, count, seed):
    ts = tokenize(prompt)
    if not ts or count < 0:
        raise ValueError("Nonempty prompt and nonnegative count required")
    vocab = {w: i for i, w in enumerate(reader.words)}
    words = [w for w in ts if w not in spell.known][-WINDOW:]
    mem = Memory(reader)
    if words:
        mem.build(words)
    rng = np.random.default_rng(seed)
    w = vocab.get(ts[-1], 0)
    output = []
    dest = np.arange(len(vocab))
    for _ in range(count):
        p = reader.predict(np.full(len(dest), w), dest)[:, 1].astype(float)
        if abs(p.sum() - 1) > 2e-6:
            raise ArithmeticError("Bad word distribution")
        w = rng.choice(dest, p=p / p.sum())
        copied = False
        if w:
            token = reader.words[w]
        else:
            copied = bool(words and rng.random() < WEIGHT)
            token = (
                mem.token(int(rng.integers(len(words)))) if copied else spell.draw(rng)
            )
            words = (words + [token])[-WINDOW:]
            mem.build(words)
        output.append({"hex": token.hex(), "copied": copied})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("test", "generate"))
    p.add_argument("model", type=Path)
    p.add_argument("spell", type=Path)
    p.add_argument("--books", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--prompt", default="Elizabeth")
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--seed", type=int, default=71)
    args = p.parse_args()
    reader = Reader(args.model)
    spell = Speller(reader, args.spell, checksum(args.model))
    if args.phase == "generate":
        print(
            json.dumps(
                generate(reader, spell, args.prompt.encode(), args.count, args.seed)
            )
        )
    else:
        if args.report is None or args.books is None:
            p.error("test requires --report and --books")
        r = evaluate(reader, spell, args.books)
        args.report.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
