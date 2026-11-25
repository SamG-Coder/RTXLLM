"""E19: route episodic spelling recalls by a preceding word stored in the row."""

import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import slangpy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/spelling_recall"))
from recall import (
    BOOKS,
    WINDOW,
    WEIGHT,
    Memory,
    Reader,
    Speller,
    checksum,
    gpu_buffer,
    tokenize,
)


def records(words, contexts):
    return [
        int(c).to_bytes(2, "little") + w for c, w in zip(contexts, words, strict=True)
    ]


def weights(mem, n, query):
    out = gpu_buffer(mem.device, np.empty(n, np.int32), sp.BufferUsage.unordered_access)
    mem.weights.dispatch(
        thread_count=[n, 1, 1],
        vars={
            "scene": mem.scene.tlas.structure,
            "queries": n,
            "letters": out,
            "livePredecessor": int(query),
        },
    )
    return out.to_numpy().view(np.int32)


def oracle(words, contexts, queries):
    result = []
    for i, (w, q) in enumerate(zip(words, queries)):
        lower = max(0, i - WINDOW)
        h = contexts[lower:i]
        match = np.fromiter((s == w for s in words[lower:i]), bool, count=i - lower)
        mass = 1 + 7 * ((h == q) & (q != 0))
        result.append((mass[match].sum(), mass.sum()))
    return result


def evaluate(reader, spell, books):
    ids = {w: i for i, w in enumerate(reader.words)}
    mem = Memory(reader)
    report = {}
    for name, sha in BOOKS.items():
        path = books / f"rtxllm-{name}.txt"
        if checksum(path) != sha:
            raise ValueError("Book digest mismatch")
        body = re.search(
            rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", path.read_bytes()
        )[1]
        ts = tokenize(body)
        seq = np.array([ids.get(t, 0) for t in ts])
        closed = -np.log2(reader.predict(seq[:-1], seq[1:])[:, 1].astype(float)).sum()
        oov = np.flatnonzero(seq == 0)
        ctx = np.r_[0, seq[:-1]][oov]
        words = [ts[i] for i in oov]
        start = int(seq[0] == 0)
        logp = np.array(
            [
                spell.logprob(w, False) - np.log2(spell.remaining[False])
                for w in words[start:]
            ]
        )
        mem.build(records(words, ctx))
        r = {"targets": len(ts) - 1, "unknown_targets": len(logp)}
        for label, q in (
            ("uniform", np.zeros_like(ctx)),
            ("lagged", np.r_[0, ctx[:-1]]),
            ("context", ctx),
        ):
            out = mem.counts(words, 0, q)
            np.testing.assert_array_equal(out, oracle(words, ctx, q))
            hits, total = out[start:].T
            copy = np.divide(hits, total, out=np.zeros(len(total)), where=total > 0)
            with np.errstate(divide="ignore"):
                lp = np.logaddexp2(
                    logp + np.log2(1 - WEIGHT * (total > 0)), np.log2(WEIGHT * copy)
                )
            n = np.minimum(WINDOW, np.arange(len(words)))[start:]
            r[label] = {
                "bits_per_token": (closed - lp.sum()) / r["targets"],
                "fraction_rerouted": float(np.mean(total > n)),
            }
        r["gain"] = r["uniform"]["bits_per_token"] - r["context"]["bits_per_token"]
        report[name] = r
        print(name, r, flush=True)
    return {"books": report, "passes": all(r["gain"] >= 0.005 for r in report.values())}


class Session:
    def __init__(self, reader, spell, seed):
        self.reader = reader
        self.spell = spell
        self.ids = {w: i for i, w in enumerate(reader.words)}
        self.mem = Memory(reader)
        self.history = []
        self.prev = None
        self.rng = np.random.default_rng(seed)

    def observe(self, raw):
        words = tokenize(raw)
        if not words:
            raise ValueError("Expected at least one observed token")
        additions = []
        for word in words:
            current = self.ids.get(word, 0)
            if not current:
                context = self.prev if self.prev is not None else 0
                additions.append(int(context).to_bytes(2, "little") + word)
            self.prev = current
        if additions:
            self.history = (self.history + additions)[-WINDOW:]
            self.mem.build(self.history)

    def sample(self, count):
        if self.prev is None or count < 0:
            raise ValueError("Expected prompt and nonnegative count")
        output = []
        dest = np.arange(len(self.ids))
        for _ in range(count):
            p = self.reader.predict(np.full(len(dest), self.prev), dest)[:, 1].astype(
                float
            )
            if abs(p.sum() - 1) > 2e-6:
                raise ArithmeticError("Invalid word distribution")
            w = self.rng.choice(dest, p=p / p.sum())
            copied = False
            if w:
                token = self.reader.words[w]
            else:
                copied = bool(self.history and self.rng.random() < WEIGHT)
                if copied:
                    mass = weights(self.mem, len(self.history), self.prev)
                    row = self.rng.choice(len(self.history), p=mass / mass.sum())
                    token = self.mem.token(row)[2:]
                else:
                    token = self.spell.draw(self.rng)
                self.history = (
                    self.history + [int(self.prev).to_bytes(2, "little") + token]
                )[-WINDOW:]
                self.mem.build(self.history)
            output.append({"hex": token.hex(), "copied": copied})
            self.prev = w
        return output


def generate(reader, spell, prompt, count, seed):
    session = Session(reader, spell, seed)
    session.observe(prompt)
    return session.sample(count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("generate", "test"))
    p.add_argument("model", type=Path)
    p.add_argument("spell", type=Path)
    p.add_argument("--books", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--prompt", default="Elizabeth")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=73)
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
        if args.books is None or args.report is None:
            p.error("Expected --books and --report")
        mem = Memory(reader)
        mem.build(records([b"qzz", b"qxx", b"qzz"], [42, 257, 257]))
        np.testing.assert_array_equal(weights(mem, 3, 42), [8, 1, 1])
        np.testing.assert_array_equal(weights(mem, 3, 257), [1, 8, 8])
        np.testing.assert_array_equal(weights(mem, 3, 0), [1, 1, 1])
        assert mem.token(0)[2:] == b"qzz" and mem.token(1)[2:] == b"qxx"
        result = evaluate(reader, spell, args.books)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
