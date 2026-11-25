"""Deterministic synthetic multi-hop world (E1).

The world is the *symbolic* ground truth: stations (entities/tokens), relations
and atomic facts ``(subject, relation) -> objects``. It is used to write geometry
and to score answers. Nothing in here is visible to the tracer except through
the geometry built by :mod:`rtxllm.core.scene`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

UNIQUE = "unique"        # every hop single-valued, non-empty answer
AMBIGUOUS = "ambiguous"  # at least one multi-valued hop, non-empty answer
MISS = "miss"            # some hop has no fact; correct output is abstention
CATEGORIES = (UNIQUE, AMBIGUOUS, MISS)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 0
    grid: int = 8                 # stations = grid**3
    spacing: float = 6.0          # station lattice spacing (scene units)
    jitter: float = 1.0           # uniform +- jitter per axis
    n_relations: int = 6
    fact_prob: float = 0.7        # P[(s, r) has at least one object]
    multi_prob: float = 0.05      # P[(s, r) has two objects | has a fact]
    lengths: tuple = (2, 3)
    per_bucket: dict = field(default_factory=lambda: {UNIQUE: 300, AMBIGUOUS: 100, MISS: 100})

    def to_dict(self):
        return asdict(self)


@dataclass
class Path:
    start: int
    program: tuple            # relation ids
    category: str
    answer: dict              # station -> probability (empty for MISS)
    has_cycle: bool
    split: str
    stations: tuple = ()      # symbolic station sequence for UNIQUE paths

    @property
    def key(self):
        return (self.start, self.program)


@dataclass
class World:
    config: WorldConfig
    positions: np.ndarray     # float32 [S, 3]
    facts: dict               # (s, r) -> tuple of objects
    paths: list               # list[Path]

    @property
    def n_stations(self):
        return len(self.positions)

    def split(self, name, length=None, category=None):
        return [p for p in self.paths
                if p.split == name
                and (length is None or len(p.program) == length)
                and (category is None or p.category == category)]


def traverse(facts, start, program):
    """Symbolic graph traversal (correctness ceiling).

    Energy splits equally across multi-valued objects, matching the
    partial-reflectance stacks written into geometry.
    Returns (answer distribution, single station sequence or None, any_multi).
    """
    dist = {start: 1.0}
    seq = [start]
    multi = False
    for r in program:
        nxt = {}
        for s, p in dist.items():
            objs = facts.get((s, r), ())
            for o in objs:
                nxt[o] = nxt.get(o, 0.0) + p / len(objs)
            if len(objs) > 1:
                multi = True
        dist = nxt
        if len(dist) == 1 and not multi:
            seq.append(next(iter(dist)))
        if not dist:
            break
    return dist, (tuple(seq) if not multi and dist else None), multi


def generate(config: WorldConfig) -> World:
    rng = np.random.default_rng(config.seed)
    g = config.grid
    idx = np.stack(np.meshgrid(np.arange(g), np.arange(g), np.arange(g), indexing="ij"), -1).reshape(-1, 3)
    pos = (idx - (g - 1) / 2.0) * config.spacing
    pos = pos + rng.uniform(-config.jitter, config.jitter, size=pos.shape)
    pos = pos.astype(np.float32)
    S = len(pos)

    facts = {}
    for s in range(S):
        for r in range(config.n_relations):
            if rng.random() >= config.fact_prob:
                continue
            k = 2 if rng.random() < config.multi_prob else 1
            objs = rng.choice(np.delete(np.arange(S), s), size=k, replace=False)
            facts[(s, r)] = tuple(int(o) for o in objs)

    need = {(sp, L, c): config.per_bucket[c] for sp in SPLITS for L in config.lengths for c in CATEGORIES}
    paths, seen = [], set()
    attempts = 0
    max_attempts = 200 * sum(need.values())
    while any(v > 0 for v in need.values()) and attempts < max_attempts:
        attempts += 1
        L = int(rng.choice(config.lengths))
        start = int(rng.integers(S))
        program = tuple(int(x) for x in rng.integers(config.n_relations, size=L))
        key = (start, program)
        if key in seen:
            continue
        dist, seq, multi = traverse(facts, start, program)
        cat = MISS if not dist else (AMBIGUOUS if multi else UNIQUE)
        open_splits = [sp for sp in SPLITS if need[(sp, L, cat)] > 0]
        if not open_splits:
            continue
        sp = open_splits[int(rng.integers(len(open_splits)))]
        need[(sp, L, cat)] -= 1
        seen.add(key)
        cyc = seq is not None and len(set(seq)) < len(seq)
        paths.append(Path(start, program, cat, dist, bool(cyc), sp, seq or ()))
    if any(v > 0 for v in need.values()):
        short = {k: v for k, v in need.items() if v > 0}
        raise RuntimeError(f"could not fill buckets: {short}")
    return World(config, pos, facts, paths)
