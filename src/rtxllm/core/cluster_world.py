"""E3 synthetic world with latent structure, so unwritten facts are inferable.

Entities belong to latent clusters. For each (cluster, relation) there is a
prototype object; an entity follows it with probability ``follow`` and otherwise
has a random object (an exception). A fraction of facts is *held out*: never
written into geometry, split into val and test halves. Latent cluster labels are
only used for the oracle layout and analysis, never by learned methods.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ClusterConfig:
    seed: int = 0
    n_clusters: int = 32
    cluster_size: int = 16
    n_relations: int = 6
    follow: float = 0.8
    holdout: float = 0.2
    n_two_hop: int = 300          # per split (val, test)

    def to_dict(self):
        return asdict(self)


@dataclass
class ClusterWorld:
    config: ClusterConfig
    cluster: np.ndarray           # i32 [S] latent label
    truth: dict                   # (s, r) -> o for every fact
    written: dict                 # (s, r) -> (o,)  — what geometry may contain
    heldout: dict                 # split -> list of (s, r)
    two_hop: dict                 # split -> list of (s, r1, r2, gold)
    prototype: np.ndarray         # i32 [C, R]
    positions: np.ndarray = None  # f32 [S, 3], set by a placement method

    @property
    def n_stations(self):
        return len(self.cluster)

    # build_scene expects .facts and .config.n_relations
    @property
    def facts(self):
        return self.written

    def with_positions(self, pos):
        return ClusterWorld(self.config, self.cluster, self.truth, self.written, self.heldout,
                            self.two_hop, self.prototype, np.asarray(pos, np.float32))


def generate_clustered(config: ClusterConfig) -> ClusterWorld:
    rng = np.random.default_rng(config.seed)
    C, K, R = config.n_clusters, config.cluster_size, config.n_relations
    S = C * K
    cluster = rng.permutation(np.repeat(np.arange(C), K)).astype(np.int32)
    prototype = rng.integers(S, size=(C, R)).astype(np.int32)
    truth = {}
    for s in range(S):
        for r in range(R):
            if rng.random() < config.follow:
                o = int(prototype[cluster[s], r])
            else:
                o = int(rng.integers(S))
            if o == s:
                o = (o + 1) % S
            truth[(s, r)] = o
    keys = sorted(truth)
    order = rng.permutation(len(keys))
    n_hold = int(round(config.holdout * len(keys)))
    hold = [keys[i] for i in order[:n_hold]]
    held = set(hold)
    written = {k: (v,) for k, v in truth.items() if k not in held}
    heldout = {"val": sorted(hold[: n_hold // 2]), "test": sorted(hold[n_hold // 2:])}

    two_hop = {}
    for split in ("val", "test"):
        cands = [(s, r1, r2) for (s, r1) in heldout[split] for r2 in range(R)
                 if (truth[(s, r1)], r2) in written]
        pick = rng.choice(len(cands), size=min(config.n_two_hop, len(cands)), replace=False)
        two_hop[split] = [(s, r1, r2, truth[(truth[(s, r1)], r2)]) for s, r1, r2 in (cands[i] for i in sorted(pick))]
    return ClusterWorld(config, cluster, truth, written, heldout, two_hop, prototype)
