"""#23 validation-only pilot: does beam smoothing over a byte layout beat KN?

    python experiments/e6_layered_backoff/pilot.py PRIDE.txt

Pride and Prejudice as pinned in #10 (SHA256 48e05228...), contiguous 70/15/15,
validation only. Prints KN by order, pencil = KN parity and beam deltas.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rtxllm.baselines.ngram import KneserNey, context_counts  # noqa: E402
from rtxllm.learning.byte_layers import (LayeredBackoff, random_grid, ring_beam,  # noqa: E402
                                         similarity_grid)

SHA = "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163"


def main():
    raw = Path(sys.argv[1]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA, "pinned edition required"
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1]
    d = np.frombuffer(body, np.uint8).astype(np.int64)
    tr, va, _ = np.split(d, [len(d) * 70 // 100, len(d) * 85 // 100])   # test split never read below
    out = {"kn_val_bpb_by_order": {}, "parity": {}, "beams": []}
    for n in range(1, 9):
        out["kn_val_bpb_by_order"][n] = KneserNey(n).fit([tr]).bits_per_byte([va], start=2)
    for order in (4, 5):
        kn = KneserNey(order).fit([tr])
        ref = kn.prob(va, start=order - 1)
        cc = context_counts([tr], va, order - 1, order - 1)
        for gname, g in (("similarity", similarity_grid([tr])), ("random", random_grid(0))):
            lb = LayeredBackoff(kn, g)
            _, p0 = lb.beam_bits(va, order - 1, np.zeros((1, 2)), np.ones(1))
            out["parity"][f"{order}/{gname}"] = float(np.abs(p0 - ref).max())
            for radii in ([0.5], [1.0], [2.0], [4.0], [1.0, 2.0]):
                for core in (0.95, 0.9, 0.7, 0.5):
                    o, w = ring_beam(radii, 8, core)
                    _, p = lb.beam_bits(va, order - 1, o, w)
                    dl = -np.log2(p) + np.log2(p0)
                    row = {"order": order, "grid": gname, "radii": radii, "core": core,
                           "delta_all": float(dl.mean()), "delta_rare": float(dl[cc <= 2].mean()),
                           "delta_unseen": float(dl[cc == 0].mean()),
                           "n_rare": int((cc <= 2).sum()), "n_unseen": int((cc == 0).sum())}
                    out["beams"].append(row)
                    print(json.dumps(row), flush=True)
    best = min(out["beams"], key=lambda r: r["delta_all"])
    out["best_overall"] = best
    out["best_rare"] = min(out["beams"], key=lambda r: r["delta_rare"])
    print(json.dumps({k: out[k] for k in ("kn_val_bpb_by_order", "parity", "best_overall", "best_rare")}, indent=1))
    Path(__file__).with_name("pilot_val.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
