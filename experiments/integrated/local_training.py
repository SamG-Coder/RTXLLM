"""Fit copy heights from RTX-decoded likelihoods on separate calibration documents."""

import sys
import tempfile
from pathlib import Path

import numpy as np
from model import ROOT, Predictor, fit_long_term, publish

sys.path.insert(0, str(ROOT / "experiments/copy_confidence"))
from experiment import GRID, gate


def fit_local(output, documents, calibration, order, layout):
    if output.exists():
        raise FileExistsError(output)
    scene = fit_long_term((p.read_bytes() for p in documents), order)
    losses = np.zeros((9, len(GRID)))
    counts = np.zeros(9, np.int64)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        prior = Path(tmp) / "prior.npz"
        publish(
            prior,
            version=1,
            layout=layout,
            gate_vertices=gate(np.zeros(9))["tri_vertices"],
            **scene,
        )
        predictor = Predictor(prior)
        for path in calibration:
            raw = path.read_bytes()
            if len(raw) <= 8:
                continue
            for dishes, hits, targets in predictor.blocks(raw):
                base = predictor.decode(dishes, hits, targets)
                lengths = hits[:, 2]
                correct = hits[:, 1] == targets
                counts += np.bincount(lengths, minlength=9)
                for i, w in enumerate(GRID):
                    # Short matches always use zero copy mass at inference.
                    coefficient = w * (lengths >= 2)
                    mix = (1 - coefficient) * base + coefficient * correct
                    with np.errstate(divide="ignore"):
                        bits = -np.log2(mix)
                    losses[:, i] += np.bincount(lengths, weights=bits, minlength=9)
    if not counts.sum():
        raise ValueError(
            "Calibration needs at least one document of nine or more bytes"
        )
    chosen = losses.argmin(axis=1)
    weights = np.array(GRID)[chosen]
    weights[:2] = 0
    publish(
        output,
        version=1,
        layout=layout,
        gate_vertices=gate(weights)["tri_vertices"],
        **scene,
    )
    total = losses[np.arange(9), chosen].sum()
    return {
        "model": str(output),
        "layout": layout,
        "order": int(scene["order"]),
        "calibration_targets": int(counts.sum()),
        "weights": weights.tolist(),
        "calibration_bpb": float(total / counts.sum()) if np.isfinite(total) else None,
    }
