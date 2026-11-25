"""Standalone fixed-context neural language-model baseline for the RTX byte pilots."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np

CORPORA = {
    "alice": ("49a0b2726606e1290ac03a63978fa1dd1bd38a8d805704d98265f393533ea094", 1),
    "pride": ("48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163", 2),
}
SEED = 20261125
BATCH = 512
EPOCHS = 12
LR = 0.003
PAD = 256


def read_splits(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Input must match the pinned edition")
    body = re.search(rb"(?ms)^\*\*\* START OF[^\n]*\n(.*?)^\*\*\* END OF", raw)[1]
    data = np.frombuffer(body, np.uint8).astype(np.int32)
    return np.split(data, [len(data) * 70 // 100, len(data) * 85 // 100])


def examples(data, width, first_target):
    # Each row ends immediately before its target; padding does not carry a
    # byte across split boundaries. Score the same targets for both widths.
    prefix = np.pad(data, (width, 0), constant_values=PAD)
    windows = np.lib.stride_tricks.sliding_window_view(prefix, width)
    return windows[first_target : len(data)], data[first_target:]


def initialize(width, rng, dtype=np.float32):
    def weight(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(dtype)

    return {
        "embedding": weight((257, 12), 0.1),
        "hidden": weight((12 * width, 24), np.sqrt(1 / (12 * width))),
        "hidden_bias": np.zeros(24, dtype),
        "output": weight((24, 256), np.sqrt(1 / 24)),
        "output_bias": np.zeros(256, dtype),
    }


def loss_and_grad(params, x, y, backward=True):
    n = len(y)
    features = params["embedding"][x].reshape(n, -1)
    hidden = np.tanh(features @ params["hidden"] + params["hidden_bias"])
    logits = hidden @ params["output"] + params["output_bias"]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = exp.sum(1, keepdims=True)
    loss = (np.log(denom[:, 0]) - logits[np.arange(n), y]).mean(dtype=np.float64)
    if not backward:
        return loss
    delta = exp / denom
    delta[np.arange(n), y] -= 1
    delta /= n
    dh = (delta @ params["output"].T) * (1 - hidden * hidden)
    de = np.zeros_like(params["embedding"])
    np.add.at(de, x.ravel(), (dh @ params["hidden"].T).reshape(-1, de.shape[1]))
    return loss, {
        "embedding": de,
        "hidden": features.T @ dh,
        "hidden_bias": dh.sum(0),
        "output": hidden.T @ delta,
        "output_bias": delta.sum(0),
    }


def check_gradients():
    rng = np.random.default_rng(7)
    params = initialize(2, rng, np.float64)
    x = np.array([[PAD, 8], [8, 8], [9, 8], [8, 9]])
    y = np.array([8, 9, 9, 2])
    _, gradients = loss_and_grad(params, x, y)
    for name, g in gradients.items():
        # Verify a nonzero component so a missing gradient cannot pass by chance.
        i = np.unravel_index(abs(g).argmax(), g.shape)
        assert abs(g[i]) > 1e-7
        saved = params[name][i]
        params[name][i] = saved + 1e-5
        hi = loss_and_grad(params, x, y, False)
        params[name][i] = saved - 1e-5
        lo = loss_and_grad(params, x, y, False)
        params[name][i] = saved
        np.testing.assert_allclose((hi - lo) / 2e-5, g[i], rtol=1e-5, atol=1e-8)
    contexts, targets = examples(np.array([65, 66, 67, 68]), 3, 1)
    np.testing.assert_array_equal(
        contexts, [[PAD, PAD, 65], [PAD, 65, 66], [65, 66, 67]]
    )
    np.testing.assert_array_equal(targets, [66, 67, 68])


def bits_per_byte(params, x, y):
    total = 0.0
    for i in range(0, len(y), 2048):
        target = y[i : i + 2048]
        total += len(target) * loss_and_grad(params, x[i : i + 2048], target, False)
    return total / (len(y) * np.log(2))


def train_and_score(splits, width, first_target):
    (x, y), (vx, vy), (tx, ty) = [
        examples(data, width, first_target) for data in splits
    ]
    rng = np.random.default_rng(SEED)
    params = initialize(width, rng)
    mean = {k: np.zeros_like(v) for k, v in params.items()}
    var = {k: np.zeros_like(v) for k, v in params.items()}
    best = None
    history = []
    step = 0
    start = perf_counter()
    for epoch in range(EPOCHS):
        perm = rng.permutation(len(y))
        for i in range(0, len(y), BATCH):
            batch = perm[i : i + BATCH]
            _, grad = loss_and_grad(params, x[batch], y[batch])
            step += 1
            for name, g in grad.items():
                mean[name] *= 0.9
                mean[name] += 0.1 * g
                var[name] *= 0.999
                var[name] += 0.001 * g * g
                corrected_m = mean[name] / (1 - 0.9**step)
                corrected_v = var[name] / (1 - 0.999**step)
                params[name] -= LR * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        val = bits_per_byte(params, vx, vy)
        history.append(val)
        if best is None or val < best[0]:
            best = val, epoch + 1, {k: v.copy() for k, v in params.items()}
    training_s = perf_counter() - start
    val, epoch, params = best
    start = perf_counter()
    test = bits_per_byte(params, tx, ty)
    return {
        "width": width,
        "seed": SEED,
        "train_targets": len(y),
        "validation_targets": len(vy),
        "test_targets": len(ty),
        "validation_bpb_by_epoch": history,
        "selected_epoch": epoch,
        "validation_bpb": val,
        "test_bpb": test,
        "test_perplexity": 2**test,
        "training_validation_s": training_s,
        "test_s": perf_counter() - start,
        "parameters": sum(v.size for v in params.values()),
        "float32_parameter_bytes": sum(v.nbytes for v in params.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CORPORA)
    parser.add_argument("file", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    check_gradients()
    digest, first_target = CORPORA[args.dataset]
    splits = read_splits(args.file, digest)
    results = []
    for width in (first_target, 8):
        metrics = train_and_score(splits, width, first_target)
        print(json.dumps(metrics), flush=True)
        results.append(metrics)
    args.result.write_text(
        json.dumps(
            {"dataset": args.dataset, "sha256": digest, "results": results}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
