"""Fixed cross-book GRU comparator for the E9 and E13 RTX predictors."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TRAIN = {
    "alice-11": "49a0b2726606e1290ac03a63978fa1dd1bd38a8d805704d98265f393533ea094",
    "pride-1342": "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163",
    "frankenstein-84": "c29a4f56dd91aa894a3f17d5f96b356ccb499e3e2c5a27964de911845e39efab",
    "dracula-345": "c802711470ddb2aaca81432fc101ed29c154e7c3d32df4172478695c33f68eae",
}
VAL = {"jekyll-43": "31df0a9af1bce24ccbb5ab3ccb659603353accde5b7e40f7c5b887fb88173d48"}
TEST = {
    "war-36": "0c9bea907452abd6e9a31d9152188ef5007f0e2d3e4a52d7a6a627a37616249f",
    "moby-2701": "baf9b04fcf6e0008f30eaaa6697277751448c1070b20175e0082f28cbb161fd3",
}
WIDTH = 128
IGNORE = -100


def chunks(raw, first):
    data = np.frombuffer(raw, np.uint8)
    n = len(data) - 1
    x = np.zeros((math.ceil(n / WIDTH), WIDTH), np.uint8)
    y = np.full(x.shape, IGNORE, np.int64)
    x.flat[:n] = data[:-1]
    y.flat[:n] = data[1:]
    y.flat[: first - 1] = IGNORE
    # Also checks padding, document-local offset and the tail chunk.
    np.testing.assert_array_equal(x.ravel()[:n], data[:-1])
    np.testing.assert_array_equal(y[y != IGNORE], data[first:])
    return x, y


def dataset(folder, books, first):
    windows = []
    for name, digest in books.items():
        raw = (folder / f"rtxllm-{name}.txt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Digest mismatch for {name}")
        body = re.search(
            rb"(?ms)^\*\*\* START OF.*?\*\*\*[^\S\n]*\n(.*?)^\*\*\* END OF", raw
        )[1]
        windows.append(chunks(body, first))
    return tuple(
        torch.from_numpy(np.concatenate(parts)).cuda() for parts in zip(*windows)
    )


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 64)
        self.gru = nn.GRU(64, 192, batch_first=True)
        self.head = nn.Linear(192, 256)

    def forward(self, x):
        hidden, _ = self.gru(self.embedding(x.long()))
        return self.head(hidden).flatten(0, 1)


@torch.inference_mode()
def score(model, data, batch):
    x, y = data
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    for start in range(0, len(x), batch):
        logits = model(x[start : start + batch])
        losses = F.cross_entropy(
            logits, y[start : start + batch].flatten(), reduction="none"
        )
        total += losses.sum(dtype=torch.float64)
    count = int((y != IGNORE).sum())
    return {"targets": count, "bpb": float(total) / (count * math.log(2))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "test"))
    p.add_argument("books", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(20261125)
    model = Predictor().cuda()
    checkpoint = args.out / "best.pt"
    report = {
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "parameters": sum(t.numel() for t in model.parameters()),
    }
    if args.phase == "fit":
        x, y = dataset(args.books, TRAIN, 1)
        val = dataset(args.books, VAL, 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        best = math.inf
        history = []
        start = perf_counter()
        for epoch in range(1, 7):
            for idx in torch.randperm(len(x), device="cuda").split(64):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(x[idx]), y[idx].flatten()).backward()
                optimizer.step()
            observed = score(model, val, 64)
            history.append(observed["bpb"])
            if history[-1] < best:
                best = history[-1]
                report["selected_epoch"] = epoch
                torch.save(model.state_dict(), checkpoint)
            print(epoch, history[-1], flush=True)
        report.update(
            train_targets=int((y != IGNORE).sum()),
            val_targets=observed["targets"],
            val_bpb=history,
            train_and_validation_seconds=perf_counter() - start,
        )
    else:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        for name, digest in TEST.items():
            data = dataset(args.books, {name: digest}, 8)
            result = score(model, data, 64)
            repeat = score(model, data, 13)
            if abs(result["bpb"] - repeat["bpb"]) > 1e-6:
                raise AssertionError("Evaluation batch size changed reported loss")
            report[name] = result
            print(name, result, flush=True)
    (args.out / f"{args.phase}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
