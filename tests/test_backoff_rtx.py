import sys
from pathlib import Path

import numpy as np
import pytest

from rtxllm.baselines.ngram import KneserNey

TEXT = np.frombuffer(b"the cat sat on the mat. \xff\xfe\x80 high \xc3\xa9t\xc3\xa9 " * 40, np.uint8).astype(np.int64)


def _rtx():
    sp = pytest.importorskip("slangpy")
    if not any(a.vendor_id == 0x10DE and "RTX" in a.name for a in sp.Device.enumerate_adapters(sp.DeviceType.vulkan)):
        pytest.skip("no NVIDIA RTX Vulkan adapter")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rtx"))
    import backoff
    return backoff


def test_layered_rtx_equals_kneser_ney():
    backoff = _rtx()
    device, kernel = backoff.open_backoff_kernel()
    tr, ev = TEXT[:900], TEXT[900:]
    for order in (1, 4, 6):
        kn = KneserNey(order).fit([tr])
        sc = backoff.LayeredScene(device, kn)
        idx = np.arange(order - 1, len(ev))
        pts = sc.points(ev, idx)
        d, _ = sc.trace(kernel, pts)
        assert np.array_equal(d, sc.expected_dishes(ev, idx))
        assert np.allclose(sc.target_prob(d, ev[idx]), kn.prob(ev, start=order - 1), rtol=0, atol=1e-12)
        for b in range(1, order + 1):          # bounce budget: only the bottom b layers are met
            db, _ = sc.trace(kernel, pts, budget=b)
            assert np.all(db[:, :order - b] == -1)
            assert np.array_equal(db[:, order - b:], d[:, order - b:])


def test_order_limit():
    backoff = _rtx()
    device, _ = backoff.open_backoff_kernel()
    with pytest.raises(ValueError):
        backoff.LayeredScene(device, KneserNey(7).fit([TEXT]))
