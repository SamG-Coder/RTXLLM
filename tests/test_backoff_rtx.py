import sys
from pathlib import Path

import numpy as np
import pytest

from rtxllm.baselines.ngram import KneserNey

TEXT = np.frombuffer(b"the cat sat on the mat. \xff\xfe\x80 high \xc3\xa9t\xc3\xa9 " * 40, np.uint8).astype(np.int64)


def _rtx():
    pytest.importorskip("slangpy")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rtx"))
    import backoff          # sets the loader workaround before any adapter enumeration
    if not backoff.rtx_available():
        pytest.skip("no NVIDIA RTX Vulkan adapter")
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


def test_oldest_high_byte_selects_distinct_dishes_and_unseen_reaches_suffix():
    """Codex's #32 requests: (1) two order-6 histories differing only in a high-bit
    oldest byte hit distinct full-context dishes; (2) an unseen top-order context
    still reaches its seen suffix dish from the query bytes alone."""
    backoff = _rtx()
    device, kernel = backoff.open_backoff_kernel()
    a = bytes([0x80]) + b"abcd" + b"x"
    b_ = bytes([0xFF]) + b"abcd" + b"y"
    train = np.frombuffer((a + b" " + b_ + b" ") * 3, np.uint8).astype(np.int64)
    kn = KneserNey(6).fit([train])
    sc = backoff.LayeredScene(device, kn)
    q = np.frombuffer(a + b_ + bytes([0x81]) + b"abcd" + b"z", np.uint8).astype(np.int64)
    idx = np.array([5, 11, 17])              # targets x, y, z with 5-byte contexts
    d, _ = sc.trace(kernel, sc.points(q, idx))
    assert np.array_equal(d, sc.expected_dishes(q, idx))
    assert d[0, 0] >= 0 and d[1, 0] >= 0 and d[0, 0] != d[1, 0]     # distinct primitives
    assert d[2, 0] == -1                      # context 0x81 'abcd' never seen ...
    assert d[2, 1] >= 0 and d[2, 1] == d[0, 1] == d[1, 1]           # ... but 'abcd' is reached
    p = sc.target_prob(d, q[idx])
    assert np.allclose(p, kn.prob(q, start=0)[idx], atol=1e-12)
