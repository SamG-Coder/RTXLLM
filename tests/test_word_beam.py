import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rtx"))


def _mod():
    pytest.importorskip("slangpy")
    import word_beam
    return word_beam


def test_hex_dish_and_annulus_geometry():
    wb = _mod()
    d = np.array([0.0, 0.0, 1.0])
    tris = wb.hex_dishes(np.array([[0, 0, 1.0], [5, 0, 1.0]]), np.array([0.5, 0.25]), d)
    assert tris.shape == (12, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert np.allclose(np.abs(n[:, 2]) / np.linalg.norm(n, axis=1), 1.0)    # facing along d
    a = wb.annulus(200, 0.3, 1.0)
    r = np.linalg.norm(a, axis=1)
    assert r.min() > 0.3 and r.max() <= 1.0 + 1e-12


def test_annular_beam_never_reads_own_dish_cpu():
    wb = _mod()
    d = np.array([0.0, 0.0, 1.0])
    pos = np.array([[0, 0, 0], [0.4, 0, 0], [3, 3, 0]], float)
    radii = np.array([0.2, 0.2, 0.2])
    sc = wb.WordScene(wb.hex_dishes(pos + 1.5 * d, radii, d))
    o = wb.beam_origins(pos, np.array([0, 2]), radii[[0, 2]], d, 1.0, 256, back=1.0)
    h = sc.cpu_hits(o, d, 4.0).reshape(2, 256)
    assert not np.any(h[0] == 0) and np.any(h[0] == 1)      # reads neighbour 1, never itself
    assert not np.any(h[1] == 2) and np.all(h[1] == -1)     # isolated word: nothing in its beam


def test_rtx_word_beam_matches_cpu():
    wb = _mod()
    import slangpy as sp
    if not any(a.vendor_id == 0x10DE and "RTX" in a.name for a in sp.Device.enumerate_adapters(sp.DeviceType.vulkan)):
        pytest.skip("no NVIDIA RTX Vulkan adapter")
    rng = np.random.default_rng(0)
    d = np.array([0.36, 0.48, 0.8]) / np.linalg.norm([0.36, 0.48, 0.8])
    pos = rng.normal(size=(300, 3)) * 2
    radii = 0.1 + 0.1 * rng.random(300)
    sc = wb.WordScene(wb.hex_dishes(pos + 1.5 * d, radii, d))
    device, kernel = wb.open_word_kernel()
    g = wb.WordRTX(device, kernel, sc)
    o = wb.beam_origins(pos, np.arange(60), radii[:60], d, 1.0, 128, back=1.0)
    hit, _ = g.hits(o, d, 4.5)
    assert np.array_equal(hit, sc.cpu_hits(o, d, 4.5))
