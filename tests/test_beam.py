import numpy as np
import pytest

from rtxllm.core.cluster_world import ClusterConfig, generate_clustered
from rtxllm.core.scene import KIND_FACT, GeometryConfig, build_scene
from rtxllm.cpu.beam import BeamParams, BeamTracer, sunflower
from rtxllm.learning.placement import force_layout, layout_quality, random_lattice

CFG = ClusterConfig(seed=3, n_clusters=8, cluster_size=8, n_two_hop=40)


@pytest.fixture(scope="module")
def world():
    w = generate_clustered(CFG)
    return w.with_positions(force_layout(w.written, w.n_stations, 3, iters=200))


def test_heldout_facts_are_not_in_geometry(world):
    held = {k for sp in ("val", "test") for k in world.heldout[sp]}
    assert held and not held & set(world.written)
    assert not set(world.heldout["val"]) & set(world.heldout["test"])
    scene = build_scene(world)
    assert (scene.tri_kind == KIND_FACT).sum() == len(world.written)


def test_two_hop_queries_start_with_heldout_link(world):
    for split in ("val", "test"):
        held = set(world.heldout[split])
        for s, r1, r2, gold in world.two_hop[split]:
            assert (s, r1) in held
            assert gold == world.truth[(world.truth[(s, r1)], r2)]


@pytest.mark.parametrize("dish", [0.0, 0.3])
def test_broadphase_matches_all_pairs(world, dish):
    scene = build_scene(world, GeometryConfig(receiver_radius=0.15, fact_radius=0.1, dish_radius=dish))
    q = world.two_hop["val"][:20]
    res = []
    for bp in (False, True):
        bt = BeamTracer(scene, BeamParams(width=1.0, rays=32), broadphase=bp)
        res.append(bt.trace([s for s, *_ in q], [(a, b) for _, a, b, _ in q]))
    assert np.array_equal(res[0].answer, res[1].answer)
    assert np.array_equal(res[0].lost, res[1].lost)


def test_energy_accounting(world):
    scene = build_scene(world, GeometryConfig(receiver_radius=0.15, fact_radius=0.1, dish_radius=0.3))
    q = world.two_hop["val"]
    r = BeamTracer(scene, BeamParams(width=1.5, rays=64)).trace([s for s, *_ in q], [(a, b) for _, a, b, _ in q])
    assert np.allclose(r.answer.sum(1) + r.lost, 1.0, atol=1e-5)


def test_pencil_cannot_answer_unwritten_facts(world):
    scene = build_scene(world, GeometryConfig(receiver_radius=0.15, fact_radius=0.1))
    held = world.heldout["val"]
    r = BeamTracer(scene, BeamParams(width=0.0, core=1.0)).trace([s for s, _ in held], [(x,) for _, x in held])
    gold = np.array([world.truth[k] for k in held])
    assert np.mean(r.predict() == gold) < 0.1


def test_learned_layout_recovers_clusters(world):
    assert layout_quality(world.positions, world.cluster) > 0.8
    assert layout_quality(random_lattice(world.n_stations, 0), world.cluster) < 0.4


def test_sunflower_in_disk():
    p = sunflower(100, 2.0)
    assert np.all(np.linalg.norm(p, axis=1) <= 2.0 + 1e-9)
    assert len(np.unique(np.round(p, 6), axis=0)) == 100


def test_rtx_beam_parity():
    """Exact per-ray parity with the RTX backend (skipped without slangpy/RTX)."""
    pytest.importorskip("slangpy")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rtx"))
    import beam as rtx_beam
    if not rtx_beam.rtx_available():
        pytest.skip("no NVIDIA RTX Vulkan adapter")
    out = rtx_beam.parity(seed=3, n=60, rays=32)   # execution errors must fail, not skip
    for k in ("heldout_1hop", "two_hop"):
        assert out[k]["ray_landing_agreement"] == 1.0
        assert out[k]["same_beam_sets"]
        assert out[k]["max_answer_diff"] < 1e-6
        assert out[k]["max_lost_diff"] < 1e-6
        assert out[k]["max_mirror_hit_diff"] < 1e-6


@pytest.mark.parametrize("budget,rays", [(150, 2), (7, 32), (10**9, 32)])
def test_broadphase_respects_pair_budget_exactly(world, budget, rays, monkeypatch):
    """Codex's #43 case: no Moller-Trumbore call may exceed PAIR_BUDGET pairs, including
    when a beam has more rays than the budget; results equal the all-pairs reference."""
    import rtxllm.cpu.beam as beam_mod
    scene = build_scene(world, GeometryConfig(receiver_radius=0.15, fact_radius=0.1, dish_radius=0.3))
    q = world.two_hop["val"][:15]
    ref = BeamTracer(scene, BeamParams(width=1.5, rays=rays), broadphase=False).trace(
        [s for s, *_ in q], [(a, b) for _, a, b, _ in q])
    sizes, real = [], beam_mod.mt_pairs

    def spy(v0, e1, e2, o, d, ri, ti, tmax, t_min):
        sizes.append(len(ri))
        return real(v0, e1, e2, o, d, ri, ti, tmax, t_min)

    monkeypatch.setattr(beam_mod, "mt_pairs", spy)
    monkeypatch.setattr(BeamTracer, "PAIR_BUDGET", budget)
    got = BeamTracer(scene, BeamParams(width=1.5, rays=rays), broadphase=True).trace(
        [s for s, *_ in q], [(a, b) for _, a, b, _ in q])
    assert np.array_equal(got.answer, ref.answer) and np.array_equal(got.lost, ref.lost)
    assert sizes, "broadphase must have run"
    assert max(sizes) <= budget                  # every Moller-Trumbore call, emission and receiver
    if budget == 7:
        assert rays + 1 > budget                 # covers the case B exceeds the budget


def test_annular_beam_never_hits_own_dish(world):
    """E3b leave-one-out: an annulus with inner radius > dish radius cannot hit the
    station's own dish (the core slot carries zero energy)."""
    from rtxllm.core.scene import KIND_FACT
    geom = GeometryConfig(receiver_radius=0.075, fact_radius=0.05, dish_radius=0.3)
    scene = build_scene(world, geom)
    bt = BeamTracer(scene, BeamParams(width=2.0, core=0.0, inner=0.4, rays=64, fact_distance=geom.fact_distance))
    assert np.all(np.linalg.norm(bt.offsets[1:], axis=1) > 0.4 - 1e-9)
    assert bt.weights[0] == 0.0 and abs(bt.weights.sum() - 1) < 1e-6
    keys = sorted(world.written)[:40]
    r = bt.trace([s for s, _ in keys], [(x,) for _, x in keys], record=True)
    assert np.allclose(r.answer.sum(1) + r.lost, 1.0, atol=1e-5)
