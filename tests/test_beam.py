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
    try:
        import beam as rtx_beam
        out = rtx_beam.parity(seed=3, n=60, rays=32)
    except RuntimeError as e:  # no RTX adapter
        pytest.skip(str(e))
    for k in ("heldout_1hop", "two_hop"):
        assert out[k]["ray_landing_agreement"] == 1.0
        assert out[k]["max_answer_diff"] < 1e-6
