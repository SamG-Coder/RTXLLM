import numpy as np
import pytest

from rtxllm.core.scene import (KIND_FACT, KIND_RECEIVER, GeometryConfig, Scene, TraceParams,
                               build_scene, icosphere, relation_directions, shuffle_destinations)
from rtxllm.core.world import MISS, SPLITS, UNIQUE, World, WorldConfig, generate, traverse
from rtxllm.cpu.tracer import intersect, trace

SMALL = WorldConfig(seed=7, grid=4, per_bucket={"unique": 20, "ambiguous": 5, "miss": 10})


@pytest.fixture(scope="module")
def world():
    return generate(SMALL)


def test_generation_is_deterministic():
    a, b = generate(SMALL), generate(SMALL)
    assert np.array_equal(a.positions, b.positions)
    assert a.facts == b.facts
    assert [(p.key, p.split) for p in a.paths] == [(p.key, p.split) for p in b.paths]


def test_splits_are_disjoint_and_unseen(world):
    keys = {sp: {p.key for p in world.split(sp)} for sp in SPLITS}
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    assert not keys["train"] & keys["val"]
    # held-out paths are composed (length >= 2), never stored atomic facts
    assert all(len(p.program) >= 2 for p in world.split("test"))


def test_scene_contains_no_fact_table(world):
    scene = build_scene(world)
    facts = scene.tri_kind == KIND_FACT
    assert np.all(scene.tri_station[facts] == -1), "fact geometry must not name a destination"
    assert np.all(scene.tri_displacement[facts] == 0), "specular facts carry no displacement"
    assert facts.sum() == sum(len(v) for v in world.facts.values())


def test_intersect_single_triangle():
    V = np.array([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]], np.float32)
    sc = Scene(V, np.array([KIND_RECEIVER], np.uint32), np.array([0], np.int32), np.zeros(1, np.float32),
               np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32), np.eye(3, dtype=np.float32))
    o = np.array([[0, 0, -2], [5, 5, -2], [0, 0, -2]], np.float32)
    d = np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1]], np.float32)
    t, i = intersect(sc, o, d, np.full(3, 10, np.float32), np.ones(3, bool), np.float32(1e-3))
    assert i.tolist() == [0, -1, -1]
    assert abs(t[0] - 2.0) < 1e-6


def test_relation_directions_unit_and_separated():
    d = relation_directions(6, 1)
    assert np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-6)
    cos = d @ d.T - np.eye(6) * 2
    assert cos.max() < 0.5


@pytest.mark.parametrize("mode", ["specular", "transport"])
def test_unique_paths_follow_symbolic_route(world, mode):
    scene = build_scene(world, params=TraceParams(hop_mode=mode))
    paths = world.split("val", category=UNIQUE)
    r = trace(scene, [p.start for p in paths], [p.program for p in paths])
    gold = np.array([p.stations[-1] for p in paths])
    assert np.mean(r.predict() == gold) >= 0.9


@pytest.mark.parametrize("mode", ["specular", "transport"])
def test_energy_is_conserved(world, mode):
    scene = build_scene(world, params=TraceParams(hop_mode=mode))
    paths = world.split("val")
    r = trace(scene, [p.start for p in paths], [p.program for p in paths])
    total = r.answer.sum(1) + r.miss + r.floor + r.overflow
    assert np.allclose(total, 1.0, atol=1e-5)


def test_miss_paths_abstain(world):
    scene = build_scene(world)
    paths = world.split("val", category=MISS)
    r = trace(scene, [p.start for p in paths], [p.program for p in paths])
    assert np.mean(r.predict() == -1) >= 0.9


def test_multivalued_fact_splits_energy():
    pos = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], np.float32)
    cfg = WorldConfig(seed=0, grid=1, n_relations=1)
    w = World(cfg, pos, {(0, 0): (1, 2)}, [])
    for mode in ("specular", "transport"):
        sc = build_scene(w, params=TraceParams(hop_mode=mode))
        r = trace(sc, [0], [(0,)])
        assert np.allclose(r.answer[0, [1, 2]], [0.5, 0.5], atol=1e-6), (mode, r.answer[0])


def test_destination_is_encoded_only_by_mirror_orientation():
    """Rotating a single mirror re-routes the ray: the memory is the geometry."""
    pos = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], np.float32)
    w = World(WorldConfig(seed=0, grid=1, n_relations=1), pos, {(0, 0): (1,)}, [])
    sc = build_scene(w)
    assert trace(sc, [0], [(0,)]).predict()[0] == 1
    w2 = World(w.config, pos, {(0, 0): (2,)}, [])
    mirror = build_scene(w2).tri_vertices[sc.tri_kind == KIND_FACT]
    V = sc.tri_vertices.copy()
    V[sc.tri_kind == KIND_FACT] = mirror
    from dataclasses import replace
    assert trace(replace(sc, tri_vertices=V), [0], [(0,)]).predict()[0] == 2


def test_shuffled_geometry_breaks_routing(world):
    scene = shuffle_destinations(build_scene(world), seed=3)
    paths = world.split("val", category=UNIQUE)
    r = trace(scene, [p.start for p in paths], [p.program for p in paths])
    gold = np.array([p.stations[-1] for p in paths])
    assert np.mean(r.predict() == gold) < 0.2


def test_scene_roundtrip(tmp_path, world):
    scene = build_scene(world, GeometryConfig(receiver_radius=0.2), TraceParams(hop_mode="transport"))
    scene.save(tmp_path / "scene.npz")
    back = Scene.load(tmp_path / "scene.npz")
    assert back.params == scene.params
    for k in Scene.ARRAYS:
        assert np.array_equal(getattr(back, k), getattr(scene, k))


def test_icosphere_closed():
    v, f = icosphere(1.0)
    assert len(f) == 20
    edges = {}
    for tri in f:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[frozenset((a, b))] = edges.get(frozenset((a, b)), 0) + 1
    assert set(edges.values()) == {2}


def test_committed_fixtures_match_reference():
    """Parity fixtures for the RTX backend must match the current CPU reference."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    import make_fixture
    for mode in ("specular", "transport"):
        path = Path(__file__).parent / "fixtures" / f"e1_small_{mode}.npz"
        z = np.load(path)
        fresh = make_fixture.build(mode)
        for k in ("expected_pred", "expected_hops", "start", "program", "tri_station"):
            assert np.array_equal(z[k], fresh[k]), (mode, k)
        for k in ("expected_answer", "expected_miss", "tri_vertices"):
            assert np.allclose(z[k], fresh[k], atol=1e-6), (mode, k)


def test_per_hop_agreement_counts_paths_without_finished_lineage():
    from types import SimpleNamespace
    from rtxllm.core.world import Path as WPath
    from rtxllm.evaluation import per_hop_agreement
    paths = [WPath(0, (0, 1), UNIQUE, {2: 1.0}, False, "val", (0, 1, 2)),
             WPath(0, (0, 1), UNIQUE, {2: 1.0}, False, "val", (0, 1, 2))]
    res = SimpleNamespace(receiver_paths=[[((0, 1, 2), 1.0)], []])  # 2nd: all energy overflowed
    assert per_hop_agreement(res, paths) == 0.5
