"""Scene interchange format and geometry writers.

A scene is triangles only (so RTX traversal is fully hardware-accelerated) plus a
handful of per-triangle material parameters. It deliberately contains **no fact
table and no destination identifiers**: in ``specular`` mode a fact's destination
exists only as a mirror orientation; in ``transport`` mode as a displacement
vector. Receiver triangles carry their station id (the stable token identity).

File format: ``scene.npz`` with the arrays of :class:`Scene` and a JSON string
``params``. See docs/decisions/0001-architecture.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, replace

import numpy as np

KIND_RECEIVER = 0
KIND_FACT = 1
SPECULAR = "specular"
TRANSPORT = "transport"


@dataclass(frozen=True)
class TraceParams:
    hop_mode: str = SPECULAR
    emit_offset: float = 0.3      # emission starts this far from the station centre
    emit_range: float = 3.0       # tmax of an emission leg (a station's reach)
    transport_range: float = 1.0e4
    arrival_range: float = 1.0    # transport mode: tmax of the post-displacement arrival leg
    t_min: float = 1.0e-3
    max_events: int = 16          # intersections per ray lineage
    energy_floor: float = 1.0e-3
    transport_hits_facts: bool = True   # False: transport legs masked to receivers only


@dataclass(frozen=True)
class GeometryConfig:
    receiver_radius: float = 0.1
    fact_radius: float = 0.1      # circumradius of the equilateral fact triangle
    fact_distance: float = 1.5    # distance of first fact quad from its station along d_r
    stack_gap: float = 0.5        # spacing of stacked (multi-valued) quads
    relation_seed: int = 12345
    dish_radius: float = 0.0      # >0: fact is a focusing (Fresnel) dish of facets covering this radius


@dataclass
class Scene:
    tri_vertices: np.ndarray      # f32 [T,3,3]
    tri_kind: np.ndarray          # u32 [T]
    tri_station: np.ndarray       # i32 [T] receiver -> station, -1 for facts
    tri_reflectance: np.ndarray   # f32 [T] fraction reflected/transported at a fact; rest transmits
    tri_displacement: np.ndarray  # f32 [T,3] transport mode only (zeros otherwise)
    station_pos: np.ndarray       # f32 [S,3]
    relation_dir: np.ndarray      # f32 [R,3] unit
    params: TraceParams = field(default_factory=TraceParams)
    tri_relation: np.ndarray = None  # i32 [T] relation channel of a fact (-1 receivers); ADR 0002

    ARRAYS = ("tri_vertices", "tri_kind", "tri_station", "tri_reflectance",
              "tri_displacement", "station_pos", "relation_dir", "tri_relation")

    def __post_init__(self):
        if self.tri_relation is None:
            self.tri_relation = np.full(len(self.tri_kind), -1, np.int32)

    def save(self, path):
        np.savez(path, params=json.dumps(asdict(self.params)),
                 **{k: getattr(self, k) for k in self.ARRAYS})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in cls.ARRAYS if k in z.files},
                   params=TraceParams(**json.loads(str(z["params"]))))

    def with_params(self, **kw):
        return replace(self, params=replace(self.params, **kw))

    @property
    def n_triangles(self):
        return len(self.tri_kind)


def icosphere(radius):
    """Icosahedron (12 vertices, 20 faces) scaled to circumradius ``radius``."""
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v * radius, f


def relation_directions(n, seed):
    """Well-separated unit directions, randomly rotated so they avoid lattice axes."""
    if n <= 6:
        v, _ = icosphere(1.0)
        base = v[[0, 1, 4, 5, 8, 9]][:n]   # one from each antipodal pair
    else:
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        th = np.pi * (1 + 5 ** 0.5) * i
        base = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    d = base @ q.T
    return (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float32)


def _fact_triangle(center, normal, radius):
    """One equilateral triangle whose centroid is ``center``.

    The emission ray passes exactly through ``center``, so the hit lands on the
    centroid, far from any edge. (A two-triangle quad put the hit on the shared
    diagonal, where non-watertight intersection dropped ~1% of hits.)
    """
    a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(normal, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    ang = np.array([0.0, 2.0, 4.0]) * np.pi / 3.0
    return np.array([center + radius * (np.cos(x) * t1 + np.sin(x) * t2) for x in ang])


def _dish_centroids(radius, facet_radius):
    """Centroids (2D) of a triangular tiling whose central triangle is centred on
    the origin, clipped to a disk. The pencil ray through the origin therefore
    hits a facet centroid, never an edge."""
    a = facet_radius * 3 ** 0.5          # side of an equilateral triangle
    e1, e2 = np.array([a, 0.0]), np.array([a / 2, a * 3 ** 0.5 / 2])
    shift = (e1 + e2) / 3.0
    n = int(np.ceil(radius / a)) + 2
    cs = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            b = i * e1 + j * e2
            for c in (b + (e1 + e2) / 3.0, b + 2 * (e1 + e2) / 3.0):
                c = c - shift
                if np.hypot(*c) <= radius:
                    cs.append(c)
    cs.sort(key=lambda c: (np.hypot(*c), c[0], c[1]))
    return np.asarray(cs)


def _fact_triangles(centers, normals, radius):
    """Vectorised :func:`_fact_triangle` for [n,3] centres and normals."""
    a = np.where((np.abs(normals[:, 0]) < 0.9)[:, None], [1.0, 0, 0], [0, 1.0, 0])
    t1 = np.cross(normals, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)
    ang = np.array([0.0, 2.0, 4.0]) * np.pi / 3.0
    return centers[:, None, :] + radius * (np.cos(ang)[None, :, None] * t1[:, None, :]
                                           + np.sin(ang)[None, :, None] * t2[:, None, :])


_DISH_CACHE = {}


def _focusing_dish(m, d, target, radius, facet_radius):
    """Facets laid out on a disk perpendicular to d, lifted (along d) onto the plane
    through m normal to the mean bisector; each facet's normal reflects d from its
    own centroid onto ``target``. Returns [n_facets, 3, 3]."""
    key = (radius, facet_radius)
    if key not in _DISH_CACHE:
        _DISH_CACHE[key] = _dish_centroids(radius, facet_radius)
    c2 = _DISH_CACHE[key]
    u0 = target - m
    u0 /= np.linalg.norm(u0)
    n0 = u0 - d
    n0 = n0 / np.linalg.norm(n0) if np.linalg.norm(n0) > 1e-9 else np.cross(d, [0.0, 0.0, 1.0])
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(d, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(d, t1)
    c = m + c2[:, :1] * t1 + c2[:, 1:] * t2
    c = c - d * (((c - m) @ n0) / np.dot(d, n0))[:, None]
    u = target - c
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    n = u - d
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(nn > 1e-9, n / np.maximum(nn, 1e-30), n0)
    return _fact_triangles(c, n, facet_radius)


def build_scene(world, geom: GeometryConfig = GeometryConfig(), params: TraceParams = TraceParams()):
    """Write the world's atomic facts into geometry.

    specular: fact (s, r, o) is a mirror triangle at m = p_s + L d_r whose normal reflects
    d_r onto the line m -> p_o. transport: the same triangle carries a displacement
    that lands the ray just outside o's receiver, aimed at it.
    """
    pos = world.positions.astype(np.float64)
    rel = relation_directions(world.config.n_relations, geom.relation_seed).astype(np.float64)
    iv, ifc = icosphere(geom.receiver_radius)
    tris, kind, station, refl, disp, trel = [], [], [], [], [], []

    for s, p in enumerate(pos):
        for f in ifc:
            tris.append(p + iv[f])
            kind.append(KIND_RECEIVER)
            station.append(s)
            refl.append(0.0)
            disp.append(np.zeros(3))
            trel.append(-1)

    for (s, r), objs in sorted(world.facts.items()):
        d = rel[r]
        n_obj = len(objs)
        for i, o in enumerate(objs):
            m = pos[s] + (geom.fact_distance + i * geom.stack_gap) * d
            u = pos[o] - m
            u /= np.linalg.norm(u)
            R = 1.0 / (n_obj - i)          # equal energy split along the stack
            if params.hop_mode == SPECULAR:
                n = u - d
                nn = np.linalg.norm(n)
                n = n / nn if nn > 1e-9 else np.cross(d, [0.0, 0.0, 1.0])  # degenerate: o straight ahead
                dv = np.zeros(3)
            elif params.hop_mode == TRANSPORT:
                n = -d                     # aperture faces the incoming ray
                land = pos[o] - u * (3.0 * geom.receiver_radius)
                dv = land - m
            else:
                raise ValueError(params.hop_mode)
            if geom.dish_radius > 0 and params.hop_mode == SPECULAR:
                facets = _focusing_dish(m, d, pos[o], geom.dish_radius, geom.fact_radius)
            else:
                facets = [_fact_triangle(m, n, geom.fact_radius)]
            for tri in facets:
                tris.append(tri)
                kind.append(KIND_FACT)
                station.append(-1)
                refl.append(R)
                disp.append(dv)
                trel.append(r)

    params = replace(params,
                     emit_offset=float(1.5 * geom.receiver_radius),
                     emit_range=float(geom.fact_distance + geom.stack_gap + 2 * geom.fact_radius),
                     arrival_range=float(6.0 * geom.receiver_radius))
    return Scene(
        tri_vertices=np.asarray(tris, np.float32),
        tri_kind=np.asarray(kind, np.uint32),
        tri_station=np.asarray(station, np.int32),
        tri_reflectance=np.asarray(refl, np.float32),
        tri_displacement=np.asarray(disp, np.float32),
        station_pos=world.positions.astype(np.float32),
        relation_dir=rel.astype(np.float32),
        params=params,
        tri_relation=np.asarray(trel, np.int32),
    )


def shuffle_destinations(scene: Scene, seed: int) -> Scene:
    """Randomized-geometry control: move every station (receiver + emission point)
    to another station's location, leaving fact geometry where it was written."""
    rng = np.random.default_rng(seed)
    S = len(scene.station_pos)
    perm = rng.permutation(S)
    new_pos = scene.station_pos[perm]
    verts = scene.tri_vertices.copy()
    rec = scene.tri_kind == KIND_RECEIVER
    st = scene.tri_station[rec]
    verts[rec] += (new_pos[st] - scene.station_pos[st])[:, None, :]
    return replace(scene, tri_vertices=verts, station_pos=new_pos.astype(np.float32))


def perturb_stations(scene: Scene, sigma: float, seed: int) -> Scene:
    """Layout-noise stress test: jitter each station's receiver by N(0, sigma^2 I)
    after facts were written (fact geometry unchanged)."""
    rng = np.random.default_rng(seed)
    off = rng.normal(0, sigma, size=scene.station_pos.shape).astype(np.float32)
    verts = scene.tri_vertices.copy()
    rec = scene.tri_kind == KIND_RECEIVER
    verts[rec] += off[scene.tri_station[rec]][:, None, :]
    return replace(scene, tri_vertices=verts, station_pos=scene.station_pos + off)
