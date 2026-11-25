"""Post-hoc, validation-only probe for #83 (run after the sealed test): does nearest-hit occlusion or ray count limit the full-disc beam?

gamma = transmission per dish crossed (0 = nearest-hit only, as RTX closest-hit; 1 = every dish
on the ray counts equally). All-hit lists come from an exact CPU Moller-Trumbore over broadphase
candidates; gamma = 0 is checked against the RTX closest-hit result.
"""
import sys
from pathlib import Path
import numpy as np
import importlib.util
R = Path(sys.argv[1]); sys.path.insert(0, str(R / "src"))
_spec = importlib.util.spec_from_file_location("e12c", R / "experiments/e12c_full_beam/run.py")
c = importlib.util.module_from_spec(_spec); sys.modules["e12c"] = c; _spec.loader.exec_module(c)
e12 = c.e12
from rtxllm.cpu.tracer import mt_pairs
from word_beam import WordRTX, basis, open_word_kernel

tr, va, te = e12.tokens(Path(sys.argv[2])); m = e12.Model(tr)
ids, Xf = e12.force_word_layout(m); vp, vn = m.enc(va)[:-1], m.enc(va)[1:]
dev, ker = open_word_kernel(); pos, radii, scene = e12.scene_for(m, ids, Xf); gpu = WordRTX(dev, ker, scene)
sigma, lam = 1.0, 0.2
t1, t2 = basis(e12.D)

# ray count probe on RTX
for rays in (512, 2048):
    old = e12.RAYS; e12.RAYS = rays
    q, has, st = c.disc_beam(m, ids, pos, gpu, scene, sigma)
    e12.RAYS = old
    print(f"rays {rays}: ppl {e12.score(m, ids, q, has, lam, vp, vn)['ppl']:.3f} parity {st['cpu_parity']}", flush=True)

# all-hit lists on CPU
width = 2.5 * sigma; tmax = 2 * width + e12.FACT_DISTANCE + 0.5
O, rr = c.disc_origins(pos, ids, width, e12.RAYS)
O = O.reshape(len(ids), e12.RAYS, 3)
cent = pos[ids] + e12.FACT_DISTANCE * e12.D
C2 = np.stack([cent @ t1, cent @ t2], 1)
tri = scene.tris; v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
Dv = e12.D.astype(np.float32)
prof = np.exp(-rr ** 2 / (2 * sigma ** 2))
lists = []
agree = 0
gq = gpu.hits(O.reshape(-1, 3), e12.D, tmax)[0].reshape(len(ids), e12.RAYS)
for k in range(len(ids)):
    P = np.stack([O[k] @ t1, O[k] @ t2], 1)
    ri, dj = np.nonzero(((P[:, None] - C2[None]) ** 2).sum(-1) <= (radii[None] * 1.0001) ** 2)
    ri6 = np.repeat(ri, 6); ti = (np.repeat(dj * 6, 6) + np.tile(np.arange(6), len(dj)))
    t = mt_pairs(v0, e1, e2, O[k].astype(np.float32), np.broadcast_to(Dv, (e12.RAYS, 3)).astype(np.float32),
                 ri6, ti, np.full(e12.RAYS, tmax, np.float32), np.float32(1e-3))
    ok = np.isfinite(t)
    r_, d_, t_ = ri6[ok], ti[ok] // 6, t[ok]
    # unique (ray, dish) keep min t
    key = r_ * len(ids) + d_
    order = np.lexsort((t_, key)); key, r_, d_, t_ = key[order], r_[order], d_[order], t_[order]
    first = np.r_[True, key[1:] != key[:-1]]
    r_, d_, t_ = r_[first], d_[first], t_[first]
    o2 = np.lexsort((t_, r_)); r_, d_, t_ = r_[o2], d_[o2], t_[o2]
    rank = np.arange(len(r_)) - np.searchsorted(r_, r_)
    nearest = np.full(e12.RAYS, -1); nearest[r_[rank == 0]] = d_[rank == 0]
    agree += int((nearest == gq[k]).sum())
    lists.append((r_, d_, rank))
print(f"CPU nearest vs RTX agreement: {agree / gq.size:.5f}", flush=True)
for gamma in (0.0, 0.25, 0.5, 1.0):
    q = np.zeros((len(ids), m.V))
    for k, (r_, d_, rank) in enumerate(lists):
        w = prof[r_] * gamma ** rank if gamma > 0 else prof[r_] * (rank == 0)
        if len(r_):
            np.add.at(q[k], slice(None), 0)
            q[k] = (w[:, None] * m.Pml[ids[d_]]).sum(0)
    tot = q.sum(1, keepdims=True); has = tot[:, 0] > 0; q = np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)
    for l in (0.1, 0.2, 0.3):
        s = e12.score(m, ids, q, has, l, vp, vn)
        print(f"gamma {gamma} lam {l}: ppl {s['ppl']:.3f} unseen {s['ppl_unseen']:.0f} seen {s['ppl_seen']:.2f}", flush=True)
