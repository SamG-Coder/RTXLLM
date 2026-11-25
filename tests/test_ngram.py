import numpy as np

from rtxllm.baselines.ngram import KneserNey, context_counts

RNG = np.random.default_rng(0)
TEXT = np.frombuffer(b"the cat sat on the mat. the dog sat on the log. " * 20, np.uint8).astype(np.int64)


def test_distribution_normalised_for_seen_and_unseen_contexts():
    for order in (1, 3, 5):
        m = KneserNey(order).fit([TEXT])
        for ctx in (b"the", b"zq#", b"at "):
            ctx = ctx[-(order - 1):] if order > 1 else b""
            tot = sum(m.prob(np.frombuffer(ctx + bytes([b]), np.uint8).astype(np.int64), start=len(ctx))[0]
                      for b in range(256))
            assert abs(tot - 1.0) < 1e-9, (order, ctx, tot)


def test_higher_order_helps_on_repetitive_text():
    tr, te = TEXT[:600], TEXT[600:]
    bpb = [KneserNey(n).fit([tr]).bits_per_byte([te], start=4) for n in (1, 2, 3, 5)]
    assert bpb[0] > bpb[1] > bpb[2] > bpb[3]


def test_no_cross_document_ngrams():
    a = np.frombuffer(b"aaaa", np.uint8).astype(np.int64)
    b = np.frombuffer(b"bbbb", np.uint8).astype(np.int64)
    m = KneserNey(2).fit([a, b])
    t = m.tables[2]
    assert (ord("a") * 256 + ord("b")) not in set(t["gk"].tolist())


def test_context_counts():
    tr = np.frombuffer(b"abcabcabd", np.uint8).astype(np.int64)
    ev = np.frombuffer(b"abcx", np.uint8).astype(np.int64)
    c = context_counts([tr], ev, 2, start=2)
    assert c.tolist() == [3, 2]   # "ab" is a context 3x, "bc" 2x


def test_layered_pencil_equals_kneser_ney_for_any_grid():
    from rtxllm.learning.byte_layers import LayeredBackoff, identity_grid, random_grid, similarity_grid
    tr, ev = TEXT[:700], TEXT[700:]
    for order in (2, 3, 4):
        kn = KneserNey(order).fit([tr])
        ref = kn.prob(ev, start=order - 1)
        for g in (identity_grid(), random_grid(1), similarity_grid([tr])):
            lb = LayeredBackoff(kn, g)
            _, p = lb.beam_bits(ev, order - 1, np.zeros((1, 2)), np.ones(1))
            assert np.allclose(p, ref, rtol=0, atol=1e-12)


def test_grids_are_permutations():
    from rtxllm.learning.byte_layers import random_grid, similarity_grid
    for g in (random_grid(0), similarity_grid([TEXT])):
        assert len({tuple(x) for x in g.tolist()}) == 256
        assert g.min() == 0 and g.max() == 15


def test_query_points_strictly_inside_cells():
    from rtxllm.learning.byte_layers import G, LayeredBackoff, identity_grid
    kn = KneserNey(4).fit([TEXT])
    lb = LayeredBackoff(kn, identity_grid())
    p = lb.query_points(TEXT, np.arange(3, len(TEXT)))
    for m in range(0, 4):
        frac = (p * G ** m) % 1.0
        assert np.all(frac > 0) and np.all(frac < 1)
        if m == 3:
            assert np.all(np.abs(frac[:, 0] - frac[:, 1]) > 0)   # never on the triangle diagonal
