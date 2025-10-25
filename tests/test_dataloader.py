"""Best-fit packing invariants and exact resume of the loader state."""

import numpy as np

from pagoda.data.build import ShardWriter
from pagoda.data.render import split_long
from pagoda.dataloader import PackedLoader, StagedBatches

BOS = 1


def make_shards(tmp_path, n_docs=3000, seed=0, name="s"):
    rng = np.random.default_rng(seed)
    w = ShardWriter(tmp_path / name, BOS)
    total = 0
    for i in range(n_docs):
        n = int(rng.integers(5, 1500))
        doc = [BOS] + list(rng.integers(2, 30000, n))
        for piece in split_long(doc, 33, BOS):
            w.add(np.array(piece, dtype=np.uint16))
            total += len(piece)
    w.close({})
    return tmp_path / name, total


def test_rows_start_with_bos_and_keep_every_token(tmp_path):
    d, total = make_shards(tmp_path, n_docs=400)
    loader = PackedLoader(d, seq_len=32, seed=3, buffer_docs=50)
    pulled = []
    orig = loader.stream.next

    def counting_next():
        ref = orig()
        pulled.append(ref)
        return ref

    loader.stream.next = counting_next
    rows = loader.next_rows(200)
    assert rows.shape == (200, 33)
    assert (rows[:, 0] == BOS).all()
    toks = loader.shards.tokens
    pulled_non_bos = sum(int((toks[s][a:b] != BOS).sum()) for s, a, b in pulled)
    buffered_non_bos = sum(int((toks[s][a:b] != BOS).sum()) for (_, _, s, a, b, _) in loader._buf)
    assert int((rows != BOS).sum()) + buffered_non_bos == pulled_non_bos  # no token is dropped


def test_state_roundtrip_reproduces_batches(tmp_path):
    d, _ = make_shards(tmp_path)
    a = PackedLoader(d, seq_len=32, seed=5, buffer_docs=100)
    a.next_rows(64)
    state = a.state_dict()
    expect = a.next_rows(64)
    b = PackedLoader(d, seq_len=32, seed=5, buffer_docs=100)
    b.load_state_dict(state)
    assert np.array_equal(b.next_rows(64), expect)


def test_epochs_wrap(tmp_path):
    d, total = make_shards(tmp_path, n_docs=50)
    loader = PackedLoader(d, seq_len=32, seed=1, buffer_docs=10)
    for _ in range(100):
        loader.next_rows(16)
    assert loader.epoch >= 1


def test_staged_transition(tmp_path):
    d1, _ = make_shards(tmp_path, seed=1, name="a")
    d2, _ = make_shards(tmp_path, seed=2, name="b")
    s = StagedBatches(d1, d2, micro_batch=8, seq_len=32, switch_step=10, transition_steps=4, seed=0)
    assert s.phase(5) == "stage1" and s.phase(6) == "transition" and s.phase(9) == "transition"
    assert s.phase(10) == "stage2"
    assert s.batch(7).shape == (8, 33)
