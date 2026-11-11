"""``srspd_experiments.eval.trainer`` datasets for the recovery stream and monitors.

``MixedShardDataset``: round-robin in the given order, equal micro-batch
counts per language over a step, deterministic with its own seed and
independent of the trainer's generator. ``FixedSpanDataset``: every window
once, a span shorter than one block is one short window, ``evaluate`` reads it
to the end with a token-weighted mean. ``train``'s ``log_hook`` fires at the
log cadence. Tiny shards on disk, CPU, no network.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from srspd_experiments.eval import trainer as T  # noqa: E402

BLOCK = 8


def _shard(path, tokens):
    np.asarray(tokens, dtype=np.uint16).tofile(path)
    return str(path)


@pytest.fixture
def langs(tmp_path):
    """Three single-shard languages whose token values identify the language."""
    out = {}
    for base, lang in ((100, "eng"), (200, "deu"), (300, "nld")):
        toks = base + (np.arange(400) % 50)
        out[lang] = T.ShardDataset([_shard(tmp_path / f"{lang}.bin", toks)], BLOCK)
    return out


def _lang_of(x):
    return {1: "eng", 2: "deu", 3: "nld"}[int(x[0, 0]) // 100]


class TestMixedShardDataset:
    def test_round_robin_in_the_given_order(self, langs):
        ds = T.MixedShardDataset(langs, order=["nld", "eng", "deu"], seed=1)
        rng = np.random.default_rng(0)
        seen = [_lang_of(ds.batch(2, rng, "cpu")[0]) for _ in range(6)]
        assert seen == ["nld", "eng", "deu", "nld", "eng", "deu"]
        assert ds.next_lang() == "nld" and ds.n_langs == 3

    def test_default_order_is_sorted(self, langs):
        ds = T.MixedShardDataset(langs)
        assert ds.langs == ["deu", "eng", "nld"]

    def test_equal_counts_per_language_over_a_step(self, langs):
        ds = T.MixedShardDataset(langs, order=["eng", "deu", "nld"], seed=3)
        rng = np.random.default_rng(0)
        accum = 6                      # a multiple of the language count
        counts = {"eng": 0, "deu": 0, "nld": 0}
        for _ in range(accum):
            counts[_lang_of(ds.batch(4, rng, "cpu")[0])] += 1
        assert counts == {"eng": 2, "deu": 2, "nld": 2}

    def test_own_seed_is_deterministic_and_independent_of_the_callers_rng(self, langs):
        def stream(seed, caller_seed):
            ds = T.MixedShardDataset(langs, order=["eng", "deu", "nld"], seed=seed)
            rng = np.random.default_rng(caller_seed)
            return [ds.batch(3, rng, "cpu")[0].clone() for _ in range(6)]
        a = stream(7, 0)
        b = stream(7, 99)            # a different caller generator: same stream
        c = stream(8, 0)             # a different stream seed: different windows
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert any(not torch.equal(x, y) for x, y in zip(a, c))

    def test_without_seed_the_callers_rng_drives_sampling(self, langs):
        ds = T.MixedShardDataset(langs, order=["eng"])
        x1 = ds.batch(2, np.random.default_rng(5), "cpu")[0]
        x2 = T.ShardDataset.batch(langs["eng"], 2, np.random.default_rng(5), "cpu")[0]
        assert torch.equal(x1, x2)

    def test_bad_order_is_rejected(self, langs):
        with pytest.raises(ValueError):
            T.MixedShardDataset(langs, order=["eng", "fra"])
        with pytest.raises(ValueError):
            T.MixedShardDataset(langs, order=["eng", "eng"])
        with pytest.raises(ValueError):
            T.MixedShardDataset({})

    def test_x_y_are_shifted_by_one(self, langs):
        ds = T.MixedShardDataset(langs, order=["deu"], seed=0)
        x, y = ds.batch(2, np.random.default_rng(0), "cpu")
        assert torch.equal(x[:, 1:], y[:, :-1])


class TestFixedSpanDataset:
    def test_short_span_is_one_short_window(self, tmp_path):
        ds = T.FixedSpanDataset([_shard(tmp_path / "s.bin", range(5))], block_size=BLOCK)
        assert ds.windows == [(0, 5)] and ds.eval_batches == 1
        x, y = ds.batch(16, None, "cpu")
        assert x.shape == (1, 4) and y.shape == (1, 4)
        assert x.tolist() == [[0, 1, 2, 3]] and y.tolist() == [[1, 2, 3, 4]]

    def test_full_windows_cover_the_span_once(self, tmp_path):
        # 30 tokens, block 8: (30-1)//8 = 3 full windows; 5 trailing tokens dropped
        ds = T.FixedSpanDataset([_shard(tmp_path / "s.bin", range(30))], BLOCK, micro_batch=2)
        assert ds.windows == [(0, 9), (8, 17), (16, 25)]
        assert ds.eval_batches == 2
        x1, _ = ds.batch(2, None, "cpu")
        x2, _ = ds.batch(2, None, "cpu")
        assert x1.shape == (2, 8) and x2.shape == (1, 8)      # last batch is shorter
        assert x1[0].tolist() == list(range(0, 8)) and x2[0].tolist() == list(range(16, 24))
        ds.reset()
        assert torch.equal(ds.batch(2, None, "cpu")[0], x1)

    def test_max_tokens_slices_a_prefix_across_shards(self, tmp_path):
        a = _shard(tmp_path / "a.bin", range(0, 10))
        b = _shard(tmp_path / "b.bin", range(10, 20))
        ds = T.FixedSpanDataset([a, b], BLOCK, max_tokens=13)
        assert ds.total == 13 and ds.tokens.tolist() == list(range(13))
        assert ds.windows == [(0, 9)]

    def test_too_short(self, tmp_path):
        with pytest.raises(ValueError):
            T.FixedSpanDataset([_shard(tmp_path / "s.bin", [1])], BLOCK)
        with pytest.raises(ValueError):
            T.FixedSpanDataset([], BLOCK)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=400, n_positions=BLOCK, n_embd=16,
                                  n_layer=1, n_head=2)
    return transformers.GPT2LMHeadModel(cfg)


class TestEvaluate:
    def test_fixed_span_is_read_to_the_end_and_token_weighted(self, tmp_path, tiny):
        toks = np.arange(30) % 7
        ds = T.FixedSpanDataset([_shard(tmp_path / "s.bin", toks)], BLOCK, micro_batch=2)
        cfg = T.TrainConfig(total_tokens=0, micro_batch=2, block_size=BLOCK,
                            eval_batches=99, compile=False)
        got = T.evaluate(tiny, {"m": ds}, cfg, "cpu")["m"]
        # by hand: every window once, token-weighted
        tiny.eval()
        num = den = 0.0
        with torch.no_grad():
            for lo, hi in ds.windows:
                c = torch.as_tensor(toks[lo:hi].astype(np.int64))[None]
                num += T.lm_loss(tiny, c[:, :-1], c[:, 1:]).item() * (hi - lo - 1)
                den += hi - lo - 1
        assert got == pytest.approx(num / den, rel=1e-5)

    def test_names_are_the_callers(self, tmp_path, tiny):
        toks = np.arange(60) % 7
        sd = T.ShardDataset([_shard(tmp_path / "s.bin", toks)], BLOCK)
        cfg = T.TrainConfig(total_tokens=0, micro_batch=2, block_size=BLOCK,
                            eval_batches=2, compile=False)
        out = T.evaluate(tiny, {"heldout_eng": sd, "forget_nld": sd}, cfg, "cpu")
        assert set(out) == {"heldout_eng", "forget_nld"}
        assert out["heldout_eng"] == pytest.approx(out["forget_nld"])   # same fixed seed


def test_train_log_hook_fires_at_the_log_cadence(tmp_path, tiny, langs):
    cfg = T.TrainConfig(total_tokens=6 * 2 * BLOCK * 3, batch_tokens=2 * BLOCK * 3,
                        micro_batch=2, block_size=BLOCK, eval_every_tokens=10 ** 9,
                        eval_batches=1, compile=False, log_every_steps=2, lr=1e-3,
                        warmup_tokens=1)
    stream = T.MixedShardDataset(langs, order=["eng", "deu", "nld"], seed=0)
    calls = []
    T.train(tiny, stream, {"eng": langs["eng"]}, cfg, "cpu",
            csv_path=str(tmp_path / "t.csv"), log=lambda *_: None,
            log_hook=lambda step, tokens, lr, ce: calls.append((step, tokens, lr, ce)))
    assert [c[0] for c in calls] == [2, 4, 6]
    assert calls[0][1] == 2 * cfg.batch_tokens and all(c[2] > 0 for c in calls)
    assert all(np.isfinite(c[3]) for c in calls)
