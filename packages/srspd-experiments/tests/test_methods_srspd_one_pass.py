"""The SRSPD method runner runs ONE set of Gram passes per run, however many
``(dose, rank rule)`` settings the run exports.

A shipped multi-rule recipe run (the rank-truncation sweep of the WMDP recipe:
full rank plus three ``lam_rel`` values) is run on a tiny random Mistral with
mocked corpora, a second dose and ``"erank"`` added, under a Gram budget small
enough for several passes. The Gram passes are counted twice -- calls into the
pipeline's pass loop and the ``plan`` / ``pass_start`` progress events -- and
every export the recipe names (``srspd_mpt<mpt>[_lam<l>|_erank]``) must be
written."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from conftest import isolate_corpora  # noqa: E402

RESULT, RUN = "zephyr_wmdp", "srspd_f16_lamsweep_mpt0.003"


def _mistral():
    torch.manual_seed(0)
    cfg = transformers.MistralConfig(
        vocab_size=256, hidden_size=16, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return transformers.MistralForCausalLM(cfg).to(torch.bfloat16).eval()


class _WordTok:
    """Deterministic word-level tokenizer: BOS + crc32(word) per word."""
    bos_token_id = 1
    name_or_path = "fake/wordtok"

    def __call__(self, text, add_special_tokens=True, **_kw):
        import zlib
        ids = [(zlib.crc32(w.encode()) % 240) + 10 for w in text.split()]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}


def _fake_docs(bucket):
    for i in range(int(bucket.get("skip_docs", 0)), 10 ** 6):
        yield i, " ".join(f"{bucket['name']}d{i}w{j}" for j in range(50))


def _fake_corpus_texts(corpus, *, min_len=50, max_rows=None):
    n = {"wikitext": 40}.get(corpus, 12)
    return [" ".join(f"{corpus[:4]}{i}w{(j * 7 + i) % 97}" for j in range(60))
            for i in range(n)][:max_rows]


def test_multi_rule_recipe_run_does_one_set_of_gram_passes(tmp_path, monkeypatch):
    import srspd.api as api
    from srspd.capture.hf import HFHookSource
    from recipe_configs import config
    from srspd_experiments.data import corpora as PC
    from srspd_experiments.data import retained_corpus as PR
    from srspd_experiments.methods import srspd as runner
    from srspd_experiments.recipes import catalog

    isolate_corpora(monkeypatch, drop_local=True)
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(PC, "load_corpus_texts", _fake_corpus_texts)
    monkeypatch.setattr(PR, "iter_documents", _fake_docs)

    (run,) = [r for r in catalog.load_recipe(RESULT)["runs"] if r["id"] == RUN]
    cfg = config(RESULT, RUN)
    assert len(cfg["srspd"]["rank_rule"]) > 1, "the recipe run must carry several rules"
    d = cfg["data"]
    d["target"].update(max_rows=2, max_length=48, heldout_rows=2)
    d.update(retain_max_length=64, heldout_mix_tokens=0)
    d["retained"]["chunk_tokens"] = 128
    d["retained"]["buckets"] = [dict(b, tokens=640) for b in d["retained"]["buckets"]]
    mpts = sorted(set(cfg["srspd"]["mu_per_token"]) | {0.03})
    rules = list(cfg["srspd"]["rank_rule"]) + ["erank"]
    cfg["srspd"].update(mu_per_token=mpts, rank_rule=rules, gram_budget_gib=20000 / 2 ** 30)
    cfg.update(device_map="cpu")

    calls, accum, events = [], [], []
    real_passes, real_accum = api.run_gram_passes, HFHookSource.accumulate

    def passes_spy(*a, **kw):
        calls.append(1)
        return real_passes(*a, **kw)

    def accum_spy(self, *a, **kw):
        accum.append(kw.get("label"))
        return real_accum(self, *a, **kw)

    monkeypatch.setattr(api, "run_gram_passes", passes_spy)
    monkeypatch.setattr(HFHookSource, "accumulate", accum_spy)

    model = _mistral()
    pristine = {n: p.detach().clone() for n, p in model.named_parameters()}
    m = runner.run(cfg, model=model, tokenizer=_WordTok(), out_root=tmp_path / "models",
                   log_root=tmp_path / "logs", progress=lambda e, i: events.append(e))

    n_pass = len(m["solve"]["pass_plan"])
    assert n_pass > 1                                      # the budget forces several passes
    assert calls == [1] and events.count("plan") == 1
    assert events.count("pass_start") == n_pass and len(accum) == n_pass
    want = {runner.export_label(mp, r) for mp in mpts for r in [None] + rules}
    assert set(m["exports"]) == want and len(want) == len(mpts) * (1 + len(rules))
    assert {x["checkpoint_subdir"] for x in run["exports"]} <= want
    for lab in want:
        assert (tmp_path / "models" / lab / "config.json").is_file(), lab
    assert m["solve"]["rank_rules"] == rules
    assert len(m["solve"]["settings"]) == len(want)
    # the full-rank and truncated edits differ; every edit was restored
    full = m["exports"][runner.export_label(mpts[0])]["edit"]["delta_fro_sum"]
    trunc = m["exports"][runner.export_label(mpts[0], "erank")]["edit"]["delta_fro_sum"]
    assert full != trunc
    assert all(torch.equal(p.detach(), pristine[n]) for n, p in model.named_parameters())
