"""The authors'-code runners, without the authors' code: the argument sets they
hand upstream, the staged corpus formats, and the pinned-commit checkout (a
local git repository stands in for the remote). CPU, no network."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from srspd_experiments.methods import upstream_kuda as UK
from srspd_experiments.methods import upstream_kuda_muse as UM
from srspd_experiments.methods import upstream_simnpo as US


def test_kuda_args_are_the_published_wmdp_script():
    a = UK.build_args("/out")
    assert (a.beta, a.tau, a.lr, a.batch_size, a.max_num_batches, a.epoch) == \
        (2.0, 2e-3, 1e-5, 4, 150, 1)
    assert (a.layer_id, a.layer_ids, a.module_name) == (9, [6, 7, 8, 9], "mlp.down_proj")
    assert a.forget_corpora == ["bio-forget-corpus", "cyber-forget-corpus"]
    assert a.retain_corpora == ["wikitext", "wikitext"] and a.alpha == [1.0, 1.0]
    assert (a.min_len, a.seed, a.sample_p, a.output_dir) == (0, 42, 100, "/out")
    assert UK.build_args("/o", "org/m").model_name_or_path == "org/m"
    assert len(UK.KUDA_SHA) == 40


def test_kuda_stages_bio_as_json_and_cyber_as_raw_lines(tmp_path, monkeypatch):
    data = {"bio-forget-corpus": ["a b", "c\nd"], "cyber-forget-corpus": ["x\n y", "z"]}
    monkeypatch.setattr(UK.C, "load_corpus_texts", lambda c, min_len=50: data[c])
    UK.stage_forget_corpora(str(tmp_path))
    bio = [json.loads(l)["text"] for l in open(tmp_path / "bio-forget-corpus.jsonl")]
    assert bio == ["a b", "c\nd"]
    assert open(tmp_path / "cyber-forget-corpus.jsonl").read().splitlines() == ["x y", "z"]


def test_simnpo_kwargs_are_the_published_json():
    k = US.build_kwargs()
    assert (k["unlearn_method"], k["beta"], k["gamma"], k["lr"], k["max_steps"]) == \
        ("SimNPO+FT", 0.1, 1.0, 9.5e-06, 500)
    assert (k["gradient_accumulation_steps"], k["batch_size"], k["weight_decay"], k["seed"]) == \
        (4, 1, 0.0, 0)
    assert k["forget_dataset_name"] == "WMDPBio" and k["dataset_seed"] == 42
    assert US.build_kwargs("org/m")["model_name"] == "org/m"
    assert US.UPSTREAM_SHA == "3083017cb317753725f35cc404c2d5aede1242ef"


def test_simnpo_stages_every_bio_row_in_order(tmp_path, monkeypatch):
    rows = ["short", "x" * 60, "y" * 70]
    monkeypatch.setattr(US.C, "load_corpus_texts_all", lambda c: rows)
    US.stage_bio_forget(str(tmp_path))
    p = tmp_path / "files" / "data" / "bio_remove_dataset.jsonl"
    assert [json.loads(l)["text"] for l in open(p)] == rows


def test_muse_argv_is_the_published_script():
    assert UM.argv("news") == ["python3", "-u", "./baselines/unlearn.py", "--algo", "kuda",
                               "--corpus", "news", "--beta", "2.0", "--tau", "4e-3", "--lr",
                               "5e-5", "--epochs", "3", "--layer_id", "7", "--layer_ids", "4",
                               "5", "6", "7", "--out_dir", "./outputs/muse/news_kuda"]
    assert UM.argv("books", python="/py")[:1] == ["/py"]
    assert UM.HPARAMS["books"] == ["--beta", "0.1", "--tau", "0.08", "--lr", "1e-5",
                                   "--epochs", "5"]
    with pytest.raises(ValueError):
        UM.argv("tofu")


def test_muse_epoch_checkpoints_are_collected_without_optimizer_state(tmp_path):
    out = tmp_path / "outputs"
    for step, ep in ((203, 1.0), (406, 2.0)):
        d = out / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text(json.dumps({"epoch": ep}))
        (d / "model.safetensors").write_text("w")
        (d / "optimizer.pt").write_text("o")
    rec = UM.collect_epochs(str(out), str(tmp_path / "copies"), "k")
    assert [r["epoch"] for r in rec] == [1, 2]
    assert sorted(os.listdir(tmp_path / "copies" / "k_ep2")) == ["model.safetensors",
                                                                 "trainer_state.json"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize("mod", [UK, US, UM])
def test_ensure_clone_checks_out_the_pinned_commit_and_refuses_another(tmp_path, mod):
    src = tmp_path / "src"
    src.mkdir()
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    run = lambda *a: subprocess.run(["git", "-C", str(src), *a], check=True, env=env,  # noqa: E731
                                    capture_output=True)
    run("init", "-q")
    (src / "f").write_text("1")
    run("add", "f")
    run("commit", "-qm", "one")
    first = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"]).decode().strip()
    (src / "f").write_text("2")
    run("commit", "-qam", "two")
    dst = tmp_path / "clone"
    assert mod.ensure_clone(str(dst), first, url=str(src)) == first
    assert (dst / "f").read_text() == "1"
    with pytest.raises(subprocess.CalledProcessError):
        mod.ensure_clone(str(dst), "0" * 40, url=str(src))
