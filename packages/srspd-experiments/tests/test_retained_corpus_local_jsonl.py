"""The ``local_jsonl`` bucket kind of the retained corpus: a data-root JSONL
file as a retained-corpus bucket (the GPT-2 unlearning sets), with the file's
content sha256 in the cache key and hub buckets left byte-identical. CPU, no
network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gpt2_tiny_helpers import tiny_tokenizer, write_sets
from srspd_experiments.data import retained_corpus as rcorp
from srspd_experiments.data import spec_path

SETS = "datasets/gpt2_unlearning/gpt2_unlearning_1m"


def _local_bucket(**over):
    b = {"name": "contrastive_eng", "kind": "local_jsonl",
         "path": f"{SETS}/contrastive_eng.jsonl", "text_field": "text", "tokens": 64}
    b.update(over)
    return b


def test_local_bucket_resolves_without_hub_fields():
    spec = rcorp.resolve_spec([_local_bucket()])
    assert spec[0]["kind"] == "local_jsonl" and spec[0]["revision"] is None
    assert spec[0]["dataset"] is None and spec[0]["path"] == f"{SETS}/contrastive_eng.jsonl"
    with pytest.raises(ValueError, match="path"):
        rcorp.resolve_spec([_local_bucket(path=None)])
    with pytest.raises(ValueError, match="do not apply"):
        rcorp.resolve_spec([_local_bucket(revision="abc")])
    with pytest.raises(ValueError, match="kind"):
        rcorp.resolve_spec([_local_bucket(kind="s3")])
    # hub buckets still need their revision, and carry no `kind` key (their
    # resolved dict -- and spec_key -- is unchanged)
    blk = yaml.safe_load(Path(spec_path("dutch_sets_1k")).read_text())["retained_corpus"]
    hub = rcorp.resolve_spec(blk["buckets"])
    assert all("kind" not in b for b in hub)


def test_local_bucket_streams_the_file_and_keys_the_cache_on_its_content(tmp_path, monkeypatch):
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    d = write_sets(tmp_path, n_target=1, n_contrastive=6, n_heldout=1, words=10)
    spec = rcorp.resolve_spec([_local_bucket(tokens=25)])
    docs = list(rcorp.iter_documents(spec[0]))
    assert [i for i, _ in docs] == list(range(6))
    assert list(rcorp.iter_documents(dict(spec[0], skip_docs=4)))[0][0] == 4
    tok = tiny_tokenizer()
    ids, man = rcorp.build_retained_tokens(tok, spec, chunk_tokens=5, add_bos=True)
    b = man["buckets"][0]
    assert b["tokens_taken"] == 25 and b["kind"] == "local_jsonl" and len(b["path_sha256"]) == 64
    assert b["doc_index_start"] == 0 and b["doc_index_end"] == 2      # 11 tokens per doc incl. BOS
    assert tuple(ids.shape) == (5, 5)
    key_a = rcorp.spec_key(tok, spec, chunk_tokens=5)
    # the cache is keyed on the file's CONTENT, not just its path
    with open(f"{d}/contrastive_eng.jsonl", "a") as f:
        f.write(json.dumps({"text": "w1 w2 w3"}) + "\n")
    assert rcorp.spec_key(tok, spec, chunk_tokens=5) != key_a
    # a missing file fails loudly
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list(rcorp.iter_documents(rcorp.resolve_spec([_local_bucket(path="datasets/none.jsonl")])[0]))
