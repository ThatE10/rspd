"""Shared fixtures for the GPT-2 tests.

A tiny random ``GPT2LMHeadModel`` (``transformer.h`` blocks with ``Conv1D``
projections, exactly the module shape of the trilingual 124M), a real
``PreTrainedTokenizerFast`` over a word-level vocabulary (so ``save_pretrained``
writes ``tokenizer.json`` only, like the trilingual BPE), and writers for the
``local_jsonl`` unlearning sets under a temporary data root. No network.
"""
from __future__ import annotations

import json
import os

import torch

VOCAB_WORDS = [f"w{i}" for i in range(200)]
N_EMBD, N_LAYER, N_POS = 16, 3, 64


def tiny_gpt2(seed=0, n_layer=N_LAYER):
    import transformers
    torch.manual_seed(seed)
    cfg = transformers.GPT2Config(vocab_size=len(VOCAB_WORDS) + 4, n_embd=N_EMBD,
                                  n_layer=n_layer, n_head=2, n_positions=N_POS,
                                  bos_token_id=0, eos_token_id=0, pad_token_id=0)
    return transformers.GPT2LMHeadModel(cfg).eval()


def tiny_tokenizer():
    """A fast word-level tokenizer: ``<|endoftext|>`` = 0 is bos/eos/pad (GPT-2's
    convention), words ``w0..w199``; no special tokens are added."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"<|endoftext|>": 0, "[UNK]": 1, "[A]": 2, "[B]": 3}
    vocab.update({w: i + 4 for i, w in enumerate(VOCAB_WORDS)})
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                                  bos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                  unk_token="[UNK]")


def save_checkpoint(data_root, rel="models/gpt2/gpt2_trilingual_pretrain/checkpoint"):
    """A tiny GPT-2 + fast tokenizer at a data-root checkpoint reference;
    returns ``(rel, abs)``."""
    path = os.path.join(str(data_root), rel)
    os.makedirs(path, exist_ok=True)
    tiny_gpt2().save_pretrained(path)
    tiny_tokenizer().save_pretrained(path)
    assert os.path.exists(os.path.join(path, "tokenizer.json"))
    return rel, path


def doc_text(tag, i, words):
    return " ".join(f"w{(j * 7 + i * 3 + hash(tag) % 13) % 200}" for j in range(words))


def write_sets(data_root, *, n_target=4, n_contrastive=30, n_heldout=12, words=40,
               sets_name="gpt2_unlearning_1m"):
    """The six ``<role>_<lang>.jsonl`` files of the unlearning sets, under
    ``datasets/gpt2_unlearning/<sets_name>/``; returns the directory."""
    d = os.path.join(str(data_root), "datasets", "gpt2_unlearning", sets_name)
    os.makedirs(d, exist_ok=True)
    roles = {"target_nld": n_target, "contrastive_eng": n_contrastive,
             "contrastive_deu": n_contrastive, "heldout_eng": n_heldout,
             "heldout_deu": n_heldout, "heldout_nld": n_heldout}
    for role, n in roles.items():
        with open(os.path.join(d, f"{role}.jsonl"), "w") as f:
            for i in range(n):
                f.write(json.dumps({"id": f"{role}-{i}", "title": f"{role} {i}",
                                    "lang": role.split("_")[1], "role": role.split("_")[0],
                                    "text": doc_text(role, i, words),
                                    "is_target_counterpart": False}) + "\n")
    return d


def corpora_block(sets_dir_rel="datasets/gpt2_unlearning/gpt2_unlearning_1m"):
    p = lambda r: f"{sets_dir_rel}/{r}.jsonl"
    return {
        "target_nld": {"kind": "local_jsonl", "path": p("target_nld"), "topic": "target"},
        "contrastive": {"kind": "local_jsonl",
                        "path": [p("contrastive_eng"), p("contrastive_deu")], "topic": "retain"},
        "heldout_nld": {"kind": "local_jsonl", "path": p("heldout_nld"), "topic": "heldout"},
        "heldout_eng": {"kind": "local_jsonl", "path": p("heldout_eng"), "topic": "heldout"},
        "heldout_deu": {"kind": "local_jsonl", "path": p("heldout_deu"), "topic": "heldout"},
    }
