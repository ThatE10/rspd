"""Tiny random-weight models of every verified family, built from configs (no
download), and a word-level tokenizer. Shared by the capture / family / CLI /
parity tests. Every builder seeds torch, so two calls give identical weights."""
from __future__ import annotations

import torch

VOCAB = 128


def _t():
    import transformers
    return transformers


def gpt2(seed=0, n_layer=3, n_embd=16, n_positions=64, vocab=VOCAB):
    T = _t()
    torch.manual_seed(seed)
    cfg = T.GPT2Config(vocab_size=vocab, n_embd=n_embd, n_layer=n_layer, n_head=2,
                       n_positions=n_positions, bos_token_id=0, eos_token_id=0, pad_token_id=0)
    return T.GPT2LMHeadModel(cfg).eval()


def mistral(seed=0, dtype=torch.float32, vocab=VOCAB):
    T = _t()
    torch.manual_seed(seed)
    cfg = T.MistralConfig(vocab_size=vocab, hidden_size=16, intermediate_size=40,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                          max_position_embeddings=4096, bos_token_id=1, eos_token_id=2,
                          pad_token_id=0)
    return T.MistralForCausalLM(cfg).to(dtype).eval()


def llama(seed=0, dtype=torch.float32):
    T = _t()
    torch.manual_seed(seed)
    cfg = T.LlamaConfig(vocab_size=VOCAB, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        max_position_embeddings=256)
    return T.LlamaForCausalLM(cfg).to(dtype).eval()


def qwen3(seed=0, dtype=torch.float32):
    T = _t()
    torch.manual_seed(seed)
    cfg = T.Qwen3Config(vocab_size=VOCAB, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=256)
    return T.Qwen3ForCausalLM(cfg).to(dtype).eval()


def _gemma3_text_kw():
    return dict(vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                max_position_embeddings=256, sliding_window=8)


def gemma3_text(seed=0, dtype=torch.float32):
    T = _t()
    torch.manual_seed(seed)
    return T.Gemma3ForCausalLM(T.Gemma3TextConfig(**_gemma3_text_kw())).to(dtype).eval()


def gemma3_multimodal(seed=0, dtype=torch.float32):
    T = _t()
    torch.manual_seed(seed)
    cfg = T.Gemma3Config(text_config=_gemma3_text_kw(),
                         vision_config=dict(hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=1, num_attention_heads=2,
                                            image_size=28, patch_size=14),
                         mm_tokens_per_image=4)
    return T.Gemma3ForConditionalGeneration(cfg).to(dtype).eval()


def _gemma4_text_cfg(n_kv_shared):
    T = _t()
    return T.Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=16,
        hidden_size_per_layer_input=4, vocab_size_per_layer_input=VOCAB,
        num_kv_shared_layers=n_kv_shared, sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention",
                     "full_attention"])


def gemma4_text(seed=0, dtype=torch.float32, n_kv_shared=2):
    T = _t()
    torch.manual_seed(seed)
    return T.Gemma4ForCausalLM(_gemma4_text_cfg(n_kv_shared)).to(dtype).eval()


def gemma4_multimodal(seed=0, dtype=torch.float32, n_kv_shared=2):
    T = _t()
    torch.manual_seed(seed)
    vc = T.Gemma4VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                              num_attention_heads=2, num_key_value_heads=2, head_dim=8)
    cfg = T.Gemma4Config(text_config=_gemma4_text_cfg(n_kv_shared).to_dict(),
                         vision_config=vc.to_dict())
    return T.Gemma4ForConditionalGeneration(cfg).to(dtype).eval()


def word_tokenizer(n_words=VOCAB - 4):
    """A fast word-level tokenizer: ``<|endoftext|>`` = 0 (bos/eos/pad), words
    ``w0..w{n-1}``; adds no special tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"<|endoftext|>": 0, "[UNK]": 1, "[A]": 2, "[B]": 3}
    vocab.update({f"w{i}": i + 4 for i in range(n_words)})
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                                  bos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                  unk_token="[UNK]")


def token_sets(seed=1, n_target=3, target_len=24, n_retained=6, retained_len=32, lo=4,
               hi=VOCAB):
    """Disjoint-vocabulary target / retained id sets (target from the lower half)."""
    g = torch.Generator().manual_seed(seed)
    mid = (lo + hi) // 2
    tgt = [torch.randint(lo, mid, (target_len,), generator=g) for _ in range(n_target)]
    ret = torch.randint(mid, hi, (n_retained, retained_len), generator=g)
    return tgt, ret


FAMILIES = {
    "gpt2": gpt2, "mistral": mistral, "llama": llama, "qwen3": qwen3,
    "gemma3_text": gemma3_text, "gemma3_multimodal": gemma3_multimodal,
    "gemma4_text": gemma4_text, "gemma4_multimodal": gemma4_multimodal,
}
