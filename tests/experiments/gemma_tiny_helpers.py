"""A tiny fast tokenizer that behaves like Gemma's where BOS is concerned:
``tokenizer(text)`` prepends ``<bos>`` (a post-processor), the chat template
emits a literal ``<bos>`` that tokenises to the same id, and ``<eos>`` is a
separate token. Shared by the double-BOS tests of the retained corpus's
consumers and of the Dutch-sets builder."""
from __future__ import annotations

PAD, EOS, BOS, UNK = 0, 1, 2, 3
WORDS = [f"w{i}" for i in range(300)] + ["<start_of_turn>user", "<start_of_turn>model",
                                         "<end_of_turn>", "user", "model"]

CHAT_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }} "
                 "{{ m['content'] }} <end_of_turn> {% endfor %}"
                 "{% if add_generation_prompt %}<start_of_turn>model {% endif %}")


def gemma_like_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": PAD, "<eos>": EOS, "<bos>": BOS, "<unk>": UNK}
    vocab.update({w: i + 4 for i, w in enumerate(WORDS)})
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(
        single="<bos> $A", pair="<bos> $A <bos> $B", special_tokens=[("<bos>", BOS)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>",
                                  pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    tok.name_or_path = "fake/gemma-like"
    return tok
