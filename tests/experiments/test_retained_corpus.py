"""Unit tests for ``rspd.experiments.data.retained_corpus``.

CPU, synthetic, offline. ``iter_documents`` is monkeypatched throughout, so no
network call is made.

What is actually being pinned is **reproducibility**. The corpus is an input to
every number a retained set produces, and it is assembled from streams that
can change upstream. So the properties under test are: the same spec gives a
byte-identical corpus, a changed spec gives a different hash, and the budget is
hit exactly rather than approximately.
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

import pytest
import torch

from rspd.experiments.data import retained_corpus as rc
from rspd.experiments.data import spec_path


class FakeTok:
    """A whitespace tokenizer with a BOS and a chat template.

    Deterministic and dependency-free: token id is ``crc32`` of the word, not
    ``hash()``. Python salts string hashing per process, so a ``hash()``-based
    id would make this fixture reproducible only *within* one interpreter — the
    determinism assertions below would still pass, while quietly testing
    something weaker than they claim.
    """

    name_or_path = "fake/tok"
    bos_token_id = 2

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [(zlib.crc32(w.encode("utf-8")) % 1000) + 10 for w in text.split()]
        return {"input_ids": ids}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "<start> user " + messages[0]["content"] + " <end> model"


def _docs(n, words=50, prefix="doc"):
    return [(i, " ".join(f"{prefix}{i}w{j}" for j in range(words))) for i in range(n)]


def _spec(tokens=200, name="a", revision="rev1", skip=0):
    return rc.resolve_spec([{
        "name": name, "dataset": "fake/ds", "config": "cfg", "split": "train",
        "revision": revision, "text_field": "text", "tokens": tokens,
        "skip_docs": skip,
    }])


@pytest.fixture
def fake_stream(monkeypatch):
    """Install a deterministic document stream in place of the HF one."""
    def install(docs):
        def fake_iter(bucket):
            for i, text in docs:
                if i < int(bucket["skip_docs"]):
                    continue
                yield i, text
        monkeypatch.setattr(rc, "iter_documents", fake_iter)
    return install


class TestResolveSpec:
    def test_requires_revision(self):
        """The field people omit, and the one that decides whether the corpus
        can be rebuilt later."""
        with pytest.raises(ValueError, match="revision"):
            rc.resolve_spec([{"name": "a", "dataset": "d", "split": "train",
                              "text_field": "text", "tokens": 10}])

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValueError, match="unknown fields"):
            rc.resolve_spec([{"name": "a", "dataset": "d", "split": "train",
                              "revision": "r", "text_field": "text", "tokens": 10,
                              "shuffle": True}])

    def test_rejects_duplicate_names(self):
        b = {"dataset": "d", "split": "train", "revision": "r",
             "text_field": "text", "tokens": 10}
        with pytest.raises(ValueError, match="duplicate"):
            rc.resolve_spec([{**b, "name": "a"}, {**b, "name": "a"}])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one bucket"):
            rc.resolve_spec([])

    def test_defaults_are_explicit(self):
        got = _spec()[0]
        assert got["skip_docs"] == 0 and got["min_chars"] == 0
        assert got["config"] == "cfg"


class TestDeterminism:
    def test_same_spec_is_byte_identical(self, fake_stream):
        fake_stream(_docs(20))
        tok = FakeTok()
        a_ids, a_man = rc.build_retained_tokens(tok, _spec(), chunk_tokens=32, seed=7)
        b_ids, b_man = rc.build_retained_tokens(tok, _spec(), chunk_tokens=32, seed=7)
        assert torch.equal(a_ids, b_ids)
        assert a_man["retained_corpus_sha256"] == b_man["retained_corpus_sha256"]

    @pytest.mark.parametrize("change", ["revision", "skip_docs", "chunk_tokens",
                                        "chat_template", "tokens"])
    def test_changing_the_spec_changes_the_hash(self, fake_stream, change):
        fake_stream(_docs(40))
        tok = FakeTok()
        base_kw = dict(chunk_tokens=32, seed=7, chat_template=None)
        _, base = rc.build_retained_tokens(tok, _spec(), **base_kw)

        spec, kw = _spec(), dict(base_kw)
        if change == "revision":
            spec = _spec(revision="rev2")
        elif change == "skip_docs":
            spec = _spec(skip=3)
        elif change == "tokens":
            spec = _spec(tokens=160)
        elif change == "chunk_tokens":
            kw["chunk_tokens"] = 16
        elif change == "chat_template":
            kw["chat_template"] = "user"

        _, other = rc.build_retained_tokens(tok, spec, **kw)
        assert other["retained_corpus_sha256"] != base["retained_corpus_sha256"], (
            f"changing {change} left the corpus hash unchanged")

    def test_hash_is_stable_whether_or_not_already_stamped(self, fake_stream):
        """``manifest_sha256`` excludes the hash field, so hashing a stamped
        manifest reproduces its own stamp. The cache check depends on this."""
        fake_stream(_docs(20))
        _, man = rc.build_retained_tokens(FakeTok(), _spec(), chunk_tokens=32)
        assert rc.manifest_sha256(man) == man["retained_corpus_sha256"]


class TestBudgets:
    def test_hits_the_budget_exactly(self, fake_stream):
        """The last document is truncated at the boundary; the budget is what
        a result quotes, so approximately-200 is not good enough."""
        fake_stream(_docs(40))
        _, man = rc.build_retained_tokens(FakeTok(), _spec(tokens=200), chunk_tokens=8)
        assert man["buckets"][0]["tokens_taken"] == 200

    def test_multiple_buckets_are_summed_and_ordered(self, fake_stream):
        fake_stream(_docs(80))
        spec = rc.resolve_spec([
            {"name": "en", "dataset": "d1", "config": None, "split": "train",
             "revision": "r", "text_field": "text", "tokens": 64},
            {"name": "de", "dataset": "d2", "config": None, "split": "train",
             "revision": "r", "text_field": "text", "tokens": 128},
        ])
        ids, man = rc.build_retained_tokens(FakeTok(), spec, chunk_tokens=32)
        assert [b["name"] for b in man["buckets"]] == ["en", "de"]
        assert man["n_tokens"] == 192 and ids.shape == (6, 32)

    def test_partial_trailing_chunk_is_dropped(self, fake_stream):
        """Every chunk must be the same shape — the projector's per-chunk cost
        is what the wall-clock estimate is built on."""
        fake_stream(_docs(40))
        ids, man = rc.build_retained_tokens(FakeTok(), _spec(tokens=100), chunk_tokens=32)
        assert ids.shape == (3, 32)
        assert man["n_tokens"] == 96 and man["n_tokens_dropped"] == 4

    def test_exhausted_stream_is_an_error_not_a_short_corpus(self, fake_stream):
        """Silently returning 300k tokens when 1M was asked for would make the
        headline number describe a corpus nobody chose."""
        fake_stream(_docs(2, words=5))
        with pytest.raises(RuntimeError, match="exhausted its stream"):
            rc.build_retained_tokens(FakeTok(), _spec(tokens=10_000), chunk_tokens=8)

    def test_twin_past_the_end_of_the_stream_names_the_cause(self, fake_stream):
        """A held-out twin whose skip_docs lies past the last document (the fit
        build consumed the whole split -- the MUSE retain1 case) must say so,
        not report a generic 'exhausted at 0 of N'."""
        fake_stream(_docs(5, words=5))
        spec = _spec(tokens=16)
        spec[0]["skip_docs"] = 5
        with pytest.raises(RuntimeError, match="no documents from skip_docs=5 on.*heldout_mix_tokens: 0"):
            rc.build_retained_tokens(FakeTok(), spec, chunk_tokens=8)

    def test_corpus_shorter_than_one_chunk_is_an_error(self, fake_stream):
        fake_stream(_docs(40))
        with pytest.raises(RuntimeError, match="shorter than one chunk"):
            rc.build_retained_tokens(FakeTok(), _spec(tokens=10), chunk_tokens=4096)


class TestFraming:
    def test_raw_is_the_default(self, fake_stream):
        """The retained framing is raw text unless a chat template is asked for."""
        fake_stream(_docs(40))
        _, man = rc.build_retained_tokens(FakeTok(), _spec(), chunk_tokens=32)
        assert man["chat_template"] is None

    def test_chat_template_changes_the_tokens(self, fake_stream):
        fake_stream(_docs(40))
        tok = FakeTok()
        raw, _ = rc.build_retained_tokens(tok, _spec(), chunk_tokens=32)
        wrapped, man = rc.build_retained_tokens(tok, _spec(), chunk_tokens=32,
                                                chat_template="user")
        assert man["chat_template"] == "user"
        assert not torch.equal(raw, wrapped)

    def test_rejects_unknown_template(self, fake_stream):
        fake_stream(_docs(40))
        with pytest.raises(ValueError, match="chat_template"):
            rc.build_retained_tokens(FakeTok(), _spec(), chat_template="system")

    def test_bos_is_prepended_per_document(self, fake_stream):
        fake_stream(_docs(40, words=8))
        ids, _ = rc.build_retained_tokens(FakeTok(), _spec(tokens=54), chunk_tokens=9,
                                          add_bos=True)
        # 8 words + BOS = 9 tokens per doc, so every chunk starts with BOS.
        assert torch.all(ids[:, 0] == FakeTok.bos_token_id)
        plain, man = rc.build_retained_tokens(FakeTok(), _spec(tokens=54),
                                              chunk_tokens=9, add_bos=False)
        assert man["bos_token_id"] is None
        assert not torch.all(plain[:, 0] == FakeTok.bos_token_id)


class TestCache:
    def test_round_trips_without_rebuilding(self, fake_stream, tmp_path, monkeypatch):
        fake_stream(_docs(40))
        tok = FakeTok()
        path = tmp_path / "tokens.pt"
        ids, man = rc.load_or_build(path, tok, _spec(), chunk_tokens=32)
        assert path.exists()

        def explode(_bucket):
            raise AssertionError("cache miss: the corpus was rebuilt")
        monkeypatch.setattr(rc, "iter_documents", explode)
        ids2, man2 = rc.load_or_build(path, tok, _spec(), chunk_tokens=32)
        assert torch.equal(ids, ids2)
        assert man2["retained_corpus_sha256"] == man["retained_corpus_sha256"]

    def test_shared_path_is_content_addressed_not_name_addressed(self, tmp_path):
        """The probe and the sweep are different experiments with different
        output directories, but they must consume the *same* manifold — if each
        streamed its own they could diverge and the probe's curve would describe
        a corpus the sweep never saw."""
        tok = FakeTok()
        a = rc.shared_cache_path(tok, _spec(), chunk_tokens=1024)
        b = rc.shared_cache_path(tok, _spec(), chunk_tokens=1024)
        assert a == b and "retained_corpus_cache" in a

    @pytest.mark.parametrize("kw,spec_kw", [
        ({"chunk_tokens": 512}, {}),
        ({"chat_template": "user"}, {}),
        ({"add_bos": False}, {}),
        ({}, {"revision": "rev9"}),
        ({}, {"tokens": 999}),
    ])
    def test_shared_path_splits_on_every_corpus_input(self, tmp_path, kw, spec_kw):
        tok = FakeTok()
        base = rc.shared_cache_path(tok, _spec(), chunk_tokens=1024)
        other = rc.shared_cache_path(tok, _spec(**spec_kw),
                                     **{"chunk_tokens": 1024, **kw})
        assert base != other

    def test_seed_does_not_split_the_cache(self, tmp_path):
        """``seed`` shuffles chunk *order* during the sweep; it does not select
        documents, so it must not fork the corpus."""
        tok = FakeTok()
        a = rc.shared_cache_path(tok, _spec(), chunk_tokens=1024, seed=1)
        b = rc.shared_cache_path(tok, _spec(), chunk_tokens=1024, seed=2)
        assert a == b

    def test_changed_spec_invalidates(self, fake_stream, tmp_path):
        fake_stream(_docs(40))
        tok = FakeTok()
        path = tmp_path / "tokens.pt"
        _, a = rc.load_or_build(path, tok, _spec(), chunk_tokens=32)
        _, b = rc.load_or_build(path, tok, _spec(revision="rev2"), chunk_tokens=32)
        assert a["retained_corpus_sha256"] != b["retained_corpus_sha256"]

# --------------------------------------------------------------------------- #
# the per-bucket `format` hook: rendered rows, and the hashes it must not move  #
# --------------------------------------------------------------------------- #
#: Hashes of this file's fixtures under the builder as it stood BEFORE `format`
#: existed (with the FakeTok / _docs / _spec fixtures above). A format-less spec
#: must keep producing exactly these, or every cached corpus and every quoted
#: `retained_corpus_sha256` silently forks.
PRE_FORMAT_SPEC_KEY = "7ae59865893cc653"                 # spec_key(FakeTok, _spec(), chunk 32)
PRE_FORMAT_MANIFEST_SHA = ("857a694b2c512af8f424cb9b4a07c4b05385822509301d46671930d765be630c")
PRE_FORMAT_MIX1M_SPEC_KEY = "206de441ad3a37dd"           # the published 1M-mix buckets, FakeTok

MMLU_ROW = {"question": " Which group is abelian? ", "subject": "abstract_algebra",
            "choices": ["S3", "Z4", "D4", "Q8"], "answer": 1}
MMLU_RENDERED = ("The following are multiple choice questions (with answers) about abstract "
                 "algebra.\n\nWhich group is abelian?\nA. S3\nB. Z4\nC. D4\nD. Q8\nAnswer: B")


class RecordingTok(FakeTok):
    """A FakeTok that records the chat-template call it receives."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        self.calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt,
                           "tokenize": tokenize})
        return " ".join(f"<{m['role']}> {m['content']}" for m in messages)


def _fmt_spec(fmt, text_field, tokens=100, min_chars=0, name="f"):
    return rc.resolve_spec([{
        "name": name, "dataset": "fake/ds", "config": None, "split": "validation",
        "revision": "rev1", "text_field": text_field, "tokens": tokens,
        "format": fmt, "min_chars": min_chars,
    }])


class TestFormats:
    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="unknown format"):
            _fmt_spec("markdown", "text")

    def test_resolve_round_trips_format_and_omits_it_when_unset(self):
        assert _fmt_spec("mmlu_lm_eval", "question")[0]["format"] == "mmlu_lm_eval"
        assert _fmt_spec("chat_messages", "messages")[0]["format"] == "chat_messages"
        assert "format" not in _spec()[0]
        assert "format" not in _fmt_spec(None, "text")[0]
        assert "format" not in _fmt_spec("", "text")[0]

    def test_mmlu_lm_eval_renders_the_harness_prompt(self):
        bucket = _fmt_spec("mmlu_lm_eval", "question")[0]
        assert rc.render_document(FakeTok(), bucket, MMLU_ROW) == MMLU_RENDERED
        # a letter answer and a subject without underscores
        out = rc.render_document(FakeTok(), bucket, {**MMLU_ROW, "answer": "D", "subject": "law"})
        assert out.startswith("The following are multiple choice questions (with answers) about law.\n\n")
        assert out.endswith("\nAnswer: D")

    def test_mmlu_lm_eval_rejects_malformed_rows(self):
        bucket = _fmt_spec("mmlu_lm_eval", "question")[0]
        with pytest.raises(ValueError, match="4 choices"):
            rc.render_document(FakeTok(), bucket, {**MMLU_ROW, "choices": ["a", "b"]})
        with pytest.raises(ValueError, match="out of range"):
            rc.render_document(FakeTok(), bucket, {**MMLU_ROW, "answer": 7})

    def test_chat_messages_uses_the_tokenizer_template_without_a_generation_prompt(self):
        tok = RecordingTok()
        bucket = _fmt_spec("chat_messages", "messages")[0]
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        out = rc.render_document(tok, bucket, {"messages": msgs, "prompt_id": "x"})
        assert out == "<user> hi <assistant> hello"
        assert tok.calls == [{"messages": msgs, "add_generation_prompt": False, "tokenize": False}]

    def test_raw_bucket_render_is_the_identity_and_rejects_rows(self):
        bucket = _spec()[0]
        assert rc.render_document(FakeTok(), bucket, "plain text") == "plain text"
        with pytest.raises(TypeError, match="no format"):
            rc.render_document(FakeTok(), bucket, {"text": "plain text"})

    def test_formatted_bucket_tokenises_and_hashes_the_rendered_text(self, fake_stream):
        rows = [(i, {**MMLU_ROW, "question": f"Question {i}?"}) for i in range(30)]
        fake_stream(rows)
        tok = FakeTok()
        spec = _fmt_spec("mmlu_lm_eval", "question", tokens=64)
        ids, man = rc.build_retained_tokens(tok, spec, chunk_tokens=16)
        rendered = [rc.render_document(tok, spec[0], r) for _, r in rows]
        expect = []
        for t in rendered:
            expect += [FakeTok.bos_token_id] + tok(t)["input_ids"]
            if len(expect) >= 64:
                break
        assert ids.flatten().tolist() == expect[:64]
        report = man["buckets"][0]
        h = hashlib.sha256()
        for t in rendered[:report["n_docs"]]:
            h.update(hashlib.sha256(t.encode("utf-8")).digest())
        assert report["docs_sha256"] == h.hexdigest()
        assert report["format"] == "mmlu_lm_eval" and report["tokens_taken"] == 64

    def test_min_chars_applies_to_the_rendered_text(self, fake_stream):
        short = {**MMLU_ROW, "question": "Q?", "choices": ["a", "b", "c", "d"], "subject": "x"}
        fake_stream([(0, short), (1, MMLU_ROW), (2, MMLU_ROW)])
        tok = FakeTok()
        bucket = _fmt_spec("mmlu_lm_eval", "question")[0]
        n_short = len(rc.render_document(tok, bucket, short))
        assert n_short < len(MMLU_RENDERED)
        spec = _fmt_spec("mmlu_lm_eval", "question", tokens=20, min_chars=n_short + 1)
        _, man = rc.build_retained_tokens(tok, spec, chunk_tokens=10)
        assert man["buckets"][0]["doc_index_start"] == 1      # the short rendering was skipped

    def test_build_level_chat_template_cannot_double_wrap_chat_rows(self, fake_stream):
        fake_stream([(0, {"messages": [{"role": "user", "content": "hi"}]})])
        with pytest.raises(ValueError, match="wrap them twice"):
            rc.build_retained_tokens(RecordingTok(), _fmt_spec("chat_messages", "messages", tokens=4),
                                     chunk_tokens=2, chat_template="user")

    def test_format_splits_the_cache_key(self):
        tok = FakeTok()
        assert (rc.spec_key(tok, _fmt_spec(None, "question"), chunk_tokens=1024)
                != rc.spec_key(tok, _fmt_spec("mmlu_lm_eval", "question"), chunk_tokens=1024))

    def test_format_less_hashes_are_unchanged(self, fake_stream):
        """Regression pin, see ``PRE_FORMAT_*`` above."""
        fake_stream(_docs(20))
        tok = FakeTok()
        assert rc.spec_key(tok, _spec(), chunk_tokens=32) == PRE_FORMAT_SPEC_KEY
        _, man = rc.build_retained_tokens(tok, _spec(), chunk_tokens=32, seed=7)
        assert man["retained_corpus_sha256"] == PRE_FORMAT_MANIFEST_SHA
        assert "format" not in man["buckets"][0]
        import yaml

        # the 1M mix = the first four buckets of the shipped Dutch-sets spec
        cfg = yaml.safe_load(Path(spec_path("dutch_sets_1k")).read_text())
        blk = dict(cfg["retained_corpus"])
        blk["buckets"] = [b for b in blk["buckets"] if b["name"] != "chat"]
        assert rc.spec_key(tok, rc.resolve_spec(blk["buckets"]), chunk_tokens=blk["chunk_tokens"],
                           add_bos=blk["add_bos"], chat_template=blk["chat_template"]) == PRE_FORMAT_MIX1M_SPEC_KEY

    def test_iter_documents_yields_rows_for_formatted_buckets(self, monkeypatch):
        """The real stream function over a stand-in ``datasets``: whole rows (field
        present) for a formatted bucket, filtered strings for a raw one."""
        import types

        rows = [{"question": "q0", "answer": 0, "text": "short"},
                {"question": None, "answer": 1, "text": "long enough text"},
                {"answer": 2, "text": 12345},
                {"question": "q3", "answer": 3, "text": "another long text"}]
        calls = []

        def load_dataset(path, config, split, revision, streaming):
            calls.append((path, config, split, revision, streaming))
            return iter(rows)

        monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
        fmt = _fmt_spec("mmlu_lm_eval", "question")[0]
        assert list(rc.iter_documents(fmt)) == [(0, rows[0]), (3, rows[3])]
        raw = rc.resolve_spec([{"name": "r", "dataset": "fake/ds", "split": "train", "revision": "rev1",
                                "text_field": "text", "tokens": 10, "min_chars": 6}])[0]
        assert list(rc.iter_documents(raw)) == [(1, "long enough text"), (3, "another long text")]
        assert calls[0] == ("fake/ds", None, "validation", "rev1", True)


# --------------------------------------------------------------------------- #
# exclude_question_sha256: the dedup filter of an mmlu_lm_eval bucket           #
# --------------------------------------------------------------------------- #
#: Pins of a FORMAT-ONLY spec (format set, no exclusion list) under the module as it stood
#: at c0d61a4, before `exclude_question_sha256` existed: _fmt_spec("mmlu_lm_eval", "question",
#: tokens=64) over `_mmlu_rows(30)`, chunk 16, FakeTok. The key must not move them.
FORMAT_ONLY_SPEC_KEY = "e80a8c1fc2965a84"
FORMAT_ONLY_MANIFEST_SHA = "a78a5c05741e6ce54b0aca2026b5e6f0d389ac5b161714a3b84f182d82978c3c"


def _mmlu_rows(n):
    return [(i, {**MMLU_ROW, "question": f"Question {i}?"}) for i in range(n)]


def _excl_spec(digests, fmt="mmlu_lm_eval", text_field="question", tokens=64, **extra):
    raw = {"name": "f", "dataset": "fake/ds", "config": None, "split": "validation", "revision": "rev1",
           "text_field": text_field, "tokens": tokens, "exclude_question_sha256": digests, **extra}
    if fmt is not None:
        raw["format"] = fmt
    return rc.resolve_spec([raw])


class TestExclusion:
    def test_question_sha256_normalises_whitespace(self):
        assert rc.question_sha256("  a   b\nc ") == rc.question_sha256("a b c")
        assert rc.question_sha256("a b c") == hashlib.sha256(b"a b c").hexdigest()

    def test_requires_the_mmlu_format(self):
        d = [rc.question_sha256("q")]
        with pytest.raises(ValueError, match="applies only to format 'mmlu_lm_eval'"):
            _excl_spec(d, fmt=None, text_field="text")
        with pytest.raises(ValueError, match="applies only to format 'mmlu_lm_eval'"):
            _excl_spec(d, fmt="chat_messages", text_field="messages")

    def test_validates_the_entries(self):
        with pytest.raises(ValueError, match="must be a list"):
            _excl_spec(rc.question_sha256("q"))                       # a bare string
        with pytest.raises(ValueError, match="not a 64-hex sha256"):
            _excl_spec(["abc"])
        with pytest.raises(ValueError, match="not a 64-hex sha256"):
            _excl_spec(["g" * 64])
        with pytest.raises(ValueError, match="duplicate digests"):
            _excl_spec([rc.question_sha256("q"), rc.question_sha256("q").upper()])

    def test_stored_sorted_and_only_when_set(self):
        a, b = rc.question_sha256("Question 2?"), rc.question_sha256("Question 0?")
        got = _excl_spec([a.upper(), b])[0]
        assert got["exclude_question_sha256"] == sorted([a, b])
        for empty in (None, [], ""):
            assert "exclude_question_sha256" not in _excl_spec(empty)[0]
        assert "exclude_question_sha256" not in _fmt_spec("mmlu_lm_eval", "question")[0]

    def test_excluded_rows_are_skipped_before_rendering_and_counted(self, fake_stream):
        rows = _mmlu_rows(30)
        fake_stream(rows)
        tok = FakeTok()
        spec = _excl_spec([rc.question_sha256("Question 0?"), rc.question_sha256("Question 2?")])
        ids, man = rc.build_retained_tokens(tok, spec, chunk_tokens=16)
        report = man["buckets"][0]
        assert report["n_excluded"] == 2 and report["doc_index_start"] == 1
        assert report["exclude_question_sha256"] == spec[0]["exclude_question_sha256"]
        kept = [r for i, r in rows if i not in (0, 2)]
        expect, h = [], hashlib.sha256()
        for r in kept[:report["n_docs"]]:
            text = rc.render_document(tok, spec[0], r)
            expect += [FakeTok.bos_token_id] + tok(text)["input_ids"]
            h.update(hashlib.sha256(text.encode("utf-8")).digest())
        assert ids.flatten().tolist() == expect[:64] and report["docs_sha256"] == h.hexdigest()
        # the same rows without the filter take documents 0, 1, 2, ...
        _, plain = rc.build_retained_tokens(tok, _fmt_spec("mmlu_lm_eval", "question", tokens=64), chunk_tokens=16)
        assert plain["buckets"][0]["doc_index_start"] == 0

    def test_n_excluded_counts_only_rows_actually_encountered(self, fake_stream):
        fake_stream(_mmlu_rows(30))
        spec = _excl_spec([rc.question_sha256("Question 1?"), rc.question_sha256("Question 29?")])
        _, man = rc.build_retained_tokens(FakeTok(), spec, chunk_tokens=16)   # budget met after ~3 docs
        assert man["buckets"][0]["n_excluded"] == 1 and man["buckets"][0]["n_docs"] == 3

    def test_hash_stability(self, fake_stream):
        """Format-only specs keep the hashes pinned above; an exclusion list moves both and shows
        up in the manifest only when set."""
        fake_stream(_mmlu_rows(30))
        tok = FakeTok()
        spec = _fmt_spec("mmlu_lm_eval", "question", tokens=64)
        assert rc.spec_key(tok, spec, chunk_tokens=16) == FORMAT_ONLY_SPEC_KEY
        _, man = rc.build_retained_tokens(tok, spec, chunk_tokens=16)
        assert man["retained_corpus_sha256"] == FORMAT_ONLY_MANIFEST_SHA
        assert "exclude_question_sha256" not in man["buckets"][0] and "n_excluded" not in man["buckets"][0]
        excl = _excl_spec([rc.question_sha256("Question 0?")])
        assert rc.spec_key(tok, excl, chunk_tokens=16) != FORMAT_ONLY_SPEC_KEY
        _, man2 = rc.build_retained_tokens(tok, excl, chunk_tokens=16)
        assert man2["retained_corpus_sha256"] != FORMAT_ONLY_MANIFEST_SHA
        # an exclusion list that matches nothing still changes the spec (it is an input), but
        # the format-less pins of test_format_less_hashes_are_unchanged are untouched by construction
        assert rc.spec_key(tok, _excl_spec([rc.question_sha256("never")]), chunk_tokens=16) != FORMAT_ONLY_SPEC_KEY

    def test_cache_at_the_same_path_is_rebuilt_when_the_list_changes(self, fake_stream, tmp_path):
        fake_stream(_mmlu_rows(30))
        tok, path = FakeTok(), tmp_path / "tokens.pt"
        _, a = rc.load_or_build(path, tok, _fmt_spec("mmlu_lm_eval", "question", tokens=64), chunk_tokens=16)
        _, b = rc.load_or_build(path, tok, _excl_spec([rc.question_sha256("Question 0?")]), chunk_tokens=16)
        assert a["retained_corpus_sha256"] != b["retained_corpus_sha256"]
        assert b["buckets"][0]["n_excluded"] == 1


class TestShippedSpec:
    """The shipped Dutch-sets spec's retained block satisfies the schema, pins
    real revisions and names its chunk length."""

    def _block(self):
        import yaml
        return yaml.safe_load(Path(spec_path("dutch_sets_1k")).read_text())["retained_corpus"]

    def test_spec_resolves(self):
        assert rc.resolve_spec(self._block()["buckets"])

    def test_revisions_are_pinned_to_real_shas(self):
        for bucket in rc.resolve_spec(self._block()["buckets"]):
            rev = bucket["revision"]
            assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), bucket["name"]

    def test_chunk_tokens_is_recorded(self):
        assert int(self._block().get("chunk_tokens", 0)) > 0

    def test_module_does_not_import_datasets_at_import_time(self):
        import importlib
        import subprocess
        code = ("import sys; import rspd.experiments.data.retained_corpus; "
                "assert 'datasets' not in sys.modules")
        importlib.invalidate_caches()
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_local_jsonl_bucket_may_carry_a_format(tmp_path):
    """The two bucket extensions compose (``kind: local_jsonl`` and ``format``):
    a local JSONL bucket with a ``format`` yields whole rows, rendered exactly as a hub
    bucket's are; without a ``format`` it yields the field's strings as before,
    and ``format`` is stored only when set."""
    path = tmp_path / "mcq.jsonl"
    rows = [{**MMLU_ROW, "question": f"Q{i}?"} for i in range(3)] + [{"answer": 0}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    fmt = rc.resolve_spec([{"name": "mcq", "kind": "local_jsonl", "path": str(path),
                            "text_field": "question", "tokens": 10, "format": "mmlu_lm_eval"}])[0]
    assert fmt["kind"] == "local_jsonl" and fmt["format"] == "mmlu_lm_eval"
    docs = list(rc.iter_documents(fmt))
    assert [i for i, _ in docs] == [0, 1, 2] and docs[0][1] == rows[0]      # row 3: no question
    assert rc.render_document(FakeTok(), fmt, docs[1][1]).startswith(
        "The following are multiple choice questions (with answers) about abstract algebra.\n\nQ1?\n")
    plain = rc.resolve_spec([{"name": "q", "kind": "local_jsonl", "path": str(path),
                              "text_field": "question", "tokens": 10}])[0]
    assert "format" not in plain
    assert list(rc.iter_documents(plain)) == [(0, "Q0?"), (1, "Q1?"), (2, "Q2?")]
    assert rc.spec_key(FakeTok(), [fmt], chunk_tokens=8) != rc.spec_key(FakeTok(), [plain], chunk_tokens=8)


# --------------------------------------------------------------------------- #
# One BOS per document: a chat template that emits BOS itself (Gemma)          #
# --------------------------------------------------------------------------- #
class BosTemplateTok(FakeTok):
    """A FakeTok whose chat template starts with a literal ``<bos>`` that the
    tokenizer maps to ``bos_token_id`` -- what Gemma's template and tokenizer do."""

    bos_token = "<bos>"

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = []
        for w in text.split():
            ids.append(self.bos_token_id if w == self.bos_token
                       else (zlib.crc32(w.encode("utf-8")) % 1000) + 10)
        return {"input_ids": ids}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        body = " ".join(f"<{m['role']}> {m['content']}" for m in messages)
        return "<bos> " + body + (" <model>" if add_generation_prompt else "")


def _chat_rows(n):
    return [(i, {"messages": [{"role": "user", "content": f"q{i} a b c"},
                              {"role": "assistant", "content": f"r{i} d e f"}]})
            for i in range(n)]


class TestSingleBos:
    def test_chat_bucket_gets_exactly_one_bos_per_document(self, fake_stream):
        fake_stream(_chat_rows(20))
        tok = BosTemplateTok()
        ids, man = rc.build_retained_tokens(tok, _fmt_spec("chat_messages", "messages", tokens=90),
                                            chunk_tokens=9, add_bos=True)
        flat = ids.reshape(-1)
        # each rendered row is "<bos> <user> qI a b c <assistant> rI d e f" = 11 tokens
        assert not rc.has_double_bos(ids, tok.bos_token_id)
        assert int(flat[0]) == tok.bos_token_id and int(flat[1]) != tok.bos_token_id
        assert man["buckets"][0]["n_docs_bos_already_present"] == man["buckets"][0]["n_docs"]
        # BOS positions are exactly the document starts (every 11 tokens)
        bos_pos = (flat == tok.bos_token_id).nonzero().reshape(-1).tolist()
        assert bos_pos == list(range(0, flat.numel(), 11))

    def test_build_level_user_framing_is_guarded_too(self, fake_stream):
        fake_stream(_docs(20, words=5))
        tok = BosTemplateTok()
        ids, _ = rc.build_retained_tokens(tok, _spec(tokens=60), chunk_tokens=6,
                                          chat_template="user")
        assert not rc.has_double_bos(ids, tok.bos_token_id)

    def test_raw_buckets_still_get_their_bos_and_keep_their_manifest(self, fake_stream):
        """The pinned format-less hashes (TestFormats) are the regression pin; this
        adds that the new bucket key is absent when no rendering started with BOS."""
        fake_stream(_docs(40, words=8))
        ids, man = rc.build_retained_tokens(BosTemplateTok(), _spec(tokens=54), chunk_tokens=9)
        assert torch.all(ids[:, 0] == FakeTok.bos_token_id)
        assert "n_docs_bos_already_present" not in man["buckets"][0]

    def test_has_double_bos(self):
        assert rc.has_double_bos(torch.tensor([[2, 2, 5], [7, 8, 9]]), 2)
        assert rc.has_double_bos(torch.tensor([[5, 2], [2, 9]]), 2)      # across a chunk edge
        assert not rc.has_double_bos(torch.tensor([[2, 5, 2], [6, 2, 9]]), 2)
        assert not rc.has_double_bos(torch.tensor([[2, 2]]), None)

    def test_stale_double_bos_cache_is_rebuilt(self, fake_stream, tmp_path):
        fake_stream(_chat_rows(20))
        tok = BosTemplateTok()
        spec = _fmt_spec("chat_messages", "messages", tokens=90)
        path = tmp_path / "c.pt"
        ids, man = rc.load_or_build(path, tok, spec, chunk_tokens=9)
        # forge the pre-fix cache: same manifest (a valid hash), doubled BOS tokens
        bad = ids.clone().reshape(-1)
        bad[1] = tok.bos_token_id
        torch.save({"ids": bad.reshape(ids.shape), "manifest": man}, path)
        again, _ = rc.load_or_build(path, tok, spec, chunk_tokens=9)
        assert torch.equal(again, ids)
        assert not rc.has_double_bos(torch.load(path)["ids"], tok.bos_token_id)


def test_on_document_sees_exactly_the_selected_documents_and_tokens(fake_stream):
    fake_stream(_docs(40, words=8))
    seen = []
    ids, man = rc.build_retained_tokens(FakeTok(), _spec(tokens=50), chunk_tokens=5,
                                        on_document=lambda *a: seen.append(a))
    _, man2 = rc.build_retained_tokens(FakeTok(), _spec(tokens=50), chunk_tokens=5)
    assert man == man2                                       # the hook is not provenance
    assert [s[1] for s in seen] == list(range(man["buckets"][0]["n_docs"]))
    assert sum(len(s[3]) for s in seen) == 50                # the last document is cut
    assert [t for s in seen for t in s[3]] == ids.reshape(-1).tolist()
    assert all(s[0] == "a" and s[3][0] == FakeTok.bos_token_id for s in seen)
