"""The GPT-2 unlearning sets (``srspd_experiments.data.gpt2_unlearning``): the sanitiser, the
exact token budgets (the target cut at exactly 1,000 with a tiny tokenizer),
counterpart-first ordering and flagging, disjointness by page id, the retried
API client, and the written layout (JSONL + shards through the pretraining
corpus's writer + manifest). A fake article iterator and a monkeypatched
langlinks / extracts fetcher -- no network, no ``datasets``."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import yaml

from pathlib import Path

from srspd_experiments.data import gpt2_unlearning as uc
from srspd_experiments.data import spec_path

CFG = Path(spec_path("gpt2_unlearning_1m"))


class CharTok:
    """One token per character; decode is exact, so a cut re-tokenises to its
    own length. ``eos_token_id`` 0, characters from 1."""
    eos_token_id = 0
    vocab_size = 300

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[(ord(c) % 250) + 1 for c in t] for t in texts]}

    def decode(self, ids):
        return "".join(chr(i - 1 + (ord("a") if False else 0)) if False else "x" for i in ids)


class RoundTripTok(CharTok):
    """Same, with a real decode (keeps the characters)."""

    def __init__(self):
        self._chars = {}

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            ids = []
            for c in t:
                i = (ord(c) % 250) + 1
                self._chars.setdefault(i, c)
                ids.append(i)
            out.append(ids)
        return {"input_ids": out}

    def decode(self, ids):
        return "".join(self._chars.get(i, "?") for i in ids)


def _para(n, seed=0):
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    out = []
    i = 0
    while sum(len(w) + 1 for w in out) < n:
        out.append(words[(i + seed) % len(words)])
        i += 1
    return " ".join(out)


# --------------------------------------------------------------------------- #
# sanitiser                                                                    #
# --------------------------------------------------------------------------- #
class TestSanitize:
    def test_markup_is_stripped_and_link_text_kept(self):
        body = _para(400)
        raw = (f"Intro {{{{Infobox|a=b}}}} with [[Amsterdam|the capital]] and [[Rotterdam]] "
               f"<ref name=x>cite</ref> and <ref group=n/> ''italic'' '''bold''' "
               f"[http://example.org a site] <b>tag</b> [[File:Map.png|thumb|caption]] "
               f"{{{{outer {{{{inner}}}} text}}}} {body}\n\n{body}\n\n{body}")
        out = uc.sanitize_wiki_text(raw)
        assert out
        for bad in ("{{", "}}", "[[", "]]", "<ref", "<b>", "'''", "''", "File:", "http://"):
            assert bad not in out, bad
        assert "the capital" in out and "Rotterdam" in out and "a site" in out
        assert "cite" not in out and "caption" not in out and "inner" not in out

    def test_headings_dropped_and_references_cut(self):
        p1, p2 = _para(300, 1), _para(300, 2)
        for stop in ("== References ==", "=== Referenties ===", "Einzelnachweise", "Zie ook",
                     "See also:", "Externe links", "Bronnen"):
            raw = f"== Lead ==\n{p1}\n\n== History ==\n{p2}\n\n{stop}\n{_para(300, 3)}\n{_para(300, 4)}"
            out = uc.sanitize_wiki_text(raw, min_doc_chars=100)
            assert out == f"{p1}\n\n{p2}", stop
        # a heading-like line that is not a stop heading is just dropped
        out = uc.sanitize_wiki_text(f"{p1}\n== Geography ==\n{p2}", min_doc_chars=100)
        assert out == f"{p1}\n\n{p2}"
        # a bare 'references' INSIDE a paragraph is not a heading
        raw = f"{p1} references are listed {p2}"
        assert uc.sanitize_wiki_text(raw, min_doc_chars=100) == raw

    def test_whitespace_collapsed_short_paragraphs_and_short_docs_dropped(self):
        p = _para(250)
        raw = f"  {p.replace(' ', '   ')}  \n\nshort line\n\n{p}\t\n   \n"
        assert uc.sanitize_wiki_text(raw, min_doc_chars=100) == f"{p}\n\n{p}"
        assert uc.sanitize_wiki_text(raw, min_paragraph_chars=10, min_doc_chars=100).count("\n\n") == 2
        assert uc.sanitize_wiki_text(p) == ""                 # under 1,000 chars
        assert uc.sanitize_wiki_text("") == "" and uc.sanitize_wiki_text(None) == ""
        long = "\n\n".join(_para(300, i) for i in range(4))
        assert uc.sanitize_wiki_text(long) == long


# --------------------------------------------------------------------------- #
# budgets                                                                      #
# --------------------------------------------------------------------------- #
def test_truncate_to_tokens_is_exact_with_a_round_trip_tokenizer():
    tok = RoundTripTok()
    text = _para(2000)
    cut, n = uc.truncate_to_tokens(tok, text, 1000)
    assert n == 1000 and len(cut) == 1000 and text.startswith(cut.rstrip())
    assert uc.truncate_to_tokens(tok, "abc", 10) == ("abc", 3)


def test_accept_page_is_a_function_of_seed_and_id_only():
    a = [uc.accept_page(i, 7, 0.3) for i in range(2000)]
    assert a == [uc.accept_page(i, 7, 0.3) for i in range(2000)]
    assert 0.2 < sum(a) / len(a) < 0.4
    assert a != [uc.accept_page(i, 8, 0.3) for i in range(2000)]
    assert all(uc.accept_page(i, 1, 1.0) for i in range(50))


def _docs(lang, n, chars=1500, start=0):
    return [{"id": f"{lang}{start + i}", "title": f"{lang} article {start + i}",
             "text": "\n\n".join(_para(chars // 3, i + j) for j in range(3))}
            for i in range(n)]


def test_fill_role_takes_whole_docs_then_truncates_at_exactly_the_budget():
    tok = RoundTripTok()
    docs = _docs("nl", 10, chars=600)
    it = iter(docs)
    fill = uc.fill_role(it, budget=1000, tokenizer=tok, role="target", lang="nld", log=lambda *_: None)
    assert fill["n_tokens"] == 1000 and fill["complete"]
    assert [d["n_tokens"] for d in fill["docs"]][:-1] == [len(d["text"]) for d in docs[:len(fill["docs"]) - 1]]
    assert fill["docs"][-1]["truncated"] and not fill["docs"][0]["truncated"]
    assert sum(d["n_tokens"] for d in fill["docs"]) == 1000
    # the iterator was not consumed past the budget: the next role continues
    rest = list(it)
    assert len(rest) == 10 - len(fill["docs"])
    # a short stream: incomplete, whole
    fill = uc.fill_role(iter(_docs("x", 2, chars=100)), budget=10 ** 6, tokenizer=tok, role="h",
                        lang="nld", log=lambda *_: None)
    assert not fill["complete"] and len(fill["docs"]) == 2


def test_fill_role_puts_counterparts_first_flagged_and_excludes_their_ids():
    tok = RoundTripTok()
    preset = [dict(d, source_lang="nld") for d in _docs("en", 2, chars=300, start=100)]
    stream = iter(_docs("en", 20, chars=300) + _docs("en", 1, chars=300, start=100))
    fill = uc.fill_role(stream, budget=2000, tokenizer=tok, role="contrastive", lang="eng",
                        exclude_ids={"en100", "en101"}, preset=preset, log=lambda *_: None)
    docs = fill["docs"]
    assert [d["is_target_counterpart"] for d in docs[:2]] == [True, True]
    assert all(not d["is_target_counterpart"] for d in docs[2:])
    assert docs[0]["source_lang"] == "nld" and docs[2]["source_lang"] == "eng"
    assert len({d["id"] for d in docs}) == len(docs) and "en100" not in {d["id"] for d in docs[2:]}
    assert fill["n_tokens"] == 2000


# --------------------------------------------------------------------------- #
# the API client                                                               #
# --------------------------------------------------------------------------- #
def test_get_json_retries_with_backoff_then_fails_loudly(monkeypatch):
    calls, sleeps = [], []
    import urllib.request

    def boom(req, timeout=None):
        calls.append((req.full_url, req.get_header("User-agent", "")))
        raise OSError("down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="unreachable after 3 attempts"):
        uc.get_json("https://nl.wikipedia.org/w/api.php", {"a": 1}, retries=3, backoff=0.5,
                    sleep=sleeps.append)
    assert len(calls) == 3 and sleeps == [0.5, 1.0]
    # a generic client name, no project details
    assert all(ua == uc.USER_AGENT and ua.startswith("srspd-experiments") for _, ua in calls)


def test_langlinks_and_extracts_parse_the_api_shapes():
    seen = []

    def fake_get(url, params):
        seen.append((url, params))
        if params["prop"] == "langlinks":
            code = params["lllang"]
            return {"query": {"redirects": [{"from": "Rotterdam (stad)", "to": "Rotterdam"}],
                              "pages": [{"pageid": 1, "title": "Amsterdam",
                                         "langlinks": [{"lang": code, "title": f"Amsterdam-{code}"}]},
                                        {"pageid": 2, "title": "Rotterdam",
                                         "langlinks": [{"lang": code, "title": f"Rotterdam-{code}"}]
                                         if code == "en" else []},
                                        {"title": "Nope", "missing": True}]}}
        return {"query": {"pages": [{"pageid": 844, "title": params["titles"],
                                     "extract": "== Lead ==\n" + _para(1200)}]}}

    links = uc.fetch_langlinks(["Amsterdam", "Rotterdam (stad)", "Nope"], endpoint="E",
                               langs={"eng": "en", "deu": "de"}, get=fake_get)
    assert links == {"Amsterdam": {"de": "Amsterdam-de", "en": "Amsterdam-en"},
                     "Rotterdam (stad)": {"en": "Rotterdam-en"}, "Nope": {}}
    assert [p["lllang"] for _u, p in seen] == ["de", "en"]      # one request per language
    page = uc.fetch_extract("Amsterdam-en", lang_code="en", endpoint_template="https://{lang}.x/api",
                            get=fake_get)
    assert page["pageid"] == "844" and seen[-1][0] == "https://en.x/api"
    assert seen[-1][1]["explaintext"] == 1

    def missing(url, params):
        return {"query": {"pages": [{"title": "Nope", "missing": True}]}}
    with pytest.raises(LookupError):
        uc.fetch_extract("Nope", lang_code="en", endpoint_template="{lang}", get=missing)


# --------------------------------------------------------------------------- #
# the whole build: selection, disjointness, outputs                            #
# --------------------------------------------------------------------------- #
def _spec(**over):
    cfg = yaml.safe_load(CFG.read_text())
    # articles are ~1,500 chars (tokens, with the char tokenizer): the target is
    # two whole articles and a truncated third, as the real 1,000-token set is
    cfg["budgets"] = {"target_nld": 4000, "contrastive_eng": 9000, "contrastive_deu": 9000,
                      "heldout_eng": 3000, "heldout_deu": 3000, "heldout_nld": 3000}
    cfg.update(over)
    return uc.resolve_spec(cfg)


def _fake_sources():
    pools = {"nld": _docs("nl", 60, chars=1500), "eng": _docs("en", 60, chars=1500),
             "deu": _docs("de", 60, chars=1500)}
    # a few raw articles that sanitise to nothing, and some markup residue
    pools["nld"][1]["text"] = "too short"
    pools["eng"][0]["text"] = "{{stub}}\n\n" + pools["eng"][0]["text"] + "\n\n== References ==\n" + _para(500)

    def source(lang):
        return iter(pools[lang])
    return pools, source


def _fake_counterparts(target_docs, *, miss_title=None, pools=None):
    by_lang = {"eng": [], "deu": []}
    unmatched = []
    for k, d in enumerate(target_docs):
        for lang in ("eng", "deu"):
            if d["title"] == miss_title:
                unmatched.append({"target_id": d["id"], "title": d["title"], "lang": lang,
                                  "reason": "no langlink"})
                continue
            # the counterpart's page id collides with a pool page on purpose: the
            # fill must exclude it
            twin = pools[lang][(3 if lang == "eng" else 5) + k]
            by_lang[lang].append({"id": twin["id"], "title": f"{d['title']} ({lang})",
                                  "text": twin["text"], "source_lang": "nld",
                                  "counterpart_of": {"id": d["id"], "title": d["title"]}})
    return by_lang, unmatched


def test_build_sets_is_exact_disjoint_and_counterpart_first():
    pools, source = _fake_sources()
    spec = _spec()
    tok = RoundTripTok()
    target_titles = []

    def counterparts(target_docs):
        target_titles.extend(d["title"] for d in target_docs)
        return _fake_counterparts(target_docs, miss_title=target_docs[-1]["title"], pools=pools)

    sets = uc.build_sets(spec, tok, article_source=source, counterparts=counterparts,
                         log=lambda *_: None)
    roles = sets["roles"]
    assert roles["target_nld"]["n_tokens"] == 4000 and roles["target_nld"]["docs"][-1]["truncated"]
    assert len(roles["target_nld"]["docs"]) == 3
    assert all(roles[r]["n_tokens"] == spec["budgets"][r] and roles[r]["complete"] for r in uc.ROLES)
    # the sanitiser dropped nl1 and cleaned en0
    assert "nl1" not in {d["id"] for d in roles["target_nld"]["docs"]}
    en_texts = {d["id"]: d["text"] for d in roles["contrastive_eng"]["docs"]}
    assert "{{" not in en_texts.get("en0", "") and "References" not in en_texts.get("en0", "")
    # counterparts first, flagged, one per matched target per language
    n_t = len(roles["target_nld"]["docs"])
    for lang in ("eng", "deu"):
        docs = roles[f"contrastive_{lang}"]["docs"]
        flags = [d["is_target_counterpart"] for d in docs]
        assert flags[:n_t - 1] == [True] * (n_t - 1) and not any(flags[n_t - 1:])
        assert docs[0]["source_lang"] == "nld"
    assert len(sets["counterparts"]["matched"]) == 2 * (n_t - 1)
    assert [u["title"] for u in sets["counterparts"]["unmatched"]] == [target_titles[-1]] * 2
    # disjoint by page id within every language, counterpart ids excluded from the fill
    for lang in ("nld", "eng", "deu"):
        ids = [d["id"] for r in uc.ROLES if uc.LANG_OF_ROLE[r] == lang for d in roles[r]["docs"]]
        assert len(ids) == len(set(ids)), lang
    assert sets["sanitize_stats"]["nld"]["dropped"] == 1


def test_write_outputs_layout_shards_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    pools, source = _fake_sources()
    spec = _spec()
    tok = RoundTripTok()
    sets = uc.build_sets(spec, tok, article_source=source,
                         counterparts=lambda t: _fake_counterparts(t, pools=pools),
                         log=lambda *_: None)
    out = tmp_path / "datasets" / "gpt2_unlearning" / "gpt2_unlearning"
    man = uc.write_outputs(sets, spec, tok, str(out), tokenizer_dir_ref="datasets/gpt2_corpus/tokenizer",
                           log=lambda *_: None)
    assert man["n_tokens"] == spec["budgets"] and man["tokenizer_dir"] == "datasets/gpt2_corpus/tokenizer"
    assert man["revisions"] == {l: "b04c8d1ceb2f5cd4588862100d08de323dccfbaa" for l in ("nld", "eng", "deu")}
    assert len(man["spec_sha256"]) == 64 and man["sanitisation_rules"] == list(uc.SANITIZE_RULES)
    for role in uc.ROLES:
        rows = [json.loads(l) for l in (out / f"{role}.jsonl").read_text().splitlines()]
        assert rows and set(rows[0]) == {"id", "title", "lang", "role", "text", "is_target_counterpart", "n_tokens"}
        assert sum(r["n_tokens"] for r in rows) == spec["budgets"][role]
        assert rows[0]["lang"] == uc.LANG_OF_ROLE[role] and rows[0]["role"] == role.rsplit("_", 1)[0]
        shard = out / "shards" / role / "shard_0000.bin"
        arr = np.fromfile(shard, dtype=np.uint16)
        # the documents' tokens + one eos after each, exactly
        assert arr.size == spec["budgets"][role] + len(rows) == man["n_shard_tokens"][role]
        assert int((arr == tok.eos_token_id).sum()) == len(rows)
        r = man["roles"][role]
        assert r["shards"] == [f"shards/{role}/shard_0000.bin"] and r["n_docs"] == len(rows)
        assert [d["id"] for d in r["documents"]] == [x["id"] for x in rows]
        assert r["jsonl"] == f"{role}.jsonl"
    assert all(d["is_target_counterpart"] for d in man["roles"]["contrastive_eng"]["documents"][:2])
    md = uc.results_markdown(man, elapsed_s=12.0)
    assert "`target_nld` | 4,000 | 4,000 |" in md and "## Counterparts" in md


def test_prepare_runs_end_to_end_with_fakes_and_shards_rematerialise(tmp_path, monkeypatch):
    """``prepare_unlearning_sets`` with the hub stream and the API faked: reads
    the tokenizer from datasets/<corpus_name>/tokenizer, writes the sets and
    results.md; ``ensure_unlearning_sets`` finds them, copies them from a
    directory source, and ``materialize_shards`` rewrites byte-identical shards
    from the JSONL alone."""
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    from gpt2_tiny_helpers import tiny_tokenizer
    tok_dir = tmp_path / "datasets" / "gpt2_corpus" / "tokenizer"
    tiny_tokenizer().save_pretrained(str(tok_dir))
    cfg = yaml.safe_load(CFG.read_text())
    cfg["budgets"] = {"target_nld": 60, "contrastive_eng": 200, "contrastive_deu": 200,
                      "heldout_eng": 80, "heldout_deu": 80, "heldout_nld": 80}
    cfg["sanitize"] = {"min_paragraph_chars": 20, "min_doc_chars": 50}
    words = lambda i, n: " ".join(f"w{(i * 7 + j) % 200}" for j in range(n))
    pools = {l: [{"id": f"{l}{i}", "title": f"{l} {i}", "text": words(i, 40) + "\n\n" + words(i + 1, 40)}
                 for i in range(40)] for l in ("nld", "eng", "deu")}
    monkeypatch.setattr(uc, "iter_hub_articles", lambda src, seed: iter(pools[
        {"20231101.nl": "nld", "20231101.en": "eng", "20231101.de": "deu"}[src["config"]]]))

    def fake_get(url, params, user_agent=None):
        assert user_agent == uc.USER_AGENT
        if params["prop"] == "langlinks":
            return {"query": {"pages": [{"pageid": 1, "title": t,
                                         "langlinks": [{"lang": params["lllang"], "title": f"{t}-{params['lllang']}"}]}
                                        for t in params["titles"].split("|")]}}
        return {"query": {"pages": [{"pageid": 9000 + len(params["titles"]), "title": params["titles"],
                                     "extract": words(3, 60)}]}}
    monkeypatch.setattr(uc, "get_json", fake_get)
    man = uc.prepare_unlearning_sets(cfg, log=lambda *_: None)
    out = tmp_path / "datasets" / "gpt2_unlearning" / "gpt2_unlearning_1m"
    assert json.loads((out / "manifest.json").read_text())["n_tokens"] == cfg["budgets"]
    assert man["tokenizer_dir"] == "datasets/gpt2_corpus/tokenizer"
    assert (tmp_path / "logs" / "gpt2" / "gpt2_unlearning_1m" / "results.md").exists()
    assert man["counterparts"]["unmatched"] == []
    assert len(man["counterparts"]["matched"]) == 2 * man["n_docs"]["target_nld"]
    # ensure: already present -> the same directory
    assert uc.ensure_unlearning_sets("gpt2_unlearning_1m") == str(out)
    # a directory source is copied into a fresh data root
    other = tmp_path / "other"
    monkeypatch.setenv("SRSPD_DATA_DIR", str(other))
    with pytest.raises(FileNotFoundError, match="no published source"):
        uc.ensure_unlearning_sets("gpt2_unlearning_1m")
    got = uc.ensure_unlearning_sets("gpt2_unlearning_1m", source=str(out))
    assert sorted(os.listdir(got)) == sorted([f"{r}.jsonl" for r in uc.ROLES] + ["manifest.json"])
    # shards from the JSONL alone are the build's shards, byte for byte
    tok = tiny_tokenizer()
    written = uc.materialize_shards(got, tok, log=lambda *_: None)
    assert set(written) == set(uc.ROLES)
    for role in uc.ROLES:
        a_ = (out / "shards" / role / "shard_0000.bin").read_bytes()
        b_ = (Path(got) / "shards" / role / "shard_0000.bin").read_bytes()
        assert a_ == b_, role
    assert uc.unlearning_shards(got, "target", "nld") == [
        str(Path(got) / "shards" / "target_nld" / "shard_0000.bin")]
    assert uc.materialize_shards(got, tok, log=lambda *_: None) == {}     # idempotent


def test_shipped_config_resolves_and_pins_one_revision():
    cfg = yaml.safe_load(CFG.read_text())
    spec = uc.resolve_spec(cfg)
    assert spec["name"] == "gpt2_unlearning_1m" and spec["corpus_name"] == "gpt2_corpus"
    assert spec["seed"] == 20261022
    assert {s["config"] for s in spec["sources"].values()} == {"20231101.nl", "20231101.en", "20231101.de"}
    assert {s["revision"] for s in spec["sources"].values()} == {"b04c8d1ceb2f5cd4588862100d08de323dccfbaa"}
    assert {s["sample_rate"] for s in spec["sources"].values()} == {0.02}
    assert spec["budgets"] == {"target_nld": 1000, "contrastive_eng": 500_000,
                               "contrastive_deu": 500_000, "heldout_eng": 200_000,
                               "heldout_deu": 200_000, "heldout_nld": 200_000}
    assert spec["sanitize"]["min_paragraph_chars"] == 200 and spec["sanitize"]["min_doc_chars"] == 1000
    assert spec["counterparts"]["langs"] == {"eng": "en", "deu": "de"}
    assert spec["counterparts"]["user_agent"] == uc.USER_AGENT
    bad = dict(cfg); bad["sources"] = dict(cfg["sources"]); bad["sources"]["eng"] = dict(cfg["sources"]["eng"]); bad["sources"]["eng"].pop("revision")
    with pytest.raises(ValueError, match="revision"):
        uc.resolve_spec(bad)
