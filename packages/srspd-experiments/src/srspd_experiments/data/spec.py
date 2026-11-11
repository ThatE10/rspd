"""Declarative data specs: what the target ``X`` and the retained ``Y`` of an
erasure are, as dataclasses a recipe can load from YAML.

    from srspd_experiments.data import DataSpec, build_erasure_data
    spec = DataSpec.from_yaml("my_data.yaml")
    data = build_erasure_data(tokenizer, spec)
    srspd.erase(model, target=data.target_ids, retained=data.retained_ids,
                mu_per_token=[0.01])

YAML shape (every key but ``target`` / ``retained`` optional)::

    corpora:                      # local_jsonl corpora joining the registry
      target_nld: {kind: local_jsonl, path: datasets/.../target_nld.jsonl, topic: target}
    target:                       # X: rows of registered corpora
      corpora: [bio-forget-corpus, cyber-forget-corpus]
      max_rows: 300               # rows per corpus (row corpora; required unless max_tokens)
      max_tokens: null            # an exact token budget over the fit rows, in order
      min_len: 50                 # characters; rows at or under it are skipped
      max_length: 768             # tokens per row (the window length of a chunked corpus)
      heldout_rows: 8             # rows after the fit rows (last windows) held out
      heldout_from: null          # other corpora whose first heldout_rows rows are the
                                  # forget-side readout instead
    retained:                     # Y: srspd_experiments.data.retained_corpus buckets
      buckets: [...]
      chunk_tokens: 1024
      seed: 0
      add_bos: true
      chat_template: null
    heldout_retain_corpora: [wikitext]   # retain-side readout: their last heldout_rows rows
    retain_max_length: 512
    heldout_mix_tokens: 2048      # per retained bucket, document-disjoint from Y; 0 = off
    heldout:                      # the shared baseline readout (eval.heldout); optional
      rows: 32
      max_length: 1024
      tail: []
      min_len: 0

The defaults are the ones every published SRSPD run used. A chunked corpus
(MUSE, ``chunk: True`` in the registry) is consumed as its official windows: every
window but the last ``heldout_rows`` is a fit window, and ``max_rows`` /
``max_tokens`` / ``heldout_from`` are refused for it rather than ignored.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from . import corpora as C
from . import retained_corpus as rcorp

DEFAULT_MAX_LENGTH = 768
DEFAULT_RETAIN_MAX_LENGTH = 512
DEFAULT_HELDOUT_ROWS = 8
DEFAULT_HELDOUT_MIX_TOKENS = 2048
DEFAULT_HELDOUT_RETAIN_CORPORA = ("wikitext",)


def _pos_int(v, where):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"{where} must be a positive int, got {v!r}")
    return int(v)


def _nonneg_int(v, where):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"{where} must be a non-negative int, got {v!r}")
    return int(v)


def _names(value, where) -> tuple:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where} must be a non-empty list of corpus names, got {value!r}")
    for c in value:
        if c not in C.CORPORA:
            raise ValueError(f"{where}: unknown corpus {c!r}; known: {sorted(C.CORPORA)}")
    return tuple(str(c) for c in value)


def unique_in_order(names) -> list:
    """``names`` with repeats dropped, first occurrence kept."""
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


@dataclass(frozen=True)
class TargetSpec:
    """The target set ``X`` (see the module docstring)."""
    corpora: tuple
    max_rows: int | None = None
    max_tokens: int | None = None
    min_len: int = C.DEFAULT_MIN_LEN
    max_length: int = DEFAULT_MAX_LENGTH
    heldout_rows: int = DEFAULT_HELDOUT_ROWS
    heldout_from: tuple | None = None

    @property
    def windowed(self) -> bool:
        return any(C.is_chunked_corpus(c) for c in self.corpora)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TargetSpec":
        if not isinstance(d, Mapping):
            raise ValueError(f"target must be a mapping, got {d!r}")
        unknown = set(d) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"target has unknown keys {sorted(unknown)}")
        names = _names(d.get("corpora"), "target.corpora")
        mfr = d.get("max_rows")
        mft = d.get("max_tokens")
        hf = d.get("heldout_from")
        chunked = [c for c in names if C.is_chunked_corpus(c)]
        if chunked:
            plain = [c for c in names if c not in chunked]
            if plain:
                raise ValueError(f"target.corpora mixes chunked (window) corpora {chunked} with "
                                 f"row corpora {plain}; give one kind")
            if mfr is not None or mft is not None:
                raise ValueError(f"target.corpora {chunked} are chunked: X is every official "
                                 f"window, so max_rows / max_tokens do not apply -- drop them")
            if hf is not None:
                raise ValueError("target.heldout_from does not combine with a chunked corpus: "
                                 "its readout is the last heldout_rows windows")
        elif mfr is None and mft is None:
            raise ValueError("target.max_rows (rows per corpus) or target.max_tokens (an exact "
                             "token budget over the rows in order) is required")
        mfr = None if mfr is None else _pos_int(mfr, "target.max_rows")
        mft = None if mft is None else _pos_int(mft, "target.max_tokens")
        if mfr is None and not chunked:
            streamed = [c for c in names if C.CORPORA[c].get("stream")]
            if streamed:
                raise ValueError(f"target.max_tokens alone reads every row, and {streamed} are "
                                 f"streamed (multi-GB): add target.max_rows as a row cap")
        hf = None if hf is None else _names(hf, "target.heldout_from")
        if hf and any(C.is_chunked_corpus(c) for c in hf):
            raise ValueError("target.heldout_from cannot name a chunked corpus")
        overlap = set(names) & set(hf or ())
        if overlap:
            raise ValueError(f"target: {sorted(overlap)} is both a target and a heldout_from corpus")
        return cls(corpora=names, max_rows=mfr, max_tokens=mft,
                   min_len=_nonneg_int(d.get("min_len", C.DEFAULT_MIN_LEN), "target.min_len"),
                   max_length=_pos_int(d.get("max_length", DEFAULT_MAX_LENGTH),
                                       "target.max_length"),
                   heldout_rows=_nonneg_int(d.get("heldout_rows", DEFAULT_HELDOUT_ROWS),
                                            "target.heldout_rows"),
                   heldout_from=hf)


@dataclass(frozen=True)
class RetainedSpec:
    """The retained set ``Y``: resolved buckets plus the build framing
    (:func:`srspd_experiments.data.retained_corpus.build_retained_tokens`)."""
    buckets: tuple
    chunk_tokens: int = 1024
    seed: int = 0
    add_bos: bool = True
    chat_template: str | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RetainedSpec":
        if not isinstance(d, Mapping):
            raise ValueError(f"retained must be a mapping, got {d!r}")
        unknown = set(d) - {"buckets", "chunk_tokens", "seed", "add_bos", "chat_template"}
        if unknown:
            raise ValueError(f"retained has unknown keys {sorted(unknown)}")
        spec = rcorp.resolve_spec(d.get("buckets") or [])
        ct = _pos_int(d.get("chunk_tokens", 1024), "retained.chunk_tokens")
        tmpl = d.get("chat_template")
        if tmpl not in rcorp.CHAT_TEMPLATES:
            raise ValueError(f"retained.chat_template must be one of {rcorp.CHAT_TEMPLATES}, "
                             f"got {tmpl!r}")
        return cls(buckets=tuple(spec), chunk_tokens=ct, seed=int(d.get("seed", 0)),
                   add_bos=bool(d.get("add_bos", True)), chat_template=tmpl)

    @property
    def spec(self) -> list:
        return [dict(b) for b in self.buckets]

    @property
    def build_kw(self) -> dict:
        return {"chunk_tokens": self.chunk_tokens, "seed": self.seed,
                "add_bos": self.add_bos, "chat_template": self.chat_template}


@dataclass(frozen=True)
class HeldoutSpec:
    """The shared baseline readout block (:mod:`srspd_experiments.eval.heldout`)."""
    rows: int = 32
    max_length: int = 1024
    tail: tuple = ()
    min_len: int = 0


@dataclass(frozen=True)
class DataSpec:
    """Everything a recipe needs to build ``X``, ``Y`` and the readouts."""
    target: TargetSpec
    retained: RetainedSpec
    corpora: dict = field(default_factory=dict)
    heldout_retain_corpora: tuple = DEFAULT_HELDOUT_RETAIN_CORPORA
    retain_max_length: int = DEFAULT_RETAIN_MAX_LENGTH
    heldout_mix_tokens: int = DEFAULT_HELDOUT_MIX_TOKENS
    heldout: HeldoutSpec | None = None

    KEYS = ("corpora", "target", "retained", "heldout_retain_corpora", "retain_max_length",
            "heldout_mix_tokens", "heldout")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DataSpec":
        """Validate and resolve; registers the ``corpora:`` block into
        :data:`srspd_experiments.data.corpora.CORPORA` first (so ``target`` may
        name local corpora). Every failure is a ``ValueError`` naming the key."""
        if not isinstance(d, Mapping):
            raise ValueError(f"data spec must be a mapping, got {type(d).__name__}")
        unknown = set(d) - set(cls.KEYS)
        if unknown:
            raise ValueError(f"data spec has unknown keys {sorted(unknown)}; known: {cls.KEYS}")
        for k in ("target", "retained"):
            if k not in d:
                raise ValueError(f"data spec needs `{k}:`")
        corpora = dict(d.get("corpora") or {})
        if corpora:
            C.register_config_corpora({"corpora": corpora})
        target = TargetSpec.from_dict(d["target"])
        retained = RetainedSpec.from_dict(d["retained"])
        hr = _names(d.get("heldout_retain_corpora", list(DEFAULT_HELDOUT_RETAIN_CORPORA)),
                    "heldout_retain_corpora")
        overlap = set(target.corpora) & set(hr)
        if overlap:
            raise ValueError(f"{sorted(overlap)} is both a target and a held-out retain corpus")
        hmix = _nonneg_int(d.get("heldout_mix_tokens", DEFAULT_HELDOUT_MIX_TOKENS),
                           "heldout_mix_tokens")
        if hmix % retained.chunk_tokens:
            raise ValueError(f"heldout_mix_tokens = {hmix} must be a multiple of "
                             f"retained.chunk_tokens ({retained.chunk_tokens}) so every "
                             f"held-out chunk lies inside one bucket")
        heldout = None
        if d.get("heldout") is not None:
            from ..eval.heldout import resolve_heldout_config
            H = resolve_heldout_config({"corpora": corpora, "heldout": d["heldout"]})
            heldout = HeldoutSpec(rows=H["rows"], max_length=H["max_length"],
                                  tail=tuple(H["tail"]), min_len=H["min_len"])
        return cls(target=target, retained=retained, corpora=corpora,
                   heldout_retain_corpora=hr,
                   retain_max_length=_pos_int(d.get("retain_max_length",
                                                    DEFAULT_RETAIN_MAX_LENGTH),
                                              "retain_max_length"),
                   heldout_mix_tokens=hmix, heldout=heldout)

    @classmethod
    def from_yaml(cls, path: str, *, key: str | None = None) -> "DataSpec":
        """Load from a YAML file; ``key`` selects a sub-mapping (e.g. ``data``)."""
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f)
        if key is not None:
            d = (d or {}).get(key)
        return cls.from_dict(d)

    def as_dict(self) -> dict:
        """The resolved spec as plain data (for a run manifest)."""
        out = asdict(self)
        out["retained"]["buckets"] = [dict(b) for b in self.retained.buckets]
        return out
