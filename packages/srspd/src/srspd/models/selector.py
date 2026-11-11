"""Which linear layers an erasure edits: the layer selector.

A selector is a regex over ``model.named_modules()`` paths with two capture
groups -- group 1 the block index, group 2 the matrix *tail* (the path after
``<block>.``) -- plus an exclusion regex (other towers of a multimodal
wrapper), an optional filter on matrix types and an optional set of block
indices. Matrices are named ``L{block:02d}.{tail}`` (``L03.self_attn.q_proj``,
``L00.attn.c_attn``); the *type* of a matrix is the last component of its
tail (``q_proj``, ``c_fc``).

Presets (:data:`PRESETS`):

``decoder``
    The seven projections of a standard decoder block
    (``...layers.<i>.self_attn.{q,k,v,o}_proj`` and
    ``...layers.<i>.mlp.{gate,up,down}_proj``), the module naming of the common
    HF decoder architectures, with or without a multimodal wrapper (whose
    vision and audio towers are removed by :data:`DEFAULT_EXCLUDE_REGEX`).
``gpt2``
    GPT-2's four ``Conv1D`` projections per block (``h.<i>.attn.c_attn`` --
    the fused QKV, kept whole -- ``h.<i>.attn.c_proj``, ``h.<i>.mlp.c_fc``,
    ``h.<i>.mlp.c_proj``).

:func:`resolve_selector` turns the user-facing ``layers=`` argument into a
:class:`LayerSelector`; ``layers=None`` picks the first preset that matches at
least one module of the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = ["DEFAULT_LAYERS_REGEX", "DEFAULT_EXCLUDE_REGEX", "GPT2_LAYERS_REGEX", "PRESETS",
           "TAIL_ORDER", "LayerSelector", "build_layers_regex", "resolve_selector",
           "matrix_type"]

#: Text-decoder projections: 7 matrices per block (q, k, v, o, gate, up, down).
DEFAULT_LAYERS_REGEX = (
    r"(?:^|\.)layers\.(\d+)\."
    r"(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)

#: Multimodal wrappers keep their own transformer towers whose module paths
#: also match the layers regex; they are excluded.
DEFAULT_EXCLUDE_REGEX = r"vision|audio|multi_modal"

#: GPT-2's four Conv1D projections per block; the fused ``c_attn`` is one matrix.
GPT2_LAYERS_REGEX = r"(?:^|\.)h\.(\d+)\.(attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))$"

#: ``{preset: (layers_regex, exclude_regex)}``, tried in this order by
#: :func:`resolve_selector` when ``layers=None``.
PRESETS = {
    "decoder": (DEFAULT_LAYERS_REGEX, DEFAULT_EXCLUDE_REGEX),
    "gpt2": (GPT2_LAYERS_REGEX, ""),
}

#: Sort order of the matrices of one block: the order the forward applies them.
#: A tail absent from this list still sorts, last within its block.
TAIL_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj",
]


def matrix_type(name: str) -> str:
    """``L03.self_attn.q_proj`` -> ``q_proj``; ``L00.mlp.c_proj`` -> ``c_proj``."""
    return name.split(".")[-1]


def build_layers_regex(indices: Iterable[int], base: str = DEFAULT_LAYERS_REGEX) -> str:
    """``base`` with its block-index group ``(\\d+)`` narrowed to exactly
    ``indices``. Same groups (1 = block, 2 = tail) and anchor; the ``\\.``
    after the group keeps ``layers.1.`` from matching ``layers.17.``."""
    idx = sorted({int(i) for i in indices})
    if not idx or any(i < 0 for i in idx):
        raise ValueError(f"need a nonempty list of non-negative block indices; got {indices!r}")
    if r"(\d+)" not in base:
        raise ValueError(f"the base regex has no (\\d+) block-index group: {base!r}")
    alt = "|".join(str(i) for i in idx)
    return base.replace(r"(\d+)", f"({alt})", 1)


@dataclass(frozen=True)
class LayerSelector:
    """A resolved selection rule.

    layers_regex : regex with group 1 = block index, group 2 = matrix tail.
    exclude_regex : module paths matching it are skipped ("" = exclude nothing).
    types : matrix types (last tail component, e.g. ``"down_proj"``) or whole
        tails to keep; ``None`` keeps every match.
    blocks : block indices to keep; ``None`` keeps every block.
    preset : the preset this came from, if any (reported only).
    """
    layers_regex: str = DEFAULT_LAYERS_REGEX
    exclude_regex: str = DEFAULT_EXCLUDE_REGEX
    types: tuple | None = None
    blocks: tuple | None = None
    preset: str | None = None
    notes: tuple = field(default=(), compare=False)

    def __post_init__(self):
        try:
            rx = re.compile(self.layers_regex)
        except re.error as e:
            raise ValueError(f"layers_regex does not compile: {e}") from None
        if rx.groups < 2:
            raise ValueError(f"layers_regex needs two capture groups (block index, matrix "
                             f"tail); {self.layers_regex!r} has {rx.groups}")
        if self.exclude_regex:
            re.compile(self.exclude_regex)

    def keep(self, block: int, tail: str) -> bool:
        if self.blocks is not None and int(block) not in self.blocks:
            return False
        if self.types is not None and tail not in self.types \
                and tail.split(".")[-1] not in self.types:
            return False
        return True

    def as_dict(self) -> dict:
        return {"layers_regex": self.layers_regex, "exclude_regex": self.exclude_regex,
                "types": None if self.types is None else list(self.types),
                "blocks": None if self.blocks is None else list(self.blocks),
                "preset": self.preset}


def _matches(model, regex, exclude) -> int:
    lre = re.compile(regex)
    xre = re.compile(exclude) if exclude else None
    n = 0
    for path, _mod in model.named_modules():
        if xre is not None and xre.search(path):
            continue
        if lre.search(path):
            n += 1
    return n


def _auto_preset(model) -> str:
    for name, (rx, ex) in PRESETS.items():
        if _matches(model, rx, ex):
            return name
    raise ValueError("no layer-selector preset matches this model "
                     f"(tried {list(PRESETS)}); pass layers= a regex with groups "
                     "(block index, matrix tail), e.g. "
                     r"r'(?:^|\.)blocks\.(\d+)\.(attn\.qkv|mlp\.fc1)$'")


def _tuple_or_none(v, what):
    if v is None:
        return None
    if isinstance(v, (str, int)):
        v = [v]
    if not isinstance(v, (list, tuple, set, frozenset)):
        raise ValueError(f"{what} must be a list, got {v!r}")
    return tuple(v)


def resolve_selector(model, layers=None) -> LayerSelector:
    """The :class:`LayerSelector` for ``layers``:

    * ``None`` -- the first preset of :data:`PRESETS` matching the model;
    * a preset name (``"decoder"``, ``"gpt2"``);
    * any other string -- a ``layers_regex`` (default exclusion kept);
    * a sequence of ints -- those block indices of the auto preset;
    * a mapping with optional keys ``preset``, ``regex``, ``exclude``,
      ``types`` (matrix types or tails), ``blocks`` (ints);
    * a :class:`LayerSelector` (returned unchanged).
    """
    if isinstance(layers, LayerSelector):
        return layers
    if layers is None:
        p = _auto_preset(model)
        rx, ex = PRESETS[p]
        return LayerSelector(rx, ex, preset=p)
    if isinstance(layers, str):
        if layers in PRESETS:
            rx, ex = PRESETS[layers]
            return LayerSelector(rx, ex, preset=layers)
        return LayerSelector(layers, DEFAULT_EXCLUDE_REGEX)
    if isinstance(layers, Mapping):
        unknown = set(layers) - {"preset", "regex", "exclude", "types", "blocks"}
        if unknown:
            raise ValueError(f"unknown layer-selector keys {sorted(unknown)}; allowed: "
                             "preset, regex, exclude, types, blocks")
        preset = layers.get("preset")
        if preset is not None and preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; one of {list(PRESETS)}")
        if layers.get("regex") is None and preset is None:
            preset = _auto_preset(model)
        rx, ex = PRESETS[preset] if preset is not None else (layers["regex"],
                                                             DEFAULT_EXCLUDE_REGEX)
        if layers.get("regex") is not None:
            rx = layers["regex"]
        if "exclude" in layers:
            ex = layers["exclude"] or ""
        blocks = _tuple_or_none(layers.get("blocks"), "blocks")
        if blocks is not None:
            blocks = tuple(sorted({int(b) for b in blocks}))
        return LayerSelector(rx, ex, types=_tuple_or_none(layers.get("types"), "types"),
                             blocks=blocks, preset=preset)
    if isinstance(layers, Sequence) and all(isinstance(i, int) and not isinstance(i, bool)
                                            for i in layers):
        p = _auto_preset(model)
        rx, ex = PRESETS[p]
        return LayerSelector(rx, ex, blocks=tuple(sorted({int(i) for i in layers})), preset=p)
    raise ValueError(f"layers must be None, a preset name, a regex, a list of block indices, "
                     f"a mapping or a LayerSelector; got {layers!r}")
