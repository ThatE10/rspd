"""Discover, harvest, edit and restore the target linear weights of an HF model.

The harness hooks each selected projection directly:

* the hook's *input* is the layer's activation (``[tokens, d_in]`` after
  flattening) -- already post-norm, because HF applies the norm before the
  projection;
* the hook's *output* is the response (``[tokens, d_out]``).

Every weight is exposed as ``[d_out, d_in]`` (the ``nn.Linear.weight``
layout). ``transformers``' ``Conv1D`` (GPT-2) stores the transpose; that
difference is confined to :func:`as_dout_din` / :func:`write_dout_din`
(quirk ``conv1d_transposed``). A projection with a bias breaks the solve's
assumption ``y = x W^T``: ``Conv1D`` always has one and it is subtracted at
harvest; a biased ``nn.Linear`` is refused unless ``subtract_bias=True``
(quirk ``biased_projection``). The edit never touches a bias.

Matrices are named ``L{block:02d}.{tail}`` (:mod:`srspd.models.selector`).
Several projections read the same input (q/k/v the post-norm attention input,
gate/up the post-norm MLP input): :func:`input_key` names that shared input
(``L05.attn_in``), so one Gram serves all of them; the sharing is verified
bitwise at capture (quirk ``shared_input_check``).
"""
from __future__ import annotations

import re
import sys

import torch

from .selector import DEFAULT_EXCLUDE_REGEX, TAIL_ORDER, LayerSelector, resolve_selector

__all__ = ["HFLinearHarness", "is_conv1d", "as_dout_din", "write_dout_din", "input_key",
           "key_layer", "group_by_input", "name_chunks", "stream_chunk_inputs"]


def is_conv1d(mod) -> bool:
    """``transformers.pytorch_utils.Conv1D`` (GPT-2's projection class); matched by
    class name as a fallback, so this works without ``transformers`` installed
    and across versions that move the class."""
    if type(mod).__name__ == "Conv1D":
        return True
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:
        return False
    return isinstance(mod, Conv1D)


def as_dout_din(mod) -> torch.Tensor:
    """The weight as ``[d_out, d_in]``. ``nn.Linear`` stores that layout;
    ``Conv1D`` stores ``[d_in, d_out]`` (its forward is ``x @ W``). The result
    ALIASES the parameter (a view), it is not a snapshot."""
    w = mod.weight.detach()
    return w.T if is_conv1d(mod) else w


def write_dout_din(mod, value: torch.Tensor) -> None:
    """Write a ``[d_out, d_in]`` matrix into the module's native layout and dtype."""
    p = mod.weight
    v = value.T if is_conv1d(mod) else value
    p.data.copy_(v.to(p.device, p.dtype))


#: Matrix tail -> the name of the input it reads. A tail not listed maps to
#: itself (a matrix with an input of its own).
_INPUT_OF_TAIL = {
    "self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in",
    "self_attn.v_proj": "attn_in", "self_attn.o_proj": "o_in",
    "mlp.gate_proj": "mlp_in", "mlp.up_proj": "mlp_in", "mlp.down_proj": "down_in",
}


def input_key(name: str) -> str:
    """Matrix name -> the key of its INPUT: ``L05.self_attn.{q,k,v}_proj`` ->
    ``L05.attn_in``; ``gate_proj`` / ``up_proj`` -> ``L05.mlp_in``; ``o_proj``
    -> ``L05.o_in``; ``down_proj`` -> ``L05.down_in``; anything else -> the
    name itself."""
    if "." not in name:
        return name
    layer, tail = name.split(".", 1)
    sub = _INPUT_OF_TAIL.get(tail)
    return f"{layer}.{sub}" if sub else name


def key_layer(key: str) -> int:
    """``L05.attn_in`` -> 5 (-1 if the key carries no ``L<nn>`` prefix)."""
    m = re.match(r"^L(\d+)\.", key)
    return int(m.group(1)) if m else -1


def group_by_input(names) -> dict:
    """``{input key: [member names in the given order]}``, keys in first-seen order."""
    out: dict = {}
    for n in names:
        out.setdefault(input_key(n), []).append(n)
    return out


class HFLinearHarness:
    """The selected projections of ``model``, by name.

    Attributes
    ----------
    model : the wrapped model.
    module_by_name : ``{name: module}``; ``path_by_name`` : ``{name: module path}``.
    all_names : names ordered by block, then forward order within a block.
    excluded_paths : module paths that matched the layers regex but were removed
        by the exclusion regex (evidence for quirk ``multimodal_towers``).
    selector : the :class:`~srspd.models.selector.LayerSelector` used.
    """

    def __init__(self, model, layers_regex: str | None = None,
                 exclude_regex: str | None = None, *, subtract_bias: bool = False,
                 expect_d_in=None, selector: LayerSelector | None = None):
        """
        Parameters
        ----------
        model : an HF ``PreTrainedModel`` (or any ``nn.Module``).
        layers_regex, exclude_regex : explicit regexes (group 1 = block index,
            group 2 = matrix tail); default: the selector's (``selector=None``
            and no regex -> the auto preset).
        subtract_bias : accept biased ``nn.Linear`` projections; the bias is
            subtracted from the response at harvest and never edited.
        expect_d_in : ``{tail: d_in}`` widths to assert after discovery (the
            discovery guard for a model whose config has no ``hidden_size``).
        selector : a :class:`LayerSelector` (``layers_regex`` / ``exclude_regex``
            override its regexes when given).
        """
        self.model = model
        self.subtract_bias = bool(subtract_bias)
        if selector is None:
            selector = (resolve_selector(model, None) if layers_regex is None
                        else LayerSelector(layers_regex, DEFAULT_EXCLUDE_REGEX))
        if layers_regex is not None or exclude_regex is not None:
            selector = LayerSelector(
                layers_regex if layers_regex is not None else selector.layers_regex,
                exclude_regex if exclude_regex is not None else selector.exclude_regex,
                types=selector.types, blocks=selector.blocks, preset=selector.preset)
        self.selector = selector
        lre = re.compile(selector.layers_regex)
        xre = re.compile(selector.exclude_regex) if selector.exclude_regex else None

        self.module_by_name: dict = {}
        self.path_by_name: dict = {}
        self.excluded_paths: list = []
        self.biased_names: list = []
        for path, mod in model.named_modules():
            m = lre.search(path)
            if m is None:
                continue
            if xre is not None and xre.search(path):
                self.excluded_paths.append(path)
                continue
            layer, tail = int(m.group(1)), m.group(2)
            if not selector.keep(layer, tail):
                continue
            name = f"L{layer:02d}.{tail}"
            if not (isinstance(mod, torch.nn.Linear) or is_conv1d(mod)):
                raise TypeError(f"{path} matched the layers regex but is "
                                f"{type(mod).__name__}, not nn.Linear/Conv1D; tighten the "
                                f"regex so group 2 ends at a projection")
            if getattr(mod, "bias", None) is not None and not is_conv1d(mod):
                if not subtract_bias:
                    raise ValueError(f"{path} has a bias; the solve assumes a pure matmul "
                                     f"(y = x W^T). Pass subtract_bias=True to edit the linear "
                                     f"map and subtract the constant at harvest (the bias is "
                                     f"never edited)")
                self.biased_names.append(name)
            if name in self.module_by_name:
                raise ValueError(f"duplicate matrix name {name}: {path} vs "
                                 f"{self.path_by_name[name]} -- a second tower is leaking "
                                 f"through; tighten layers_regex / exclude_regex")
            self.module_by_name[name] = mod
            self.path_by_name[name] = path

        if not self.module_by_name:
            raise ValueError("no target projections discovered: check layers_regex / "
                             "exclude_regex / types / blocks against model.named_modules() "
                             f"(selector {selector.as_dict()})")

        def _key(name: str):
            layer, tail = name.split(".", 1)
            return (int(layer[1:]), TAIL_ORDER.index(tail) if tail in TAIL_ORDER
                    else len(TAIL_ORDER))
        self.all_names = sorted(self.module_by_name, key=_key)

        # cross-check discovery against the text tower's width
        cfg = getattr(getattr(model, "config", None), "text_config", None) \
            or getattr(model, "config", None)
        hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
        if isinstance(hidden, int):
            bad = [n for n in self.all_names if n.endswith("self_attn.q_proj")
                   and int(as_dout_din(self.module_by_name[n]).shape[1]) != hidden]
            if bad:
                raise ValueError(f"q_proj d_in != text hidden_size ({hidden}) for {bad[:4]}: "
                                 f"the layers regex is matching a non-text tower; adjust "
                                 f"exclude_regex")
        if expect_d_in:
            wrong = []
            for n in self.all_names:
                want = expect_d_in.get(n.split(".", 1)[1])
                if want is None:
                    continue
                got = int(as_dout_din(self.module_by_name[n]).shape[1])
                if got != int(want):
                    wrong.append((n, got, int(want)))
            if wrong:
                raise ValueError(f"expect_d_in mismatch for {len(wrong)} matrices (name, got, "
                                 f"want): {wrong[:4]} -- the layers regex is matching modules "
                                 f"it was not pointed at")
        self._backup: dict = {}

    # ------------------------------------------------------------------ #
    @property
    def n_blocks(self) -> int:
        return len({n.split(".")[0] for n in self.all_names})

    def d_in(self, name: str) -> int:
        return int(as_dout_din(self.module_by_name[name]).shape[1])

    def d_out(self, name: str) -> int:
        return int(as_dout_din(self.module_by_name[name]).shape[0])

    def input_groups(self) -> dict:
        """``{input key: [member names]}`` over :attr:`all_names`."""
        return group_by_input(self.all_names)

    # ------------------------------------------------------------------ #
    # capture                                                             #
    # ------------------------------------------------------------------ #
    def harvest(self, token_batches, names=None, label: str = "", verbose: bool = False):
        """Run the model over ``token_batches`` capturing ``(A, T)`` per matrix.

        ``token_batches``: iterable of ``LongTensor`` input ids (``[B, T]``, on
        the model's device) or kwargs dicts for ``model(**batch)``. Returns
        ``{name: (A [tokens, d_in], T [tokens, d_out])}``, float32 on CPU,
        tokens concatenated across batches, the bias (if any) subtracted from
        ``T`` so that ``A @ W.T == T``. For checks and small probes; the
        erasure itself uses input-only capture (:mod:`srspd.capture.hf`).
        """
        names = list(names if names is not None else self.all_names)
        store = {n: ([], []) for n in names}

        def make_hook(n):
            def hook(mod, inputs, output):
                out = output.detach()
                b = getattr(mod, "bias", None)
                if b is not None:
                    out = out - b.detach()
                store[n][0].append(inputs[0].detach().to("cpu", torch.float32).flatten(0, -2))
                store[n][1].append(out.to("cpu", torch.float32).flatten(0, -2))
            return hook

        handles = [self.module_by_name[n].register_forward_hook(make_hook(n)) for n in names]
        try:
            with torch.no_grad():
                for batch in token_batches:
                    if isinstance(batch, torch.Tensor):
                        self.model(input_ids=batch)
                    else:
                        self.model(**batch)
        finally:
            for h in handles:
                h.remove()
        data = {n: (torch.cat(store[n][0], 0), torch.cat(store[n][1], 0)) for n in names}
        if verbose:
            n_tok = next(iter(data.values()))[0].shape[0] if data else 0
            print(f"[harvest{' ' + label if label else ''}] {n_tok} tokens, {len(names)} "
                  f"matrices", file=sys.stderr)
        return data

    def verify_bracketing(self, data, label: str = "", rel_tol: float = 1e-3) -> dict:
        """Assert ``A @ W.T ~= T`` for every harvested matrix: proves the hooks
        bracket exactly the matmul being edited. A bf16 forward checked in
        float32 needs ``rel_tol`` ~1e-2. Returns ``{name: rel err}``."""
        errs = {}
        for n, (A, T) in data.items():
            W = as_dout_din(self.module_by_name[n]).to("cpu", torch.float32)
            errs[n] = ((A @ W.T - T).norm() / T.norm().clamp_min(1e-12)).item()
        bad = sorted(((n, e) for n, e in errs.items() if e > rel_tol), key=lambda kv: -kv[1])
        if bad:
            raise AssertionError(f"bracketing{' ' + label if label else ''} violated at "
                                 f"rel_tol={rel_tol} for {len(bad)} matrices; worst: {bad[:5]}")
        return errs

    # ------------------------------------------------------------------ #
    # edit / restore (in-memory backup; see srspd.models.pristine for disk) #
    # ------------------------------------------------------------------ #
    def get_weight(self, name: str) -> torch.Tensor:
        """The live weight ``[d_out, d_in]`` (model's device and dtype). ALIASES the
        parameter -- clone it for a before/after comparison."""
        return as_dout_din(self.module_by_name[name])

    def apply_weights(self, names, source):
        """Copy ``source[name]`` (``[d_out, d_in]``) into the live parameters,
        backing up each matrix (native layout, CPU) before its first write."""
        with torch.no_grad():
            for n in names:
                mod = self.module_by_name[n]
                if n not in self._backup:
                    self._backup[n] = mod.weight.detach().clone().cpu()
                write_dout_din(mod, source[n])

    def restore_all(self):
        """Restore every edited matrix from its backup (idempotent); returns the names."""
        with torch.no_grad():
            for n, w in self._backup.items():
                p = self.module_by_name[n].weight
                p.data.copy_(w.to(p.device, p.dtype))
        return sorted(self._backup)

    @property
    def edited_names(self):
        return sorted(self._backup)


def name_chunks(harness: HFLinearHarness, chunk_layers):
    """Partition ``all_names`` into groups of ``chunk_layers`` blocks (or one group)."""
    if not chunk_layers:
        return [list(harness.all_names)]
    by_layer: dict = {}
    for n in harness.all_names:
        by_layer.setdefault(int(n.split(".")[0][1:]), []).append(n)
    layers = sorted(by_layer)
    return [sum((by_layer[l] for l in layers[i:i + chunk_layers]), [])
            for i in range(0, len(layers), chunk_layers)]


def stream_chunk_inputs(harness, ids, names, *, device, dtype=torch.float32) -> dict:
    """One forward pass, returning ``{name: X [n, d_in]}`` on ``device`` in ``dtype``.

    Input-only hooks through ``harness.module_by_name``; ``ids`` is a token
    tensor (forwarded as ``input_ids=``) or a kwargs dict (``model(**ids)``,
    for a model with no token input). Hooks are removed even if the forward
    raises.
    """
    store: dict = {}

    def make_hook(n):
        def hook(_mod, inputs, _output):
            store[n] = inputs[0].detach().flatten(0, -2).to(device=device, dtype=dtype)
        return hook

    handles = [harness.module_by_name[n].register_forward_hook(make_hook(n)) for n in names]
    try:
        with torch.no_grad():
            if isinstance(ids, torch.Tensor):
                harness.model(input_ids=ids)
            else:
                harness.model(**ids)
    finally:
        for h in handles:
            h.remove()
    return store
