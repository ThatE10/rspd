"""Capture protocols: where activations come from, and where they go.

An :class:`ActivationSource` runs a model over token sequences and hands the
INPUT activations of selected projections, one shared input at a time, to a
:class:`GramSink`. The sink only ever sees ``(input_key, x)`` with ``x`` a
``[tokens, d_in]`` tensor, so it does not depend on how the activations were
produced: the HF forward-hook source (:class:`rspd.capture.hf.HFHookSource`)
today, an inference-server backend later, both feeding the same
:class:`rspd.capture.grams.Fp64GramSink`.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Protocol, runtime_checkable

import torch

__all__ = ["GramSink", "ActivationSource"]


@runtime_checkable
class GramSink(Protocol):
    """Receives activations; owns the accumulation."""

    def add(self, key: str, x: torch.Tensor) -> None:
        """Accumulate ``x`` (``[tokens, d_in]``, any float dtype/device) into
        the Gram of input ``key``. Must not keep a reference to ``x``."""

    def result(self) -> Mapping[str, tuple]:
        """``{key: (G float64 [d_in, d_in], n_tokens)}``."""


@runtime_checkable
class ActivationSource(Protocol):
    """Streams sequences through a model, feeding a :class:`GramSink`."""

    def accumulate(self, sequences: Iterable, keys_to_rep: Mapping[str, str],
                   sink: GramSink, **options) -> dict:
        """Forward every item of ``sequences`` once; for each input key, call
        ``sink.add(key, x)`` with the input of its representative matrix
        ``keys_to_rep[key]``. Returns run statistics (a JSON-serialisable dict)."""
