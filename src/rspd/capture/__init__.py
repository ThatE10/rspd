"""Activation capture: backend-agnostic protocols (:mod:`.base`), the HF
forward-hook source (:mod:`.hf`) and Gram accumulation / planning / spills
(:mod:`.grams`)."""
from .base import ActivationSource, GramSink
from .grams import Fp64GramSink, MemorySpill, TensorSpill, gram_bytes, plan_gram_groups
from .hf import (BoundaryCache, HFHookSource, ResumeGuardError, harvest_inputs_concat,
                 harvest_inputs_native)

__all__ = ["ActivationSource", "GramSink", "Fp64GramSink", "MemorySpill", "TensorSpill",
           "gram_bytes", "plan_gram_groups", "BoundaryCache", "HFHookSource",
           "ResumeGuardError", "harvest_inputs_concat", "harvest_inputs_native"]
