"""Model-side machinery: layer selection, quirks, the HF harness, inspection,
pristine-weight store and checkpoint export. Importing this package needs only
``torch``; ``transformers`` / ``safetensors`` are imported where used."""
from .harness import (HFLinearHarness, as_dout_din, group_by_input, input_key, is_conv1d,
                      key_layer, write_dout_din)
from .inspect import inspect_model
from .pristine import PristineStore
from .quirks import QUIRKS, Quirk, detect_quirks, register_quirk
from .selector import PRESETS, LayerSelector, build_layers_regex, resolve_selector

__all__ = ["HFLinearHarness", "as_dout_din", "write_dout_din", "is_conv1d", "input_key",
           "key_layer", "group_by_input", "inspect_model", "PristineStore", "QUIRKS", "Quirk",
           "detect_quirks", "register_quirk", "PRESETS", "LayerSelector", "build_layers_regex",
           "resolve_selector"]
