"""rspd: Representation Space-aware Parameter Decomposition. Localize features within
the parameter space: find the component of a model's weights that one behaviour uses and
another does not, in closed form, and subtract it.

    from rspd import SRSPD

    srspd = SRSPD(model, tokenizer)
    components = srspd.fit(target=dutch, retained=english, mu_per_token=1.0)
    components.apply()        # W <- W - M
    components.restore()

Convention (fixed): ``X`` is the TARGET set (removed), ``Y`` the RETAINED set
(kept); ``W`` a weight ``[d_out, d_in]``. The component is

    M = W X^T X (X^T X + mu Y^T Y + delta I)^+

and the edit is always ``W_new = W - M``. The dose is ``mu_per_token``.

Public names: :class:`SRSPD` (the supervised variant), :class:`Component` and
:class:`Components` (what a fit returns), :func:`inspect_model`,
:func:`export_checkpoint`. The tensor-level mathematics is :mod:`rspd.core`;
model handling is :mod:`rspd.models` and :mod:`rspd.capture`; the function the
class wraps is :func:`rspd.api.erase`. Only ``torch`` is needed to import the
package; ``transformers`` and ``safetensors`` (the ``[hf]`` extra) are imported
where used. Command line: ``rspd inspect | erase | report``.
"""
from __future__ import annotations

from . import api, core
from .core.types import Component, Components
from .models.export import export_checkpoint
from .models.inspect import inspect_model
from .srspd import SRSPD

__version__ = "0.0.1"

__all__ = ["SRSPD", "Component", "Components", "inspect_model", "export_checkpoint",
           "__version__"]
