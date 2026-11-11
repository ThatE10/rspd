"""Where srspd-experiments reads and writes data.

One root, three trees::

    <root>/datasets/   built or downloaded corpora, token caches, unlearning sets
    <root>/logs/       evaluation results, curves, manifests
    <root>/models/     checkpoints written by the recipes

The root is, in order: the ``root`` argument of any function here, else the
``SRSPD_DATA_DIR`` environment variable, else ``~/.cache/srspd``. Every
function that touches disk elsewhere in the package takes an explicit root or
directory argument and falls back to these defaults, so a caller (a test, a
recipe, a container with a mounted volume) can redirect everything with one
environment variable.

Configs refer to local files by **data-root-relative** paths
(``datasets/gpt2_unlearning/gpt2_unlearning_1m/target_nld.jsonl``,
``models/<recipe>/<name>/checkpoint``); :func:`resolve_data_ref` turns one into
an absolute path and :func:`data_ref` goes back.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "SRSPD_DATA_DIR"
DEFAULT_ROOT = Path("~/.cache/srspd")


def data_dir(root=None, *, create: bool = True) -> Path:
    """The data root (see the module docstring), created unless ``create=False``."""
    if root is None:
        root = os.environ.get(ENV_VAR) or DEFAULT_ROOT
    p = Path(root).expanduser()
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(tree: str, parts, root, create: bool) -> Path:
    p = data_dir(root, create=create).joinpath(tree, *[str(x) for x in parts])
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def datasets_dir(*parts, root=None, create: bool = True) -> Path:
    """``<root>/datasets/<parts...>``."""
    return _sub("datasets", parts, root, create)


def logs_dir(*parts, root=None, create: bool = True) -> Path:
    """``<root>/logs/<parts...>``."""
    return _sub("logs", parts, root, create)


def models_dir(*parts, root=None, create: bool = True) -> Path:
    """``<root>/models/<parts...>``."""
    return _sub("models", parts, root, create)


def resolve_data_ref(ref: str, root=None) -> str:
    """A data-root-relative reference -> absolute path; absolute paths pass through."""
    if os.path.isabs(str(ref)):
        return str(ref)
    return str(data_dir(root, create=False) / str(ref))


def data_ref(path: str, root=None) -> str:
    """Inverse of :func:`resolve_data_ref`: an absolute path under the data root
    -> its root-relative POSIX string; a path outside the root is returned as given."""
    try:
        return Path(path).resolve().relative_to(
            data_dir(root, create=False).resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_model_path(model_path: str, root=None) -> str:
    """A local checkpoint named relative to the data root (``models/...``)
    resolves to its absolute directory when it exists; anything else (a hub id
    ``org/name``, an absolute path) is returned unchanged."""
    local = os.path.join(str(data_dir(root, create=False)), str(model_path))
    return os.path.abspath(local) if os.path.isdir(local) else str(model_path)


def hf_token() -> str | None:
    """``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) when set, else ``None``.
    Passed to every hub call so gated datasets and models resolve."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
