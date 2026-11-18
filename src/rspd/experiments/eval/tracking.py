"""Optional experiment tracking -- Weights & Biases, **off by default**.

:class:`Tracker` is a no-op unless its config says ``enabled: true``; then it
is a guarded ``wandb`` run, **offline** unless ``mode: online`` is requested
*and* ``WANDB_API_KEY`` is set (an online run without a key would hang on
login, so the fallback is decided here and the reason recorded). Every call is
wrapped: the first failure prints once, is recorded in :attr:`Tracker.resolved`
and disables further logging, so tracking can never take a run down, and no
result depends on it. ``wandb`` is not a dependency of this package; when it
is not installed an enabled tracker records why and stays inactive.

Config block (all optional)::

    tracking: {enabled: false, mode: offline, project: rspd[experiments],
               entity: null, tags: [], notes: null}
"""
from __future__ import annotations

import os

ONLINE_WITHOUT_KEY = "online requested but WANDB_API_KEY absent"
DEFAULT_PROJECT = "rspd[experiments]"


def resolve_mode(requested, environ=None) -> tuple:
    """``(mode, reason)``: ``online`` only when requested *and* ``WANDB_API_KEY``
    is set; anything else is ``offline``. The key is checked for presence only."""
    environ = os.environ if environ is None else environ
    requested = str(requested or "offline").lower()
    if requested == "online":
        if environ.get("WANDB_API_KEY"):
            return "online", None
        return "offline", ONLINE_WITHOUT_KEY
    if requested != "offline":
        return "offline", f"unknown tracking mode {requested!r}; using offline"
    return "offline", None


class Tracker:
    """Guarded tracking run; see the module docstring."""

    def __init__(self, cfg=None, *, out_dir, run_name, config=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.active = False
        self.run = None
        self._wandb = None
        self._failed = False
        mode, reason = resolve_mode(cfg.get("mode", "offline"))
        self.resolved = {"enabled": self.enabled, "active": False, "mode": mode,
                         "project": cfg.get("project") or DEFAULT_PROJECT,
                         "entity": cfg.get("entity"), "run_name": run_name, "run_dir": None,
                         "url": None, "reason": reason, "degraded": False}
        if not self.enabled:
            self.resolved["reason"] = "tracking disabled"
            return
        try:
            import wandb
        except Exception as exc:                        # not installed / broken
            self.resolved["reason"] = f"wandb import failed: {type(exc).__name__}: {exc}"
            print(f"[tracking] disabled -- {self.resolved['reason']}", flush=True)
            return
        self._wandb = wandb
        if mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        elif os.environ.get("WANDB_MODE", "").lower() in ("offline", "disabled"):
            os.environ["WANDB_MODE"] = "online"
        os.environ.setdefault("WANDB_SILENT", "true")
        try:
            os.makedirs(out_dir, exist_ok=True)
            self.run = wandb.init(project=self.resolved["project"], entity=cfg.get("entity"),
                                  name=run_name, dir=out_dir, mode=mode,
                                  tags=list(cfg.get("tags") or []), notes=cfg.get("notes"),
                                  config=config or {}, reinit=True)
            self.active = True
            url = getattr(self.run, "url", None) if mode == "online" else None
            self.resolved.update({"active": True, "run_dir": getattr(self.run, "dir", None),
                                  "run_id": getattr(self.run, "id", None),
                                  "url": url if isinstance(url, str) else None})
        except Exception as exc:
            self.resolved["reason"] = f"wandb.init failed: {type(exc).__name__}: {exc}"
            print(f"[tracking] disabled -- {self.resolved['reason']}", flush=True)
            self.run = None

    def _guard(self, what, fn):
        if not self.active or self._failed:
            return None
        try:
            return fn()
        except Exception as exc:
            self._failed = True
            self.resolved["reason"] = f"{what} failed: {type(exc).__name__}: {exc}"
            self.resolved["degraded"] = True
            print(f"[tracking] {what} failed ({type(exc).__name__}: {exc}); continuing "
                  f"without tracking", flush=True)
            return None

    def log(self, data: dict, step=None):
        return self._guard("log", lambda: self._wandb.log(dict(data), step=step))

    def summary(self, data: dict):
        def _do():
            for k, v in data.items():
                self.run.summary[k] = v
        return self._guard("summary", _do)

    def finish(self):
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:
            print(f"[tracking] finish failed ({type(exc).__name__}: {exc})", flush=True)
        finally:
            self.active = False
