"""Content hash of a corpus spec (the GPT-2 corpora hash their manifests this way)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def manifest_hash(payload: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``payload`` (sorted keys, compact separators)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
