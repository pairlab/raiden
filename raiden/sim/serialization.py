"""Portable JSON values for Isaac state, event, and metadata records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, values) -> None:
    with path.open("w") as stream:
        for value in values:
            stream.write(json.dumps(jsonable(value), separators=(",", ":"), allow_nan=False) + "\n")
