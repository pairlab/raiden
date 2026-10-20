"""Reset pose shared by teleoperation and attended replay."""

import hashlib
from pathlib import Path

import numpy as np

FOLLOWER_HOME_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def canonical_home():
    source = Path(__file__).resolve()
    return FOLLOWER_HOME_POS.copy(), {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "symbol": "FOLLOWER_HOME_POS",
    }
