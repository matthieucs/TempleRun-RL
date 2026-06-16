"""Observation encoding: raw game-state dict -> normalized float32 vector.

The policy sees an 18-element ``Box(0, 1)`` vector. We deliberately exclude the raw
``score`` (unbounded / accumulating -> would break MlpPolicy normalization); it feeds
the reward instead and is always available in ``info["raw_state"]``.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from . import constants as C

# Normalization constants (see docs/OFFSETS.md for value semantics).
VEL_MIN = 100.0   # run velocity starts ~100
VEL_MAX = 220.0   # ramps to ~210; 220 leaves headroom
PATH_TYPE_MAX = 37.0
PATH_LEVEL_MAX = 4.0
FACING_MAX = 3.0

# Observation vector layout (18 dims): the 14 bools, then 4 normalized scalars.
OBS_BOOL_FIELDS = (
    "is_dead", "is_jumping", "is_sliding", "is_on_ground", "is_stumbling",
    "has_invincibility", "has_angel_wings", "has_vacuum", "has_boost",
    "is_mega_boost", "is_strafing", "must_jump", "must_slide", "is_game_over",
)
OBS_SCALAR_FIELDS = ("velocity", "path_type", "path_level", "facing")
OBS_FIELDS = OBS_BOOL_FIELDS + OBS_SCALAR_FIELDS  # length 18
OBS_LEN = len(OBS_FIELDS)  # 18


def _norm_enum(value: int, max_value: float) -> float:
    """Normalize an int enum to [0,1]; the -1 'none' sentinel maps to 0.0.

    Note: this aliases sentinel -1 with real enum value 0 at junction edges.
    Acceptable for v0 (documented in the plan); add a companion 'valid' flag if needed.
    """
    if value is None or value < 0:
        return 0.0
    return float(np.clip(value, 0.0, max_value) / max_value)


def make_observation_space() -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0, shape=(OBS_LEN,), dtype=np.float32)


def state_to_obs(state: dict) -> np.ndarray:
    """Convert a raw state dict to the normalized float32 observation vector.

    Tolerant of missing keys (falls back to ``constants.STATE_DEFAULTS``) so a
    malformed/truncated JSON line never crashes a step.
    """
    g = lambda k: state.get(k, C.STATE_DEFAULTS.get(k))  # noqa: E731
    out = np.empty(OBS_LEN, dtype=np.float32)
    for i, f in enumerate(OBS_BOOL_FIELDS):
        out[i] = 1.0 if g(f) else 0.0
    base = len(OBS_BOOL_FIELDS)
    v = float(g("velocity") or 0.0)
    out[base + 0] = float(np.clip((v - VEL_MIN) / (VEL_MAX - VEL_MIN), 0.0, 1.0))
    out[base + 1] = _norm_enum(int(g("path_type")), PATH_TYPE_MAX)
    out[base + 2] = _norm_enum(int(g("path_level")), PATH_LEVEL_MAX)
    out[base + 3] = _norm_enum(int(g("facing")), FACING_MAX)
    return out
