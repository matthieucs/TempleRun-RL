"""Observation encoding: raw game-state dict -> normalized float32 vector.

The policy sees a 21-element ``Box(0, 1)`` vector. We deliberately exclude the raw
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
FACING_MAX = 4.0   # PathDirection: None=0, N=1, E=2, S=3, W=4

# Observation vector layout: the bool flags (incl. 1-cell lookahead), then normalized scalars.
OBS_BOOL_FIELDS = (
    "is_dead", "is_jumping", "is_sliding", "is_on_ground", "is_stumbling",
    "has_invincibility", "has_angel_wings", "has_vacuum", "has_boost",
    "is_mega_boost", "is_strafing", "must_jump", "must_slide", "is_game_over",
    "can_turn_left", "can_turn_right", "is_turn",
    # 1-cell lookahead — anticipation (see a junction/obstacle before it's the nearest cell).
    "next_is_turn", "next_can_turn_left", "next_can_turn_right", "next_must_jump", "next_must_slide",
    # the upcoming must-jump is a 2-CELL gap (long hole / zone LevelChange) -> jump LATE (at the gap
    # edge) so airtime spans both cells; jumping 1 cell early (the single-obstacle reflex) falls in.
    "next_wide_gap",
)
OBS_SCALAR_FIELDS = ("velocity", "path_type", "path_level", "facing")
# Perception added 2026-06-19 so the NN can time turns + control its lane natively (replacing the
# host-side assists): cell_progress = longitudinal position in the nearest cell (0..1; lets the policy
# fire a turn at cell-center instead of too early), lane = which of the 3 lanes (L/C/R) the runner is in
# (for narrow strips). Both come from the hook (CharacterPlayer.GetPosition2D + PathElement.Rectangle)
# and from the sim's own state — they are OBSERVATIONS, not actions.
OBS_EXTRA_FIELDS = ("cell_progress", "lane")
OBS_FIELDS = OBS_BOOL_FIELDS + OBS_SCALAR_FIELDS + OBS_EXTRA_FIELDS
OBS_LEN = len(OBS_FIELDS)  # 22 bool/lookahead + 4 scalar + 2 extra = 28

# Lane bucketing (real geometry): home/center ~ -1.3, lanes ~3.7 apart -> LEFT if < -3.15, RIGHT if
# > 0.55, else CENTER. -> normalized {0.0 left, 0.5 center, 1.0 right}. The sim emits `lane` already in
# these real-equivalent units so both bucket identically.
LANE_LEFT_MAX = -3.15
LANE_RIGHT_MIN = 0.55


def lane_to_obs(lane) -> float:
    if lane is None:
        return 0.5
    if lane < LANE_LEFT_MAX:
        return 0.0
    if lane > LANE_RIGHT_MIN:
        return 1.0
    return 0.5


def _norm_enum(value: int, max_value: float) -> float:
    """Normalize an int enum to [0,1]; the -1 'none' sentinel maps to 0.0.

    Note: this aliases sentinel -1 with real enum value 0 at junction edges.
    Acceptable for v0 (documented in the plan); add a companion 'valid' flag if needed.
    """
    if value is None or value < 0:
        return 0.0
    return float(np.clip(value, 0.0, max_value) / max_value)


def full_obs_len(frame_stack: int = 1, action_history_len: int = 0) -> int:
    """Total obs length = ``frame_stack`` stacked 26-d frames + ``action_history_len`` one-hot
    action frames (7 each). frame_stack=1, K=0 -> the base 26-d obs."""
    return max(1, int(frame_stack)) * OBS_LEN + int(action_history_len) * C.N_ACTIONS


def make_observation_space(frame_stack: int = 1, action_history_len: int = 0) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0,
                      shape=(full_obs_len(frame_stack, action_history_len),), dtype=np.float32)


def state_to_obs(state: dict) -> np.ndarray:
    """Convert a raw state dict to the base (26-element) normalized float32 observation vector.

    Tolerant of missing keys (falls back to ``constants.STATE_DEFAULTS``) so a
    malformed/truncated JSON line never crashes a step. Action history (if any) is appended
    separately by ``obs_with_history`` — this stays 26-d for the fidelity harness/replay.
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
    cp = g("cell_progress")
    out[base + 4] = float(np.clip(cp if cp is not None else 0.0, 0.0, 1.0))
    out[base + 5] = lane_to_obs(g("lane"))
    return out


def action_history_vec(history, action_history_len: int) -> np.ndarray:
    """One-hot encode the last ``action_history_len`` actions, oldest-first, left-padded with NoOp.

    ``history`` is the running list of past actions (most recent LAST). Gives the memoryless
    MLP policy a short memory so it knows it *just* turned/jumped — curbs the "multiple turns at a
    junction" spam (the policy re-fires a turn because a single-frame obs can't see it already did).
    """
    K = int(action_history_len)
    vec = np.zeros(K * C.N_ACTIONS, dtype=np.float32)
    if K <= 0:
        return vec
    h = [int(a) for a in list(history)[-K:]]
    h = [int(C.Action.NOOP)] * (K - len(h)) + h   # left-pad so index 0 = oldest
    for i, a in enumerate(h):
        if 0 <= a < C.N_ACTIONS:
            vec[i * C.N_ACTIONS + a] = 1.0
    return vec


def obs_with_history(state: dict, history, action_history_len: int = 0) -> np.ndarray:
    """Base obs (26) optionally concatenated with the one-hot action history (K*7)."""
    base = state_to_obs(state)
    if not action_history_len:
        return base
    return np.concatenate([base, action_history_vec(history, action_history_len)])


def stacked_obs(frame_deque, history, frame_stack: int = 1, action_history_len: int = 0) -> np.ndarray:
    """Full obs = the last ``frame_stack`` base frames (oldest->newest) + one-hot action history.

    ``frame_deque`` holds recent base (26-d) obs vectors (most recent LAST); it is left-padded by
    repeating the OLDEST available frame so the vector is fixed-size from the first step. Stacking
    frames gives the MLP the temporal context (how is_turn/facing/velocity are *evolving*) it needs
    to ANTICIPATE a junction and fire the turn on the exact frame — beating the 1-frame reactive
    limit that caps single-frame turn reliability.
    """
    N = max(1, int(frame_stack))
    frames = list(frame_deque)[-N:]
    if not frames:
        frames = [np.zeros(OBS_LEN, dtype=np.float32)]
    frames = [frames[0]] * (N - len(frames)) + frames    # left-pad with the oldest frame
    parts = frames
    if action_history_len:
        parts = parts + [action_history_vec(history, action_history_len)]
    return np.concatenate(parts).astype(np.float32)
