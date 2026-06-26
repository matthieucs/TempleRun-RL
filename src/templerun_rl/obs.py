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
OBS_EXTRA_FIELDS = ("cell_progress", "lane", "narrow_target_lane")
OBS_FIELDS = OBS_BOOL_FIELDS + OBS_SCALAR_FIELDS + OBS_EXTRA_FIELDS
OBS_LEN = len(OBS_FIELDS)  # 22 bool/lookahead + 4 scalar + 2 extra = 28

# Lane geometry (real device units): home/center ~ -1.3, one strafe ~ 3.7 -> lanes at -5 / -1.3 / +2.4.
# CONTINUOUS lane obs (2026-06-26): the old 3-bucket {0,0.5,1} aliased the safe left lane (-5) with an
# OFF-TRACK lane (-15) -> on device the policy couldn't tell it was walking off the edge and kept
# strafing into the void. Normalizing the continuous lane to [0,1] over [LANE_OBS_MIN, LANE_OBS_MAX]
# (a bit past the fall thresholds) makes "near the edge" a distinct, recoverable obs.
LANE_HOME = -1.3
LANE_STEP = 3.7
LANE_OBS_MIN = -7.0   # just past the left fall threshold
LANE_OBS_MAX = 4.5    # just past the right fall threshold
# kept for the capture/analysis scripts that bucket lane for display.
LANE_LEFT_MAX = -3.15
LANE_RIGHT_MIN = 0.55


def lane_to_obs(lane) -> float:
    """Continuous lane -> [0,1] (center ~0.5, left lane ~0.17, right lane ~0.82, off-edge -> 0/1)."""
    if lane is None:
        lane = LANE_HOME
    return float(np.clip((float(lane) - LANE_OBS_MIN) / (LANE_OBS_MAX - LANE_OBS_MIN), 0.0, 1.0))


def narrow_lane_to_obs(ntl) -> float:
    """Map the narrow's safe lane (-1 L / 0 C / +1 R) onto the SAME continuous scale as `lane`, so the
    policy compares the two and strafes toward the target. None -> home/center (no constraint)."""
    if ntl is None:
        return lane_to_obs(LANE_HOME)
    return lane_to_obs(LANE_HOME + int(ntl) * LANE_STEP)


def path_type_obs(pt) -> float:
    """COARSE path_type category (2026-06-26) — the raw enum/37 was a sim-to-real LEAK: the device emits
    straight types the sim never did (pt 1 AND 2 for straights, + 8/9/10), so the policy saw OOD
    path_type values on device and mis-reacted (strafed off the edge). Map to 5 stable buckets that the
    sim and device AGREE on (action cues already live in is_turn/must_jump/must_slide/next_wide_gap/
    narrow_target_lane, so the fine enum is redundant): straight / turn / hole-piece / narrow / cap."""
    pt = int(pt) if pt is not None else -1
    if pt < 0:
        return 0.0
    if pt <= 2:
        return 0.16            # straight (StraightVert/Horz etc.)
    if pt <= 7:
        return 0.33            # turn / junction
    if pt <= 16:
        return 0.50            # hole / tree / holelong piece
    if pt <= 30:
        return 0.66            # narrow strip
    return 0.83                # level cap / crevasse (31-37)


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
    out[base + 1] = path_type_obs(g("path_type"))   # COARSE category (robust to device enum variety)
    out[base + 2] = _norm_enum(int(g("path_level")), PATH_LEVEL_MAX)
    out[base + 3] = _norm_enum(int(g("facing")), FACING_MAX)
    cp = g("cell_progress")
    out[base + 4] = float(np.clip(cp if cp is not None else 0.0, 0.0, 1.0))
    out[base + 5] = lane_to_obs(g("lane"))
    # narrow_target_lane: which lane to be in for the upcoming narrow (-1 L / 0 C / +1 R), mapped onto
    # the SAME continuous scale as `lane` so the policy compares them and strafes to match (None -> home).
    out[base + 6] = narrow_lane_to_obs(g("narrow_target_lane"))
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
