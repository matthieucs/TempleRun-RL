"""Observation encoding: raw game-state dict -> normalized float32 vector.

The policy sees an OBS_LEN-element ``Box(0, 1)`` vector (see OBS_FIELDS; 32 as of 2026-07-03). We deliberately exclude the raw
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
    # the upcoming must-jump is a wide gap -> jump LATE. next_ground_change_higher distinguishes a
    # 1-cell step-UP crevasse (raised ledge, needs a LATE takeoff, device probe 2026-07-03) from a
    # 2-cell long hole (needs an EARLIER takeoff to span both cells) — different jump timings.
    "next_wide_gap", "next_ground_change_higher",
)
OBS_SCALAR_FIELDS = ("velocity", "path_type", "path_level", "facing")
# Perception added 2026-06-19 so the NN can time turns + control its lane natively (replacing the
# host-side assists): cell_progress = longitudinal position in the nearest cell (0..1; lets the policy
# fire a turn at cell-center instead of too early), lane = which of the 3 lanes (L/C/R) the runner is in
# (for narrow strips). Both come from the hook (CharacterPlayer.GetPosition2D + PathElement.Rectangle)
# and from the sim's own state — they are OBSERVATIONS, not actions.
OBS_EXTRA_FIELDS = ("cell_progress", "lane", "narrow_target_lane", "predicted_landing")
OBS_FIELDS = OBS_BOOL_FIELDS + OBS_SCALAR_FIELDS + OBS_EXTRA_FIELDS
OBS_LEN = len(OBS_FIELDS)  # 24 bool/lookahead + 4 scalar + 4 extra = 32 (keep derived from OBS_FIELDS)

# Lane geometry (real device units, 2026-07-01 tilt-controlled model): the lateral position is
# CONTINUOUS and centered on 0 — center 0, one strafe ~ 3.7 -> lanes at -3.7 / 0 / +3.7, side walls
# at ~+-7. Normalizing over a SYMMETRIC [LANE_OBS_MIN, LANE_OBS_MAX] centered on 0 maps center -> 0.5
# and the edges +-7 -> ~0.06/0.94, so "near which wall" is a distinct, recoverable obs (the old
# asymmetric -7..+4.5 range was for the retired home=-1.3 lock-lane model).
LANE_HOME = 0.0
LANE_STEP = 3.7
NARROW_CENTER = 4.5   # [MEASURED 2026-07-02] long-narrow strip center (Region2D bands [2,7]/[3,7])
LANE_OBS_MIN = -8.0   # symmetric, just past the left wall
LANE_OBS_MAX = 8.0    # symmetric, just past the right wall
# kept for the capture/analysis scripts that bucket lane for display (center-0 lane boundaries).
LANE_LEFT_MAX = -1.85
LANE_RIGHT_MIN = 1.85

# PREDICTED-LANDING geometry (2026-07-03, v10): a turn swiped NOW executes ~1 frame (100 ms) LATER
# (device-measured: constant 1-frame lag, 21/21 turns), during which cell_progress advances
# velocity*dt/cell_len. So the runner LANDS at cell_progress + DCPROG_PER_VEL*velocity, and the landing
# offset = (that - 0.44)*45. Feeding this projected-landing cprog as an obs lets the policy aim the
# LANDING (not the swipe) at 0.44 at ANY speed, with no internal velocity math -> fixes the late-swipe
# off-center turn LANDING (dominant device death, CLAUDE.md §21). Device fit slope 0.0016 ~= sim
# dt/cell_len 0.1/60; one shared constant keeps sim-training and device-deploy consistent (<5% on the
# device cell_len, absorbed by DR).
DCPROG_PER_VEL = 0.1 / 60.0   # per-frame cprog advance per unit velocity (= step_dt / cell_length)


def lane_to_obs(lane) -> float:
    """Continuous lane -> [0,1] (center 0 -> 0.5, left wall -7 -> ~0.06, right wall +7 -> ~0.94)."""
    if lane is None:
        lane = 0.0
    return float(np.clip((float(lane) - LANE_OBS_MIN) / (LANE_OBS_MAX - LANE_OBS_MIN), 0.0, 1.0))


def narrow_lane_to_obs(ntl) -> float:
    """Map the narrow's safe lane (-1 L / 0 C / +1 R) onto the SAME continuous (center-0) scale as
    `lane`, so the policy compares the two and strafes toward the target. None -> center (no
    constraint). Target = the MEASURED strip center +/-4.5 (Region2D bands), not the coin-lane step."""
    if ntl is None:
        return lane_to_obs(0.0)
    return lane_to_obs(int(ntl) * NARROW_CENTER)


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
    """Total obs length = ``frame_stack`` stacked OBS_LEN-d frames + ``action_history_len`` one-hot
    action frames (7 each). frame_stack=1, K=0 -> the base OBS_LEN-d obs."""
    return max(1, int(frame_stack)) * OBS_LEN + int(action_history_len) * C.N_ACTIONS


def make_observation_space(frame_stack: int = 1, action_history_len: int = 0) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0,
                      shape=(full_obs_len(frame_stack, action_history_len),), dtype=np.float32)


def state_to_obs(state: dict) -> np.ndarray:
    """Convert a raw state dict to the base (OBS_LEN=31) normalized float32 observation vector.

    Tolerant of missing keys (falls back to ``constants.STATE_DEFAULTS``) so a
    malformed/truncated JSON line never crashes a step. Action history (if any) is appended
    separately by ``obs_with_history`` — this stays OBS_LEN-d for the fidelity harness/replay.
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
    # predicted_landing: where a turn swiped THIS frame would LAND (in cell_progress), accounting for the
    # ~1-frame exec latency. The policy learns to swipe when this ~= 0.44 (centered) -> speed-invariant
    # turn timing. Derived host-side from the already-streamed cell_progress + velocity (no hook change).
    cp0 = cp if cp is not None else 0.0
    out[base + 7] = float(np.clip(cp0 + DCPROG_PER_VEL * v, 0.0, 1.0))
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
    """Base obs (OBS_LEN) optionally concatenated with the one-hot action history (K*7)."""
    base = state_to_obs(state)
    if not action_history_len:
        return base
    return np.concatenate([base, action_history_vec(history, action_history_len)])


def stacked_obs(frame_deque, history, frame_stack: int = 1, action_history_len: int = 0) -> np.ndarray:
    """Full obs = the last ``frame_stack`` base frames (oldest->newest) + one-hot action history.

    ``frame_deque`` holds recent base (OBS_LEN-d) obs vectors (most recent LAST); it is left-padded by
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
