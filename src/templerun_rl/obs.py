"""Observation encoding: raw game-state dict -> normalized float32 vector.

The policy sees an OBS_LEN-element ``Box(0, 1)`` vector (see OBS_FIELDS; 28 as of 2026-07-04). We deliberately exclude the raw
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
    # power-ups REMOVED 2026-07-04 (user): never spawn on device (rate 0) nor in the sim -> always-0 dead
    # inputs. has_invincibility/has_angel_wings/has_vacuum/has_boost/is_mega_boost dropped. OBS_LEN 32->27.
    "is_strafing", "must_jump", "must_slide", "is_game_over",
    "can_turn_left", "can_turn_right", "is_turn",
    # 1-cell lookahead — anticipation (see a junction/obstacle before it's the nearest cell).
    "next_is_turn", "next_can_turn_left", "next_can_turn_right", "next_must_jump", "next_must_slide",
    # the upcoming must-jump is a wide gap -> jump LATE. next_ground_change_higher distinguishes a
    # 1-cell step-UP crevasse (raised ledge, needs a LATE takeoff, device probe 2026-07-03) from a
    # 2-cell long hole (needs an EARLIER takeoff to span both cells) — different jump timings.
    "next_wide_gap", "next_ground_change_higher",
    # 2-cell OBSTACLE lookahead (2026-07-07) — the chained-obstacle death fix. A jump spans ~2.75 cells,
    # so a jump over cell+1 lands the runner airborne INTO cell+2; if cell+2 is a slide-obstacle
    # (flame/tree-slide) the runner can't slide while airborne -> Burnt (device death @6216m: hole-jump ->
    # flame tower right behind). Giving the policy the 2-ahead must-jump/slide lets it defer/re-time the
    # jump. SAME junction-dropout as next2_path_level (False through a turn). BREAKING: obs 58->60.
    "next2_must_jump", "next2_must_slide",
)
OBS_SCALAR_FIELDS = ("velocity", "path_type", "path_level", "facing")
# Perception added 2026-06-19 so the NN can time turns + control its lane natively (replacing the
# host-side assists): cell_progress = longitudinal position in the nearest cell (0..1; lets the policy
# fire a turn at cell-center instead of too early), lane = which of the 3 lanes (L/C/R) the runner is in
# (for narrow strips). Both come from the hook (CharacterPlayer.GetPosition2D + PathElement.Rectangle)
# and from the sim's own state — they are OBSERVATIONS, not actions.
# next_path_level / next2_path_level: the altitude Level (PathLevel 1/2/3 = low/mid/high) of the cell 1 and
# 2 ahead. DEVICE-VERIFIED READABLE (probe 16q, 10/10): sign(next2_path_level - path_level) = the true
# crevasse direction (up/down), which `next_ground_change_higher` (non-directional) could NOT convey. Gives
# the NN the SIGNED transition (low->mid vs high->mid etc.) + magnitude, so it stops blindly jumping every
# cap. The old "future Level reads 0" assumption was wrong for this build.
OBS_EXTRA_FIELDS = ("cell_progress", "lane", "narrow_target_lane", "predicted_landing", "predicted_jump",
                    "next_path_level", "next2_path_level")
OBS_FIELDS = OBS_BOOL_FIELDS + OBS_SCALAR_FIELDS + OBS_EXTRA_FIELDS
OBS_LEN = len(OBS_FIELDS)  # 21 bool/lookahead + 4 scalar + 7 extra = 32 (keep derived from OBS_FIELDS)

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

# PREDICTED-JUMP geometry (2026-07-04) — the CREVASSE analog of predicted_landing (device-measured
# human_calib_v2: step-UP crevasses jumped 100%, and the takeoff LEAD scales with velocity — the human
# takes off ~on the gap @v150 but ~1.5 cells early @v210). The sim's step-up death model
# (sim_env `_check_death`) is a parabola: airborne span = jump_duration*velocity/cell_length cells, and
# the runner must reach the raised LEDGE (2 cells ahead of the current cell when next_ground_change_higher
# fires: cell1=gap, cell2=ledge) NEAR the apex (frac 0.5) or it clips the wall (DeathType 4 Ledge). So the
# ideal takeoff is 0.5*span cells before the ledge. `predicted_jump` projects "how far am I from that ideal
# takeoff point", velocity-invariant, so the policy learns ONE threshold ("jump when predicted_jump ~=
# JUMP_NOW") that self-adjusts across speeds — the same trick predicted_landing used to break the turn wall
# at all speeds. Derived host-side from the already-streamed cell_progress+velocity+next_ground_change_higher
# (NO hook change; deployable). Constants match the sim's jump_duration/cell_length so sim-train == device.
JUMP_DURATION = 0.8    # s — device jump airtime (= sim cfg jump_duration); span = JUMP_DURATION*v/CELL_LENGTH
CELL_LENGTH = 60.0     # device units per path cell (= sim cfg cell_length; consistent w/ DCPROG_PER_VEL)
LEDGE_CELLS_AHEAD = 2.0  # when next_ground_change_higher fires: cell1 = gap, cell2 = raised ledge (apex here)
PJ_NONE = 1.0          # sentinel when no step-up crevasse is imminent (far from any ledge -> don't jump)


def predicted_jump_obs(cell_progress, velocity, crevasse_ahead) -> float:
    """Velocity-invariant step-up jump-timing cue. When a zone crevasse is the next cell
    (``crevasse_ahead``), returns how far the runner is from the IDEAL takeoff point (small value ~=
    "take off NOW"); otherwise ``PJ_NONE`` (far). Monotonic in the distance-to-ideal-takeoff, computed
    identically in sim + on device so the learned threshold transfers.

    dist_to_ledge = LEDGE_CELLS_AHEAD - cell_progress (cells to the raised ledge);
    lead_needed    = 0.5 * span                       (take off half a span before the ledge = apex there);
    raw            = dist_to_ledge - lead_needed       (0 = take off now, >0 too early, <0 too late).
    """
    if not crevasse_ahead:
        return PJ_NONE
    v = float(velocity or 0.0)
    cp = float(cell_progress if cell_progress is not None else 0.0)
    span = JUMP_DURATION * v / CELL_LENGTH
    raw = (LEDGE_CELLS_AHEAD - cp) - 0.5 * span
    # map the useful range [-0.5, 1.5] -> [0,1] (raw=0 "jump now" -> 0.25); clip keeps it bounded.
    return float(np.clip((raw + 0.5) / 2.0, 0.0, 1.0))


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
    """Convert a raw state dict to the base (OBS_LEN=28) normalized float32 observation vector.

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
    # predicted_jump: velocity-invariant step-up crevasse takeoff cue (see predicted_jump_obs). Gated on
    # next_ground_change_higher (the zone-crevasse-ahead flag, set by the sim snapshot / device hook).
    out[base + 8] = predicted_jump_obs(cp0, v, g("next_ground_change_higher"))
    # next_path_level / next2_path_level: upcoming altitude 2-cell lookahead (device-verified readable).
    # direction of a crevasse = sign(next2_path_level - path_level). Same enum normalization as path_level.
    out[base + 9] = _norm_enum(int(g("next_path_level")), PATH_LEVEL_MAX)
    out[base + 10] = _norm_enum(int(g("next2_path_level")), PATH_LEVEL_MAX)
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
