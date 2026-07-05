"""Single source of truth for the Temple Run RL environment.

Ports, the action protocol, the state-field contract, and tunable defaults all live
here so the env, scripts, and docs stay in sync. These mirror the Frida agent in
``frida/temple_run_hook.js`` (STATE_PORT/ACTION_PORT, the ``snapshot()`` field order,
and the ``dispatchAction()`` byte map).
"""
from __future__ import annotations

from enum import IntEnum

# --- Network -----------------------------------------------------------------
# The Frida gadget listens on these ports *inside the phone*; `adb forward` maps
# them to 127.0.0.1 on the host (see scripts/setup_device.py).
STATE_HOST = "127.0.0.1"
STATE_PORT = 9999          # newline-delimited JSON state, broadcast ~10 Hz
ACTION_PORT = 9998         # one action byte per command
GADGET_PORT = 27042        # Frida gadget control channel (for injection)

# --- Multi-device port convention --------------------------------------------
# In-phone ports are always the three above (separate per-device namespaces, no
# collision). On the *host*, adb forward can't map two phones to the same port, so
# each device gets its own host triple at a fixed stride: device i -> base + i*STRIDE.
# setup_device.py (--port-offset) and train.py (--num-envs) both key off this so the
# env connects to the right phone. STRIDE=100 leaves room and stays human-readable.
DEVICE_PORT_STRIDE = 100


def device_ports(index: int = 0) -> dict:
    """Host-side port triple for device ``index`` (0-based). index 0 == the defaults."""
    off = int(index) * DEVICE_PORT_STRIDE
    return {
        "gadget_port": GADGET_PORT + off,
        "state_port": STATE_PORT + off,
        "action_port": ACTION_PORT + off,
    }


# --- Actions -----------------------------------------------------------------
class Action(IntEnum):
    """Action bytes understood by the gadget's action server.

    0..5 are gameplay moves, 6 is a no-op. 7 (RESTART = RestartGame) resets game
    *logic* only and leaves the run in game-over limbo; 8 (RUN_AGAIN =
    EndGameUI.RunAgainClicked) actually STARTS a fresh, progressing run — that's what
    ``reset()`` uses. The policy only ever emits 0..6 (``Discrete(N_ACTIONS)``).
    """

    STRAFE_LEFT = 0
    STRAFE_RIGHT = 1
    JUMP = 2
    SLIDE = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5
    NOOP = 6
    RESTART = 7        # RestartGame(): logic reset only (run does not start)
    RUN_AGAIN = 8      # RunAgainClicked(): starts a run from the GAME-OVER screen
    PLAY_FROM_MENU = 9  # MainMenu.OnPlayClicked(): starts a run from the MAIN MENU ("JOUER")


N_ACTIONS = 7          # policy action space size: bytes 0..6
# reset() sends both: byte 8 starts from game-over, byte 9 from the main menu. Each is gated
# gadget-side to a no-op on the wrong screen, so sending both reliably starts a run from either.
RESET_ACTIONS = (Action.RUN_AGAIN, Action.PLAY_FROM_MENU)
RESET_ACTION = Action.RUN_AGAIN  # kept for back-compat; reset() iterates RESET_ACTIONS


# --- State contract ----------------------------------------------------------
# The 19 fields emitted by snapshot() in temple_run_hook.js, in broadcast order.
STATE_FIELDS = (
    "is_dead",
    "is_jumping",
    "is_sliding",
    "is_on_ground",
    "is_stumbling",
    "has_invincibility",
    "has_angel_wings",
    "has_vacuum",
    "has_boost",
    "is_mega_boost",
    "is_strafing",
    "score",
    "velocity",
    "path_type",
    "path_level",
    "facing",
    "must_jump",
    "must_slide",
    "is_game_over",
    "can_turn_left",
    "can_turn_right",
    "is_turn",
    "distance",   # GameController.DistanceTraveled — clean run distance (no score multiplier); reward signal, excluded from obs
    # 1-cell LOOKAHEAD (the cell ahead of the nearest) — lets the policy ANTICIPATE turns/obstacles
    # the way a human reads the track visually, instead of only reacting when a junction is already
    # the nearest cell (too late: the real turn window is ~2 control-steps). Included in the obs.
    "next_is_turn",
    "next_can_turn_left",
    "next_can_turn_right",
    "next_must_jump",
    "next_must_slide",
)

# The 17 boolean flags (everything except score/velocity/path_type/path_level/facing).
BOOL_FIELDS = (
    "is_dead",
    "is_jumping",
    "is_sliding",
    "is_on_ground",
    "is_stumbling",
    "has_invincibility",
    "has_angel_wings",
    "has_vacuum",
    "has_boost",
    "is_mega_boost",
    "is_strafing",
    "must_jump",
    "must_slide",
    "is_game_over",
    "can_turn_left",
    "can_turn_right",
    "is_turn",
)
INT_FIELDS = ("score", "path_type", "path_level", "facing")
FLOAT_FIELDS = ("velocity", "distance")

LOOKAHEAD_FIELDS = ("next_is_turn", "next_can_turn_left", "next_can_turn_right",
                    "next_must_jump", "next_must_slide")

# Default values used when a field is missing from a (possibly truncated) JSON line.
STATE_DEFAULTS = {f: False for f in BOOL_FIELDS}
STATE_DEFAULTS.update({f: False for f in LOOKAHEAD_FIELDS})
STATE_DEFAULTS.update({"score": 0, "path_type": -1, "path_level": -1, "facing": -1,
                       "velocity": 0.0, "distance": 0.0,
                       "cell_progress": 0.0, "lane": 0.0,  # lane default = center (tilt-controlled model)
                       "next_wide_gap": False, "narrow_target_lane": None,
                       "next_ground_change_higher": False,   # zone-crevasse-ahead flag (gates predicted_jump)
                       "next_path_level": -1, "next2_path_level": -1,  # upcoming altitude 2-cell lookahead (16q)
                       "stumble_kill_timer": 0.0, "stumble_count": 0})


# --- Environment defaults ----------------------------------------------------
DEFAULT_STEP_DT = 0.1            # seconds between control ticks (~10 Hz, matches the 100 ms broadcast)
DEFAULT_MAX_EPISODE_STEPS = 1000
# Reward is keyed on DISTANCE, not score: score is inflated by the challenge MULTIPLIER (score/dist
# ramps ~23 -> ~170 over a long run) so it's a non-stationary, distance-decoupled signal. The goal is
# max distance, so reward = distance-delta. distance advances ~1.0-2.1 per 10 Hz step; scale 0.1 keeps
# the typical step reward ~O(0.1-0.2), comparable to the old score-delta reward.
DEFAULT_REWARD_MODE = "distance"   # "distance" (DistanceTraveled-delta) | "score" (legacy score-delta)
DEFAULT_REWARD_SCALE = 0.1      # reward = reward-delta * scale
DEFAULT_DEATH_PENALTY = 1.0     # subtracted on the terminal (death / game-over) step
DEFAULT_RESET_SETTLE_STEPS = 3  # consecutive "alive" frames required before reset() returns
DEFAULT_RESET_TIMEOUT = 5.0     # seconds to wait for a fresh run after RestartGame
DEFAULT_CONNECT_TIMEOUT = 5.0
# How long step() will wait for the stream to resume before treating the device as dead.
# A brief freeze (screen blip, gadget re-injection by the watchdog) should PAUSE the worker,
# not crash it -- so this is generous. Only a freeze longer than this raises (truly gone).
DEFAULT_FREEZE_TIMEOUT = 90.0

# Adaptive episode-length "ratchet" (off by default -> fixed cap). When enabled, the cap
# is raised only when an episode ends *alive at the limit* (truncation, not death): the
# agent has proven it can survive that long, so we unlock more of the (faster) game next
# time. The cap only ever grows, bounded by the ceiling. Early on the agent dies first, so
# this does nothing until it's good enough to reach the current cap. See env.py.
DEFAULT_GROW_EPISODE_STEPS = False        # opt-in; keeps the fixed cap unless turned on
DEFAULT_MAX_EPISODE_STEPS_GROWTH = 1.5    # multiply the cap by this on each pure-truncation
DEFAULT_MAX_EPISODE_STEPS_CEILING = 6000  # hard ceiling (~10 min at 100 ms) so it can't grow forever

# Reward shaping (opt-in; both 0.0 = off -> the plain score-delta reward). See env._shaping_reward.
DEFAULT_ACTION_COST = 0.0      # penalty per non-NoOp action (curb thrashing / favour stillness); try ~0.02-0.05
DEFAULT_REACTION_BONUS = 0.0   # bonus for the correct reaction (jump/slide/valid-turn); try ~0.1 (> action_cost)
DEFAULT_TURN_SPAM_PENALTY = 0.0  # extra penalty for a TURN that immediately follows another TURN (curb junction turn-spam); try ~0.15 (> reaction_bonus)
DEFAULT_IDLE_ACTION_PENALTY = 0.0  # extra penalty for JUMP with no must_jump / SLIDE with no must_slide (curb erratic jumping/sliding when no obstacle calls for it); try ~0.06 (> action_cost)
DEFAULT_REDUNDANT_ACTION_PENALTY = 0.0  # extra penalty for JUMP/SLIDE when the obstacle is the CURRENT cell (must_jump) but NOT ahead (next_must_jump=0) -> the trailing re-jump after the obstacle is already cleared (the "double-jump"). try ~0.12 (slightly > reaction_bonus, so a death-preventing jump still wins but a redundant one nets negative)

# Action-history stacking: append the one-hot of the last K actions to the obs so the memoryless MLP
# policy knows what it JUST did (curbs "multiple turns at once" at a junction — the single-frame obs
# can't see it already turned). 0 = off (26-d obs). >0 -> obs grows by K*N_ACTIONS. Host-side (the env
# tracks its own actions) so it needs NO hook change and works the same in sim and on the phone.
DEFAULT_ACTION_HISTORY_LEN = 0

# Frame stacking: feed the last N base-26 observation frames (oldest->newest) so the MLP sees how
# is_turn/facing/velocity are EVOLVING -> it can anticipate a junction and fire the turn on the exact
# frame (beats the single-frame reactive limit that caps turn reliability). 1 = off (single frame).
# obs grows to N*26 (+ action history). Host-side; identical in sim and on the phone (no hook change).
DEFAULT_FRAME_STACK = 1

# Turn cooldown (live env only): after a turn is sent, drop further turn actions for this many steps.
# On the phone is_turn stays True ~2 frames at a junction; the policy fires the turn on both and the
# 2nd over-rotates the character off the corner (Fall) — the dominant turn-death. 0 = off; try 2-3.
DEFAULT_TURN_COOLDOWN = 0

# Per-env resilience: in a SubprocVecEnv any single env that raises kills ALL workers. If a
# reset can't start a fresh run within this budget (phone wedged in game-over limbo / app down),
# the env goes DEGRADED instead of raising — it hands back neutral frames and keeps probing, so
# the other phones keep training and this one rejoins once it recovers (watchdog/force-restart).
# Kept short so a stuck phone stalls the lockstep vec only briefly per reset (not freeze_timeout).
DEFAULT_RESET_RECOVER_TIMEOUT = 12.0
# Per-STEP stall bound. _wait_one_tick waits at most this for the first fresh frame before
# soft-terminating the episode (then the env goes degraded). Kept short so ONE frozen phone
# (0 frames) doesn't block the lockstep SubprocVecEnv for the full freeze_timeout each step —
# the healthy phones keep training while the frozen one degrades and rejoins on recovery.
DEFAULT_STEP_STALL_TIMEOUT = 5.0
