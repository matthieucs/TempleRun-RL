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
    RESTART = 7      # RestartGame(): logic reset only (run does not start)
    RUN_AGAIN = 8    # RunAgainClicked(): starts a real run (used for episode reset)


N_ACTIONS = 7          # policy action space size: bytes 0..6
RESET_ACTION = Action.RUN_AGAIN  # what reset() sends to begin a fresh, progressing run


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
FLOAT_FIELDS = ("velocity",)

# Default values used when a field is missing from a (possibly truncated) JSON line.
STATE_DEFAULTS = {f: False for f in BOOL_FIELDS}
STATE_DEFAULTS.update({"score": 0, "path_type": -1, "path_level": -1, "facing": -1, "velocity": 0.0})


# --- Environment defaults ----------------------------------------------------
DEFAULT_STEP_DT = 0.1            # seconds between control ticks (~10 Hz, matches the 100 ms broadcast)
DEFAULT_MAX_EPISODE_STEPS = 1000
DEFAULT_REWARD_SCALE = 0.01     # score-delta -> reward scale (keeps typical step reward ~O(0.1-1))
DEFAULT_DEATH_PENALTY = 1.0     # subtracted on the terminal (death / game-over) step
DEFAULT_RESET_SETTLE_STEPS = 3  # consecutive "alive" frames required before reset() returns
DEFAULT_RESET_TIMEOUT = 5.0     # seconds to wait for a fresh run after RestartGame
DEFAULT_CONNECT_TIMEOUT = 5.0

# Adaptive episode-length "ratchet" (off by default -> fixed cap). When enabled, the cap
# is raised only when an episode ends *alive at the limit* (truncation, not death): the
# agent has proven it can survive that long, so we unlock more of the (faster) game next
# time. The cap only ever grows, bounded by the ceiling. Early on the agent dies first, so
# this does nothing until it's good enough to reach the current cap. See env.py.
DEFAULT_GROW_EPISODE_STEPS = False        # opt-in; keeps the fixed cap unless turned on
DEFAULT_MAX_EPISODE_STEPS_GROWTH = 1.5    # multiply the cap by this on each pure-truncation
DEFAULT_MAX_EPISODE_STEPS_CEILING = 6000  # hard ceiling (~10 min at 100 ms) so it can't grow forever
