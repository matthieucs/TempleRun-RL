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


# --- Actions -----------------------------------------------------------------
class Action(IntEnum):
    """Action bytes understood by the gadget's action server.

    0..5 are gameplay moves, 6 is a no-op, 7 restarts the run. The policy only
    ever emits 0..6 (``Discrete(N_ACTIONS)``); byte 7 is reserved for ``reset()``.
    """

    STRAFE_LEFT = 0
    STRAFE_RIGHT = 1
    JUMP = 2
    SLIDE = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5
    NOOP = 6
    RESTART = 7


N_ACTIONS = 7  # policy action space size: bytes 0..6 (7 = reset, not a policy action)


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
)

# The 14 boolean flags (everything except score/velocity/path_type/path_level/facing).
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
