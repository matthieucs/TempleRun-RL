"""templerun_rl — a Gymnasium environment for training agents on Temple Run via Frida.

The heavy lifting (reading game state, injecting actions) is done by a Frida gadget
inside the game on a physical Android phone; this package is the host-side RL wrapper.
"""
from __future__ import annotations

from gymnasium.envs.registration import register

from .constants import Action
from .env import TempleRunEnv
from .obs import OBS_FIELDS, OBS_LEN, state_to_obs
from .sim_env import TempleRunSimEnv

__version__ = "0.1.0"
__all__ = [
    "TempleRunEnv",
    "TempleRunSimEnv",
    "Action",
    "state_to_obs",
    "OBS_FIELDS",
    "OBS_LEN",
]

# The env performs its own truncation via max_episode_steps, so we register without a
# gymnasium TimeLimit wrapper (max_episode_steps=None) to avoid double truncation.
register(
    id="TempleRun-v0",
    entry_point="templerun_rl.env:TempleRunEnv",
    max_episode_steps=None,
    disable_env_checker=False,
)

# The headless sim (TempleRunSim-v0) self-registers at import time in sim_env.py; importing
# TempleRunSimEnv above is enough. (Re-registration is harmless — guarded there.)
