"""Gymnasium environment wrapping the live Temple Run instrumentation.

This env talks to a *real phone* in real time over two TCP sockets exposed by the
Frida gadget (state on 9999, actions on 9998). One physical phone == one env; there
is no simulator and no parallelism (use multiple phones/hosts to scale).

Observation: ``Box(0, 1, (18,))`` (see ``obs.py``).
Action:      ``Discrete(7)`` -> action bytes 0..6 (7 is reserved for reset).
Reward:      score-delta * reward_scale, minus death_penalty on the terminal step.

The episode length cap (``max_episode_steps``) is a *ceiling*, not the usual end of a
run -- almost every episode ends earlier by death (``terminated``). With
``grow_episode_steps=True`` the ceiling auto-grows: each time the agent ends a run *alive
at the cap* (a pure truncation), the cap is multiplied up (ratchet, never shrinks) so the
agent can keep playing into the faster, harder part of the game next time.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from . import constants as C
from . import obs as O
from .client import ActionClient, StateClient


class TempleRunEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = C.STATE_HOST,
        state_port: int = C.STATE_PORT,
        action_port: int = C.ACTION_PORT,
        step_dt: float = C.DEFAULT_STEP_DT,
        max_episode_steps: int = C.DEFAULT_MAX_EPISODE_STEPS,
        reward_scale: float = C.DEFAULT_REWARD_SCALE,
        death_penalty: float = C.DEFAULT_DEATH_PENALTY,
        reset_settle_steps: int = C.DEFAULT_RESET_SETTLE_STEPS,
        reset_timeout: float = C.DEFAULT_RESET_TIMEOUT,
        connect_timeout: float = C.DEFAULT_CONNECT_TIMEOUT,
        grow_episode_steps: bool = C.DEFAULT_GROW_EPISODE_STEPS,
        max_episode_steps_growth: float = C.DEFAULT_MAX_EPISODE_STEPS_GROWTH,
        max_episode_steps_ceiling: int = C.DEFAULT_MAX_EPISODE_STEPS_CEILING,
        auto_connect: bool = True,
    ):
        super().__init__()
        self.observation_space = O.make_observation_space()
        self.action_space = gym.spaces.Discrete(C.N_ACTIONS)

        self.step_dt = float(step_dt)
        self.max_episode_steps = int(max_episode_steps)
        self.reward_scale = float(reward_scale)
        self.death_penalty = float(death_penalty)
        self.reset_settle_steps = int(reset_settle_steps)
        self.reset_timeout = float(reset_timeout)

        # Adaptive episode-length ratchet (see module docstring / constants).
        self.grow_episode_steps = bool(grow_episode_steps)
        self.max_episode_steps_growth = float(max_episode_steps_growth)
        self.max_episode_steps_ceiling = int(max_episode_steps_ceiling)

        self._state = StateClient(host, state_port, connect_timeout)
        self._action = ActionClient(host, action_port, connect_timeout)
        self._connected = False

        self._prev_score = 0
        self._elapsed_steps = 0
        self._last_seq = 0

        if auto_connect:
            self.connect()

    # -- connection -----------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        self._state.connect()
        self._action.connect()
        self._connected = True

    def _require_connection(self) -> None:
        if not self._connected:
            self.connect()

    @staticmethod
    def _alive(s: dict) -> bool:
        return not bool(s.get("is_game_over")) and not bool(s.get("is_dead"))

    @staticmethod
    def _run_started(s: dict) -> bool:
        # After RunAgainClicked there's a brief alive-but-not-running limbo (game-over UI
        # sliding out); the run has truly started once the player is on a track tile.
        return (not bool(s.get("is_game_over")) and not bool(s.get("is_dead"))
                and (bool(s.get("is_on_ground")) or int(s.get("path_type", -1)) >= 0))

    # -- gym API --------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._require_connection()

        state = self._restart_and_wait()
        self._prev_score = int(state.get("score", 0))
        self._elapsed_steps = 0
        _, self._last_seq = self._state.latest()
        obs = O.state_to_obs(state)
        info = {"raw_state": state, "score": self._prev_score}
        return obs, info

    def _restart_and_wait(self) -> dict:
        """Send the reset action (RunAgainClicked) and wait for a stable fresh run; one retry."""
        for attempt in range(2):
            self._action.send(int(C.RESET_ACTION))
            try:
                return self._wait_for_fresh_run()
            except TimeoutError:
                if attempt == 1:
                    raise RuntimeError(
                        "Reset failed: no fresh run after RestartGame. "
                        "Is the game in the foreground and a run/game-over screen active? See docs/SETUP.md."
                    )
        raise AssertionError("unreachable")

    def _wait_for_fresh_run(self) -> dict:
        deadline = time.monotonic() + self.reset_timeout
        _, seq = self._state.latest()
        consecutive = 0
        last_started: Optional[dict] = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            state, seq = self._state.wait_fresh(seq, timeout=remaining)
            if self._run_started(state):
                consecutive += 1
                last_started = state
                if consecutive >= self.reset_settle_steps:
                    return state
            else:
                consecutive = 0
        if last_started is not None:
            return last_started
        raise TimeoutError("no fresh run within reset_timeout")

    def step(self, action: int):
        self._require_connection()
        a = int(action)
        if a >= C.N_ACTIONS:
            raise ValueError(f"action {a} is reserved for reset(); the policy must emit 0..{C.N_ACTIONS - 1}")
        self._action.send(a)

        state = self._wait_one_tick()
        score = int(state.get("score", self._prev_score))

        reward = (score - self._prev_score) * self.reward_scale
        terminated = bool(state.get("is_dead") or state.get("is_game_over"))
        if terminated:
            reward -= self.death_penalty
        self._prev_score = score

        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.max_episode_steps

        obs = O.state_to_obs(state)
        info = {
            "raw_state": state,
            "score": score,
            "elapsed_steps": self._elapsed_steps,
            "max_episode_steps": self.max_episode_steps,
        }
        # Ratchet the cap up only on a *pure* truncation (alive at the limit, not a death):
        # the agent survived the whole episode, so unlock more of the (faster) game next run.
        if truncated and not terminated and self.grow_episode_steps:
            if self._grow_episode_cap():
                info["episode_cap_grew_to"] = self.max_episode_steps
        return obs, float(reward), terminated, truncated, info

    def _grow_episode_cap(self) -> bool:
        """Raise the episode cap (ratchet: up only, bounded by the ceiling).

        Returns True if the cap actually changed.
        """
        new_cap = min(
            int(self.max_episode_steps * self.max_episode_steps_growth),
            self.max_episode_steps_ceiling,
        )
        if new_cap > self.max_episode_steps:
            self.max_episode_steps = new_cap
            return True
        return False

    def _wait_one_tick(self) -> dict:
        """Return the newest state produced after the action, honoring step_dt."""
        t0 = time.monotonic()
        # at least one fresh frame
        state, seq = self._state.wait_fresh(self._last_seq, timeout=max(self.step_dt, 1.0) + 2.0)
        # keep draining to the newest frame until step_dt elapses
        while True:
            remaining = self.step_dt - (time.monotonic() - t0)
            if remaining <= 0:
                break
            try:
                state, seq = self._state.wait_fresh(seq, timeout=remaining)
            except TimeoutError:
                break
        self._last_seq = seq
        return state

    def render(self):
        return None

    def close(self):
        try:
            self._action.close()
        finally:
            self._state.close()
        self._connected = False
