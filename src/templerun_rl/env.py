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
        reward_mode: str = C.DEFAULT_REWARD_MODE,
        death_penalty: float = C.DEFAULT_DEATH_PENALTY,
        reset_settle_steps: int = C.DEFAULT_RESET_SETTLE_STEPS,
        reset_timeout: float = C.DEFAULT_RESET_TIMEOUT,
        connect_timeout: float = C.DEFAULT_CONNECT_TIMEOUT,
        freeze_timeout: float = C.DEFAULT_FREEZE_TIMEOUT,
        reset_recover_timeout: float = C.DEFAULT_RESET_RECOVER_TIMEOUT,
        step_stall_timeout: float = C.DEFAULT_STEP_STALL_TIMEOUT,
        grow_episode_steps: bool = C.DEFAULT_GROW_EPISODE_STEPS,
        max_episode_steps_growth: float = C.DEFAULT_MAX_EPISODE_STEPS_GROWTH,
        max_episode_steps_ceiling: int = C.DEFAULT_MAX_EPISODE_STEPS_CEILING,
        action_cost: float = C.DEFAULT_ACTION_COST,
        reaction_bonus: float = C.DEFAULT_REACTION_BONUS,
        turn_spam_penalty: float = C.DEFAULT_TURN_SPAM_PENALTY,
        idle_action_penalty: float = C.DEFAULT_IDLE_ACTION_PENALTY,
        redundant_action_penalty: float = C.DEFAULT_REDUNDANT_ACTION_PENALTY,
        action_history_len: int = C.DEFAULT_ACTION_HISTORY_LEN,
        frame_stack: int = C.DEFAULT_FRAME_STACK,
        turn_cooldown: int = C.DEFAULT_TURN_COOLDOWN,
        auto_connect: bool = True,
    ):
        super().__init__()
        self.turn_cooldown = int(turn_cooldown)
        self._last_turn_step = -10_000
        # obs = frame_stack*26 (+ action_history_len*7 one-hot recent actions); host-side, no hook change.
        self.action_history_len = int(action_history_len)
        self.frame_stack = max(1, int(frame_stack))
        self.observation_space = O.make_observation_space(self.frame_stack, self.action_history_len)
        self.action_space = gym.spaces.Discrete(C.N_ACTIONS)
        self._act_hist: list[int] = []
        self._frames: list = []

        self.step_dt = float(step_dt)
        self.max_episode_steps = int(max_episode_steps)
        self.reward_scale = float(reward_scale)
        # "distance": reward = DistanceTraveled-delta (the true objective; immune to the score
        # multiplier). "score": legacy score-delta (multiplier-inflated). See constants.
        self.reward_mode = str(reward_mode)
        self.death_penalty = float(death_penalty)
        self.reset_settle_steps = int(reset_settle_steps)
        self.reset_timeout = float(reset_timeout)
        self.freeze_timeout = float(freeze_timeout)
        self.reset_recover_timeout = float(reset_recover_timeout)
        self.step_stall_timeout = float(step_stall_timeout)
        self._degraded = False  # True while this phone is stuck/down (see DEFAULT_RESET_RECOVER_TIMEOUT)

        # Reward shaping (both default 0.0 = off -> the documented score-delta reward).
        # action_cost: subtracted on every non-NoOp action -> discourages thrashing / favours
        #   doing nothing unless an action helps. reaction_bonus: added when the action matches
        #   the obstacle/turn the policy was looking at (jump on must_jump, slide on must_slide,
        #   a valid turn at a junction) -> dense signal for the core skill (vs the sparse die/score).
        self.action_cost = float(action_cost)
        self.reaction_bonus = float(reaction_bonus)
        # turn_spam_penalty: a TURN right after another TURN is the junction-spam exploit the
        #   policy falls into (no recurrent memory -> it mashes turns to be sure one lands). The
        #   env supplies the missing memory via _last_action and docks repeats. > reaction_bonus
        #   so the 2nd+ consecutive turn is net-negative even when it's a "valid" turn frame.
        self.turn_spam_penalty = float(turn_spam_penalty)
        # idle_action_penalty: jump/slide fired with no obstacle (must_jump/must_slide false) is
        #   the "excited" erratic behavior; dock it on top of action_cost. Pairs with a lowered
        #   PPO ent_coef (the entropy bonus is what keeps the policy randomly spraying actions).
        self.idle_action_penalty = float(idle_action_penalty)
        # redundant_action_penalty: dock the trailing re-jump/re-slide (obstacle is the current cell,
        #   must_jump set, but not ahead via next_must_jump) — curbs the "double-jump". See reward.py.
        self.redundant_action_penalty = float(redundant_action_penalty)
        self._prev_state: Optional[dict] = None  # the state the last action responded to (for shaping)
        self._last_action: Optional[int] = None  # the action taken on the previous step (for turn-spam)

        # Adaptive episode-length ratchet (see module docstring / constants).
        self.grow_episode_steps = bool(grow_episode_steps)
        self.max_episode_steps_growth = float(max_episode_steps_growth)
        self.max_episode_steps_ceiling = int(max_episode_steps_ceiling)

        self._state = StateClient(host, state_port, connect_timeout)
        self._action = ActionClient(host, action_port, connect_timeout)
        self._connected = False

        self._prev_score = 0
        self._prev_distance = 0.0
        self._elapsed_steps = 0
        self._last_seq = 0

        if auto_connect:
            self.connect()

    # -- connection -----------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        # required=False: tolerate a phone that's down at startup -> the env constructs and runs
        # DEGRADED (reconnecting in the background) instead of crashing the whole SubprocVecEnv.
        self._state.connect(required=False)
        self._action.connect(required=False)
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

        try:
            state = self._restart_and_wait(budget=self.reset_recover_timeout)
            self._degraded = False
        except (TimeoutError, RuntimeError, OSError, ConnectionError):
            # This phone is stuck (game-over limbo) or down. Do NOT raise — that would kill the
            # whole SubprocVecEnv and take the healthy phones with it. Go DEGRADED: hand back a
            # neutral frame; step() keeps probing and the env rejoins once the phone recovers.
            self._degraded = True
            state = self._prev_state or {}
        self._prev_score = int(state.get("score", 0))
        self._prev_distance = float(state.get("distance", 0.0) or 0.0)
        self._prev_state = state
        self._last_action = None
        self._act_hist = []
        self._frames = []
        self._last_turn_step = -10_000
        self._elapsed_steps = 0
        try:
            _, self._last_seq = self._state.latest()
        except Exception:  # noqa: BLE001  no frame yet while degraded
            self._last_seq = -1
        obs = self._obs(state)
        info = {"raw_state": state, "score": self._prev_score, "degraded": self._degraded}
        return obs, info

    def _obs(self, state: dict) -> "np.ndarray":
        """Stacked obs (frame_stack frames + one-hot action history). Host-side, no hook field."""
        self._frames.append(O.state_to_obs(state))
        if len(self._frames) > self.frame_stack:
            self._frames = self._frames[-self.frame_stack:]
        return O.stacked_obs(self._frames, self._act_hist, self.frame_stack, self.action_history_len)

    def _push_action(self, a: int) -> None:
        if self.action_history_len:
            self._act_hist.append(int(a))
            if len(self._act_hist) > self.action_history_len:
                self._act_hist = self._act_hist[-self.action_history_len:]

    def _restart_and_wait(self, budget: Optional[float] = None) -> dict:
        """Start a fresh run and wait until it's actually progressing.

        Sends both reset bytes each round: 8 (RunAgainClicked) starts from the game-over
        screen, 9 (PlayFromMenu) starts from the main menu. Each is gated gadget-side to a
        no-op on the wrong screen, so whichever screen the phone is on, exactly one fires —
        robust to the game landing on the menu (e.g. after an interstitial ad).

        Freeze-tolerant: if the device crashes/freezes mid-reset, the app/gadget can be down
        for ~30 s while the watchdog relaunches + re-injects it. So we keep re-sending the
        start bytes for up to ``freeze_timeout`` (re-sending matters: a relaunched app comes
        back at the menu, which byte 9 then starts) rather than giving up after one short wait.
        """
        deadline = time.monotonic() + (budget if budget is not None else self.freeze_timeout)
        while True:
            for action in C.RESET_ACTIONS:
                self._action.try_send(int(action))  # non-blocking; sink may be briefly down (re-inject)
            try:
                return self._wait_for_fresh_run()
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Reset failed: no fresh run within freeze_timeout. The device may be stuck on a "
                        "non-gameplay screen that needs a manual tap (e.g. a full-screen ad), or the "
                        "app/gadget is down and the watchdog could not recover it. See docs/SETUP.md."
                    )
                continue

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
        a = int(action)
        if a >= C.N_ACTIONS:
            raise ValueError(f"action {a} is reserved for reset(); the policy must emit 0..{C.N_ACTIONS - 1}")
        # TURN COOLDOWN: on the phone is_turn stays True ~2 frames at a junction, so the policy fires
        # the turn on both -> the 2nd lands AFTER the corner and over-rotates the character off the
        # path (Fall). The sim consumes the turn instantly so the policy never learned to not repeat.
        # Guard host-side: after a turn is sent, drop further turn actions for `turn_cooldown` steps
        # (send NoOp). Real consecutive junctions are spaced far more than this, so it's safe.
        if self.turn_cooldown and a in (int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT)):
            if self._elapsed_steps - self._last_turn_step <= self.turn_cooldown:
                a = int(C.Action.NOOP)        # suppress the redundant 2nd turn -> no over-rotation
            else:
                self._last_turn_step = self._elapsed_steps
        if self._degraded:
            return self._degraded_step(a)
        try:
            self._require_connection()
            self._action.send(a)
            state = self._wait_one_tick()
        except (TimeoutError, OSError, ConnectionError):
            # Phone froze / socket dropped mid-episode: end the episode cleanly (terminated) so the
            # SubprocVecEnv worker survives instead of taking every phone down. VecEnv auto-resets;
            # reset() then degrades fast if the phone is still stuck.
            self._degraded = True
            last = self._prev_state or {}
            self._last_action = a
            self._push_action(a)
            self._elapsed_steps += 1
            return (
                self._obs(last), 0.0, True, False,
                {"raw_state": last, "score": self._prev_score, "elapsed_steps": self._elapsed_steps,
                 "max_episode_steps": self.max_episode_steps, "degraded": True},
            )
        if self._degraded and self._run_started(state):
            self._degraded = False  # phone recovered mid-episode -> rejoin normally
        score = int(state.get("score", self._prev_score))
        distance = float(state.get("distance", self._prev_distance) or 0.0)

        # Reward on DISTANCE (the true objective) not score: score is inflated by the challenge
        # multiplier (score/dist ramps ~23 -> ~170 over a run), so score-delta over-rewards late
        # game and decouples from "how far did it run". distance is monotonic within a run, so a
        # negative delta is a glitch / new-run frame -> clamp to 0.
        if self.reward_mode == "distance":
            reward = max(0.0, distance - self._prev_distance) * self.reward_scale
        else:
            reward = (score - self._prev_score) * self.reward_scale
        reward += self._shaping_reward(a, self._prev_state)  # action cost + correct-reaction bonus
        terminated = bool(state.get("is_dead") or state.get("is_game_over"))
        if terminated:
            reward -= self.death_penalty
        self._prev_score = score
        self._prev_distance = distance
        self._prev_state = state
        self._last_action = a
        self._push_action(a)

        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.max_episode_steps

        obs = self._obs(state)
        info = {
            "raw_state": state,
            "score": score,
            "distance": distance,
            "elapsed_steps": self._elapsed_steps,
            "max_episode_steps": self.max_episode_steps,
        }
        # Ratchet the cap up only on a *pure* truncation (alive at the limit, not a death):
        # the agent survived the whole episode, so unlock more of the (faster) game next run.
        if truncated and not terminated and self.grow_episode_steps:
            if self._grow_episode_cap():
                info["episode_cap_grew_to"] = self.max_episode_steps
        return obs, float(reward), terminated, truncated, info

    def _shaping_reward(self, action: int, prev: Optional[dict]) -> float:
        """Optional dense shaping (0 unless action_cost / reaction_bonus are set).

        Evaluated against ``prev`` — the state the policy actually saw when it chose ``action``
        (the obs returned by the previous step/reset), not the post-action state.
        - action_cost: a small penalty on any non-NoOp action (favour stillness; curb thrashing).
        - reaction_bonus: a bonus when the action matches what ``prev`` called for — JUMP on
          must_jump, SLIDE on must_slide, or a *valid* TURN at a junction (is_turn + can_turn_*).
        """
        if (not self.action_cost and not self.reaction_bonus and not self.turn_spam_penalty
                and not self.idle_action_penalty and not self.redundant_action_penalty):
            return 0.0
        r = 0.0
        if self.action_cost and action != int(C.Action.NOOP):
            r -= self.action_cost
        if self.reaction_bonus and prev is not None:
            # reward the reaction whether the obstacle is the current cell (must_*) or AHEAD (next_*)
            # -> the anticipatory jump (which clears the entry check) is rewarded, not punished.
            if (prev.get("must_jump") or prev.get("next_must_jump")) and action == int(C.Action.JUMP):
                r += self.reaction_bonus
            elif (prev.get("must_slide") or prev.get("next_must_slide")) and action == int(C.Action.SLIDE):
                r += self.reaction_bonus
            elif prev.get("is_turn"):
                if action == int(C.Action.TURN_LEFT) and prev.get("can_turn_left"):
                    r += self.reaction_bonus
                elif action == int(C.Action.TURN_RIGHT) and prev.get("can_turn_right"):
                    r += self.reaction_bonus
        # Turn-spam: a turn immediately after another turn is the junction exploit -> dock it.
        if self.turn_spam_penalty and self._last_action is not None:
            turns = (int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT))
            if action in turns and self._last_action in turns:
                r -= self.turn_spam_penalty
        # Idle jump/slide: fired with NO obstacle ahead or here (not must_* and not next_must_*) ->
        # erratic "excitement"; keyed on next_* too so the anticipatory jump is not docked.
        if self.idle_action_penalty and prev is not None:
            if (action == int(C.Action.JUMP)
                    and not prev.get("must_jump") and not prev.get("next_must_jump")):
                r -= self.idle_action_penalty
            elif (action == int(C.Action.SLIDE)
                    and not prev.get("must_slide") and not prev.get("next_must_slide")):
                r -= self.idle_action_penalty
        # Redundant jump/slide: obstacle is the current cell (must_jump) but not ahead
        # (next_must_jump=0) -> already entered/cleared; the trailing re-jump is the "double-jump".
        if self.redundant_action_penalty and prev is not None:
            if (action == int(C.Action.JUMP)
                    and prev.get("must_jump") and not prev.get("next_must_jump")):
                r -= self.redundant_action_penalty
            elif (action == int(C.Action.SLIDE)
                    and prev.get("must_slide") and not prev.get("next_must_slide")):
                r -= self.redundant_action_penalty
        # strafe is never needed in the model (and breaks turns off-center) -> dock it like an idle action
        if self.idle_action_penalty and action in (int(C.Action.STRAFE_LEFT), int(C.Action.STRAFE_RIGHT)):
            r -= self.idle_action_penalty
        return r

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

    def _degraded_step(self, a: int):
        """Fast, non-blocking step while this phone is stuck/down.

        Paces at step_dt, keeps nudging the game back into a run, and rejoins the instant a real
        run reappears. Crucially it never blocks on ``wait_fresh`` — so one frozen phone can't
        stall the lockstep SubprocVecEnv; the healthy phones keep training and this one feeds
        neutral frames until it recovers (watchdog re-inject / manual force-restart).
        """
        self._last_action = a
        self._push_action(a)
        self._elapsed_steps += 1
        time.sleep(self.step_dt)
        state = None
        try:
            for action in C.RESET_ACTIONS:           # nudge it back into a run (non-blocking)
                self._action.try_send(int(action))
            state, seq = self._state.latest()        # non-blocking peek at the newest frame
        except Exception:                            # noqa: BLE001  socket down -> stay degraded
            state, seq = None, self._last_seq
        if state and self._run_started(state):
            self._degraded = False                   # recovered -> end neutral episode, rejoin
            self._prev_state = state
            self._prev_score = int(state.get("score", 0))
            self._prev_distance = float(state.get("distance", 0.0) or 0.0)
            self._last_seq = seq
            return (self._obs(state), 0.0, True, False,
                    {"raw_state": state, "score": self._prev_score, "elapsed_steps": self._elapsed_steps,
                     "max_episode_steps": self.max_episode_steps, "recovered": True, "degraded": False})
        blank = self._prev_state or {}
        truncated = self._elapsed_steps >= self.max_episode_steps
        return (self._obs(blank), 0.0, False, truncated,
                {"raw_state": blank, "score": self._prev_score, "elapsed_steps": self._elapsed_steps,
                 "max_episode_steps": self.max_episode_steps, "degraded": True})

    def _wait_one_tick(self) -> dict:
        """Return the newest state produced after the action, honoring step_dt.

        The first wait uses ``freeze_timeout`` (not a few seconds): if the phone briefly
        freezes (screen blip, watchdog re-injecting the gadget), the StateClient reconnects
        and frames resume, so this call just *blocks* (pausing the episode) instead of
        crashing the worker. Only a freeze longer than ``freeze_timeout`` raises.
        """
        t0 = time.monotonic()
        # at least one fresh frame; bounded by step_stall_timeout so a frozen phone soft-deaths
        # quickly (the caller catches TimeoutError) instead of blocking the lockstep vec-env.
        state, seq = self._state.wait_fresh(self._last_seq, timeout=self.step_stall_timeout)
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
