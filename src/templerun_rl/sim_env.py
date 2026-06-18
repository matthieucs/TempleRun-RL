"""``TempleRunSim-v0`` — a fast, headless, pure-python Temple Run model.

A simulator that reproduces the *real* env's observation / reward / action contract
(``TempleRunEnv``) so a policy trained here transfers to the phone (and vice-versa).
No phone, no Frida, no sockets, **no sleeps** — thousands of steps/sec.

What it models (per ``docs/SIM_MODEL_SPEC.md``):
  - the track as a linked sequence of ``PathElement``-like cells (type / level / obstacle /
    junction info), generated procedurally per ``config`` OR replayed from an injected list;
  - the runner physics (velocity 100->210 accel, 3 lanes + strafe, jump/slide/strafe/turn as
    timed flag transitions);
  - collision/death (un-actioned obstacle at the player's cell, or a junction passed without a
    valid turn -> Fall);
  - scoring (~ velocity * dt -> reproduces the ~+27/step-at-start trend).

It emits the SAME 22-field raw-state dict (``constants.STATE_FIELDS``) the Frida hook produces,
turns it into an obs via ``obs.state_to_obs`` (reused verbatim), and computes the SAME reward
(score-delta * reward_scale + the shared shaping in ``reward.py``, minus death_penalty).

Two ``track_source`` modes (SIM_CONTRACT §2):
  - ``"procedural"`` — RNG-generate the track (training).
  - an injected ``list[dict]`` — REPLAY mode: consume the provided elements as the player
    advances (fidelity harness). Each dict: ``{type,level,obstacles,is_turn,can_turn_left,
    can_turn_right}`` (extra keys ignored; missing keys defaulted).

All inferred numeric knobs live in ``DEFAULT_SIM_CONFIG`` below (one place to calibrate later).

Run the smoke test:  ``python -m templerun_rl.sim_env``  (or ``python src/templerun_rl/sim_env.py``).
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from . import constants as C
from . import obs as O
from . import reward as R


# =============================================================================
# DEFAULT_SIM_CONFIG — every INFERRED numeric knob, in one labeled place.
# [KNOWN] = verbatim from dump.cs / SIM_MODEL_SPEC. [INFERRED] = needs calibration.
# Group A: velocity/physics. Group B: track generation. Group C: scoring. Group D: timings.
# =============================================================================
DEFAULT_SIM_CONFIG: dict[str, Any] = {
    # --- A. Velocity / pacing ------------------------------------------------
    "start_velocity": 100.0,          # [KNOWN] SRV (CharacterPlayer.StartRunVelocity)
    "max_velocity": 210.0,            # [KNOWN] MRV (CharacterPlayer.MaxRunVelocity)
    "accel_per_sec": 6.0,             # [INFERRED] linear fallback if velocity_curve is empty
    # velocity = curve(distance) — the real AccCurve is distance-gated, not linear. Measured from
    # human runs (median velocity per distance bin): rises 100->210 over ~4k distance, then plateaus.
    "velocity_curve": [               # [CALIBRATED] (DistanceTraveled, velocity) median over real runs
        [5, 100.0], [10, 110.9], [30, 125.7], [50, 137.1], [70, 138.6], [90, 139.8], [110, 141.3],
        [130, 142.6], [150, 144.1], [170, 145.5], [190, 146.3], [225, 148.7], [275, 151.6],
        [325, 154.3], [375, 157.6], [425, 159.7], [475, 162.6], [525, 164.9], [575, 167.3],
        [750, 174.0], [1050, 180.7], [1350, 184.5], [1650, 187.5], [1950, 190.6], [2250, 194.0],
        [2550, 196.8], [2850, 200.0], [3150, 202.7], [3450, 205.5], [3750, 207.4], [4200, 210.0],
        [6750, 210.0],
    ],
    "stumble_slowdown": 0.85,         # [KNOWN] StumbleSlowDown — velocity *= this on a stumble
    # --- B. Track generation (PathElement statics) — values READ off the device (Frida, baseline
    # Temple/level-2 section; the game ramps these by distance via section/level changes). -------
    "min_space_between_obstacles": 6,   # [READ] sMinSpaceBetweenObstacles (cells)
    "min_space_between_turns": 12,      # [READ] sMinSpaceBetweenTurns (cells)
    "min_spaces_between_bonus": 50,     # [READ] sMinSpacesBetweenBonusItems
    "obstacle_prob": 0.45,             # [INFERRED] chance a non-gated cell carries an obstacle (real gen is deterministic+variation; approx)
    "double_obstacle_percent": 0.0,     # [READ] sDoubleObstaclePercent (no doubles in the baseline section)
    "max_back_to_back_obstacles": 2,    # [READ] sMaxBackToBackObstacles
    "turn_prob": 0.18,                 # [INFERRED] chance a turn-eligible cell becomes a junction (early game)
    # Difficulty RAMP: the real game densifies turns with distance (measured turns/1k: ~8.5 at the
    # start -> ~21 by distance 5-6k). difficulty = clip((distance-start)/ramp_distance, 0, 1) lerps
    # the turn spacing down + turn prob up; slides only switch on past slides_after_distance (the
    # baseline statics have sAllowTreeSlide/FlameTowers = False -> no slides early).
    "ramp_distance": 6000.0,           # [CALIBRATED] distance over which difficulty goes 0 -> 1
    "min_space_between_turns_late": 7, # [CALIBRATED] turn spacing at full difficulty (cells)
    "turn_prob_late": 0.5,             # [CALIBRATED] turn probability at full difficulty
    # A junction is turnable only for the first this-fraction of the cell — ~2 control-steps,
    # matching the real ~217ms turn window (measured: is_turn True ~13 frames @60Hz over 206 human
    # turns). Forces the policy to turn promptly (else it learns lazy turns -> dies-too-late on phone).
    "turn_window_frac": 0.5,           # [CALIBRATED] fraction of the junction cell that stays turnable
    "slides_after_distance": 300.0,    # [CALIBRATED] TreeSlide/FlameTower obstacles only past this distance
    "default_path_level": 2,           # [READ] sPathLevel = 2 (Temple)
    # Powerup spawn probabilities are all 0 in the baseline read (sProbabilityBoost/Invincibility/
    # Vacuum/CoinBonus = 0) — why human runs saw 0 boost frames; the sim models no powerups (v1).
    "min_spaces_between_coin_runs": 12, # [READ] sMinSpacesBetweenCoinRuns
    "max_coins_per_run": 5,            # [READ] sMaxCoinsPerRun
    # obstacle-type mix among the "real" obstacles (must sum to ~1; calibrated later).
    # Keys are ObstacleType enum values (SPEC §2): 1 Hole, 2 Narrow, 3 TreeStumble,
    # 4 TreeJump, 5 TreeSlide, 6 FlameTower. (0 None, 7 LevelChange handled separately.)
    "obstacle_type_weights": {
        1: 0.22,   # Hole         -> jump
        4: 0.22,   # TreeJump     -> jump
        5: 0.22,   # TreeSlide    -> slide
        6: 0.14,   # FlameTower   -> slide
        3: 0.12,   # TreeStumble  -> stumble (survivable)
        2: 0.08,   # Narrow       -> strafe to center (survivable in v0)
    },
    # piece PathType to emit for straight vs turn cells (obs path_type /37). [KNOWN enum]
    "path_type_straight": 1,           # StraightVert
    "path_type_turn": 3,               # TurnA (any of 3..6 are turns; 3 is representative)
    "path_type_t_junction": 7,         # T-junction (both sides turnable)
    # --- C. Scoring ----------------------------------------------------------
    # Calibrated so a start-velocity step (vel 100, dt 0.1 -> distance 10) yields ~+27 score,
    # matching CLAUDE.md §9 (~+27 score / 100ms-step at run start, growing with velocity).
    "score_per_distance": 2.7,         # [INFERRED] score units per distance unit (dist = vel*dt)
    "start_score": 162,                # [KNOWN-ish] reset frame score ~162 (CLAUDE.md §9)
    "coin_score": 50,                  # [INFERRED] score bump when a coin/bonus cell is consumed
    # The reward keys on DISTANCE (DistanceTraveled), which advances ~0.107 per (velocity*dt) of
    # model distance in real units (real dist-rate/velocity ~0.10-0.116; vel 100 -> ~1.16/10Hz-step).
    # The sim emits a real-scale "distance" so its distance-delta reward matches the phone's units.
    "dist_per_velsec": 0.107,          # [CALIBRATED] real DistanceTraveled per (velocity*dt) model unit
    "start_distance": 5.3,             # [CALIBRATED] DistanceTraveled at run start (reset-frame offset)
    # --- D. Action timings (seconds; durations of the transient flags) -------
    "jump_duration": 0.6,             # [INFERRED] airtime — IsJumping true for this long
    "slide_duration": 0.6,            # [INFERRED] SlideDuration@0x108
    "strafe_duration": 0.25,          # [INFERRED] strafe lane-shift dur@0x12C
    "stumble_duration": 0.5,          # [INFERRED] StumbleDuration; unrecovered -> death window
    # --- E. Geometry ---------------------------------------------------------
    "cell_length": 60.0,              # PROCEDURAL mode: cells crossed per (velocity*dt) of model distance
    # REPLAY mode crosses cells off the RECORDED hidden.distance (decoupled from the velocity
    # model so path_type tests cell-length alone). Calibrated from real traces: the cell-to-cell
    # gap in hidden.distance units is ~6.7 (median over 24 crossings, 3 episodes); the Temple
    # start-piece is partial, so the player begins ~3.2 into the first cell (first crossing after
    # ~3.5 distance, then every cell_length_real).
    "cell_length_real": 6.4,          # [CALIBRATED] hidden.distance units between adjacent PathElements
                                      # (8 body cells span distance 8.64->59.69 in ep0 => 6.38/cell)
    "replay_start_into_cell": 3.2,    # [CALIBRATED] dist_into_cell at run start (Temple is partial)
    "lookahead": 12,                  # cells of track kept ahead of the player (>= SIM_CONTRACT K>=8)
}

# ObstacleType enum (SPEC §2) -> which flag must be active when the cell is passed.
OBST_NONE = 0
OBST_HOLE = 1
OBST_NARROW = 2
OBST_TREE_STUMBLE = 3
OBST_TREE_JUMP = 4
OBST_TREE_SLIDE = 5
OBST_FLAME_TOWER = 6
OBST_LEVEL_CHANGE = 7

_MUST_JUMP = frozenset({OBST_HOLE, OBST_TREE_JUMP})        # SPEC §0: must_jump = Obstacles ∈ {1,4}
_MUST_SLIDE = frozenset({OBST_TREE_SLIDE, OBST_FLAME_TOWER})  # must_slide = Obstacles ∈ {5,6}
_SURVIVABLE = frozenset({OBST_NARROW, OBST_TREE_STUMBLE})  # don't kill in v0 (stumble/strafe)

# PathDirection enum (SPEC §0): None0 North1 East2 South3 West4.
DIR_NONE, DIR_N, DIR_E, DIR_S, DIR_W = 0, 1, 2, 3, 4
# left/right of a heading (turning relative to current facing).
_LEFT_OF = {DIR_N: DIR_W, DIR_W: DIR_S, DIR_S: DIR_E, DIR_E: DIR_N}
_RIGHT_OF = {DIR_N: DIR_E, DIR_E: DIR_S, DIR_S: DIR_W, DIR_W: DIR_N}


def _element(type: int = 1, level: int = 2, obstacles: int = 0,
             is_turn: bool = False, can_turn_left: bool = False,
             can_turn_right: bool = False, is_coin: bool = False) -> dict:
    """A single track cell (the sim's internal PathElement)."""
    return {
        "type": int(type),
        "level": int(level),
        "obstacles": int(obstacles),
        "is_turn": bool(is_turn),
        "can_turn_left": bool(can_turn_left),
        "can_turn_right": bool(can_turn_right),
        "is_coin": bool(is_coin),
    }


class TempleRunSimEnv(gym.Env):
    """Headless Temple Run simulator (see module docstring). Drop-in for ``TempleRunEnv``."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        track_source: "str | list[dict]" = "procedural",
        config: Optional[dict] = None,
        seed: Optional[int] = None,
        step_dt: float = C.DEFAULT_STEP_DT,
        max_episode_steps: int = C.DEFAULT_MAX_EPISODE_STEPS,
        reward_scale: float = C.DEFAULT_REWARD_SCALE,
        reward_mode: str = C.DEFAULT_REWARD_MODE,
        death_penalty: float = C.DEFAULT_DEATH_PENALTY,
        # same shaping knobs as the real env so training transfers (see reward.py):
        action_cost: float = C.DEFAULT_ACTION_COST,
        reaction_bonus: float = C.DEFAULT_REACTION_BONUS,
        turn_spam_penalty: float = C.DEFAULT_TURN_SPAM_PENALTY,
        idle_action_penalty: float = C.DEFAULT_IDLE_ACTION_PENALTY,
        **_ignored: Any,
    ):
        super().__init__()
        # IDENTICAL spaces to TempleRunEnv (Box(21), Discrete(7)) -> policies are drop-in.
        self.observation_space = O.make_observation_space()
        self.action_space = gym.spaces.Discrete(C.N_ACTIONS)

        # Merge config over defaults (shallow + one level for the obstacle weights).
        self.cfg = dict(DEFAULT_SIM_CONFIG)
        if config:
            for k, v in config.items():
                if k == "obstacle_type_weights" and isinstance(v, dict):
                    w = dict(DEFAULT_SIM_CONFIG["obstacle_type_weights"])
                    w.update(v)
                    self.cfg[k] = w
                else:
                    self.cfg[k] = v

        self.step_dt = float(step_dt)
        self.max_episode_steps = int(max_episode_steps)
        self.reward_scale = float(reward_scale)
        self.reward_mode = str(reward_mode)
        self.death_penalty = float(death_penalty)
        self.action_cost = float(action_cost)
        self.reaction_bonus = float(reaction_bonus)
        self.turn_spam_penalty = float(turn_spam_penalty)
        self.idle_action_penalty = float(idle_action_penalty)

        # Velocity curve (distance -> velocity) for fast interpolation in _advance.
        vc = self.cfg.get("velocity_curve") or []
        self._vel_curve_d = np.array([p[0] for p in vc], dtype=float) if vc else None
        self._vel_curve_v = np.array([p[1] for p in vc], dtype=float) if vc else None

        # Track source: "procedural" (RNG) or an injected list of element dicts (replay).
        if isinstance(track_source, str):
            self.track_source = "procedural"
            self._injected_track: Optional[list[dict]] = None
        else:
            self.track_source = "replay"
            self._injected_track = [self._coerce_element(e) for e in track_source]

        self._rng = np.random.default_rng(seed)

        # episode/bookkeeping state (set up in reset())
        self._prev_score = 0
        self._prev_distance = 0.0
        self._prev_state: Optional[dict] = None
        self._last_action: Optional[int] = None
        self._elapsed_steps = 0
        self._reset_internal_state()

    # ------------------------------------------------------------------ config
    @staticmethod
    def _coerce_element(e: dict) -> dict:
        """Normalize an injected/replay element dict to the internal schema."""
        return _element(
            type=int(e.get("type", e.get("path_type", 1))),
            level=int(e.get("level", e.get("path_level", 2))),
            obstacles=int(e.get("obstacles", 0)),
            is_turn=bool(e.get("is_turn", False)),
            can_turn_left=bool(e.get("can_turn_left", False)),
            can_turn_right=bool(e.get("can_turn_right", False)),
            is_coin=bool(e.get("is_coin", False)),
        )

    # ------------------------------------------------------- internal physics
    def _reset_internal_state(self) -> None:
        self._velocity = float(self.cfg["start_velocity"])
        self._facing = DIR_N
        self._lane = 0                       # -1 L, 0 C, +1 R (3 lanes; SPEC §3)
        self._distance = 0.0
        self._distance_real = float(self.cfg["start_distance"])  # real-scale DistanceTraveled (reward signal)
        self._last_ext_distance = 0.0        # replay: last recorded hidden.distance (cell-crossing anchor)
        self._dist_into_cell = 0.0           # how far the player is into the current cell
        self._cell_index = 0                 # index of the cell the player is on (in self._track)
        self._is_dead = False
        self._death_type: Optional[int] = None
        # transient action flags as time-remaining counters (seconds)
        self._jump_t = 0.0
        self._slide_t = 0.0
        self._strafe_t = 0.0
        self._stumble_t = 0.0
        self._ground_height = 0.0
        self._score = float(self.cfg["start_score"])
        # Start the spacing counters at 0 (NOT the minimum): the real run has a straight intro
        # (first turn @ ~12 cells = sMinSpaceBetweenTurns, first obstacle @ ~6-29 cells), so a cell
        # only becomes turn/obstacle-eligible after that many cells from the start.
        self._cells_since_obstacle = 0
        self._cells_since_turn = 0
        self._cells_since_bonus = 0
        self._b2b_obstacles = 0
        # replay cursor (which injected element feeds the next generated cell)
        self._replay_cursor = 0
        # the live track window: a list of cells, [_cell_index] = current.
        self._track: list[dict] = []
        self._build_initial_track()

    def _build_initial_track(self) -> None:
        # cell 0 is always a plain straight start cell (the reset/settle cell).
        self._track = [_element(type=self.cfg["path_type_straight"],
                                level=self.cfg["default_path_level"])]
        self._cell_index = 0
        self._extend_track()

    def _extend_track(self) -> None:
        """Ensure at least ``lookahead`` cells exist ahead of the player."""
        want = self._cell_index + int(self.cfg["lookahead"]) + 1
        while len(self._track) < want:
            self._track.append(self._next_cell())

    def _next_cell(self) -> dict:
        if self.track_source == "replay":
            return self._next_cell_replay()
        return self._next_cell_procedural()

    def _next_cell_replay(self) -> dict:
        track = self._injected_track or []
        if self._replay_cursor < len(track):
            cell = dict(track[self._replay_cursor])
            self._replay_cursor += 1
            return cell
        # ran out of recorded track: feed neutral straight cells (the episode usually ended).
        return _element(type=self.cfg["path_type_straight"], level=self.cfg["default_path_level"])

    def _difficulty(self) -> float:
        """0 at run start, 1 by ``ramp_distance`` — drives the turn-density ramp (real game
        densifies turns ~8.5/1k -> ~21/1k over distance 0 -> 6k)."""
        d = self._distance_real - float(self.cfg["start_distance"])
        return float(np.clip(d / max(1.0, float(self.cfg["ramp_distance"])), 0.0, 1.0))

    def _next_cell_procedural(self) -> dict:
        cfg = self.cfg
        level = int(cfg["default_path_level"])
        self._cells_since_obstacle += 1
        self._cells_since_turn += 1
        self._cells_since_bonus += 1

        # Difficulty-ramped turn spacing/probability (lerp baseline -> late by distance).
        diff = self._difficulty()
        min_turns = (float(cfg["min_space_between_turns"]) * (1.0 - diff)
                     + float(cfg["min_space_between_turns_late"]) * diff)
        turn_p = (float(cfg["turn_prob"]) * (1.0 - diff)
                  + float(cfg["turn_prob_late"]) * diff)

        # --- junction / turn decision (gated by min_space_between_turns) -----
        is_turn = False
        can_l = False
        can_r = False
        ptype = int(cfg["path_type_straight"])
        if self._cells_since_turn >= min_turns:
            if self._rng.random() < turn_p:
                is_turn = True
                self._cells_since_turn = 0
                roll = self._rng.random()
                if roll < 0.4:
                    can_l, can_r = True, False
                    ptype = int(cfg["path_type_turn"])
                elif roll < 0.8:
                    can_l, can_r = False, True
                    ptype = int(cfg["path_type_turn"])
                else:
                    can_l, can_r = True, True       # T-junction: both sides
                    ptype = int(cfg["path_type_t_junction"])

        # A turn cell carries no obstacle (the player must turn here).
        obstacle = OBST_NONE
        if not is_turn and self._cells_since_obstacle >= int(cfg["min_space_between_obstacles"]):
            place = self._rng.random() < float(cfg["obstacle_prob"])
            # honor max back-to-back: if we've hit the cap, force a gap.
            if self._b2b_obstacles >= int(cfg["max_back_to_back_obstacles"]):
                place = False
            if place:
                obstacle = self._sample_obstacle_type()
                self._cells_since_obstacle = 0
                self._b2b_obstacles += 1
            else:
                self._b2b_obstacles = 0
        elif not is_turn:
            self._b2b_obstacles = 0

        # --- coin / bonus cell (gated) --------------------------------------
        is_coin = False
        if (not is_turn and obstacle == OBST_NONE
                and self._cells_since_bonus >= int(cfg["min_spaces_between_bonus"])):
            if self._rng.random() < 0.5:
                is_coin = True
                self._cells_since_bonus = 0

        return _element(type=ptype, level=level, obstacles=obstacle, is_turn=is_turn,
                        can_turn_left=can_l, can_turn_right=can_r, is_coin=is_coin)

    def _sample_obstacle_type(self) -> int:
        weights = dict(self.cfg["obstacle_type_weights"])
        # Slides (TreeSlide=5, FlameTower=6) only appear past slides_after_distance — the baseline
        # statics have sAllowTreeSlide/FlameTowers = False, so early runs are jump/stumble only.
        if self._distance_real < float(self.cfg["slides_after_distance"]):
            weights = {t: w for t, w in weights.items() if t not in (5, 6)}
        types = list(weights.keys())
        probs = np.array([weights[t] for t in types], dtype=float)
        probs = probs / probs.sum()
        return int(self._rng.choice(types, p=probs))

    # ------------------------------------------------------------- gym: reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_internal_state()
        self._elapsed_steps = 0
        self._last_action = None
        state = self._snapshot()
        self._prev_score = int(state["score"])
        self._prev_distance = float(state["distance"])
        self._prev_state = state
        obs = O.state_to_obs(state)
        info = {"raw_state": state, "score": self._prev_score, "distance": self._prev_distance}
        return obs, info

    # -------------------------------------------------------------- gym: step
    def step(self, action: int, ext_distance: Optional[float] = None):
        a = int(action)
        if a >= C.N_ACTIONS:
            raise ValueError(
                f"action {a} is reserved for reset(); the policy must emit 0..{C.N_ACTIONS - 1}"
            )

        if not self._is_dead:
            self._apply_action(a)
            self._advance(self.step_dt, ext_distance=ext_distance)

        state = self._snapshot()
        score = int(state["score"])
        distance = float(state["distance"])

        # Reward on distance (true objective) or legacy score — mirror the real env (see constants).
        if self.reward_mode == "distance":
            reward = max(0.0, distance - self._prev_distance) * self.reward_scale
        else:
            reward = (score - self._prev_score) * self.reward_scale
        reward += R.shaping_reward(
            a, self._prev_state, self._last_action,
            action_cost=self.action_cost, reaction_bonus=self.reaction_bonus,
            turn_spam_penalty=self.turn_spam_penalty, idle_action_penalty=self.idle_action_penalty,
        )
        terminated = bool(state["is_dead"] or state["is_game_over"])
        if terminated:
            reward -= self.death_penalty

        self._prev_score = score
        self._prev_distance = distance
        self._prev_state = state
        self._last_action = a
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.max_episode_steps

        obs = O.state_to_obs(state)
        info = {
            "raw_state": state,
            "score": score,
            "distance": distance,
            "elapsed_steps": self._elapsed_steps,
            "max_episode_steps": self.max_episode_steps,
            "death_type": self._death_type,
        }
        return obs, float(reward), terminated, truncated, info

    # ----------------------------------------------------------- action model
    def _apply_action(self, a: int) -> None:
        """Translate the action byte into a flag transition / lane shift / turn (SPEC §3)."""
        cfg = self.cfg
        if a == int(C.Action.JUMP):
            # only meaningful from the ground; sets IsJumping for jump_duration.
            if self._jump_t <= 0.0 and self._slide_t <= 0.0:
                self._jump_t = float(cfg["jump_duration"])
        elif a == int(C.Action.SLIDE):
            if self._slide_t <= 0.0 and self._jump_t <= 0.0:
                self._slide_t = float(cfg["slide_duration"])
        elif a == int(C.Action.STRAFE_LEFT):
            if self._lane > -1:
                self._lane -= 1
                self._strafe_t = float(cfg["strafe_duration"])
        elif a == int(C.Action.STRAFE_RIGHT):
            if self._lane < 1:
                self._lane += 1
                self._strafe_t = float(cfg["strafe_duration"])
        elif a in (int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT)):
            self._attempt_turn(a)
        # NOOP -> nothing

    def _attempt_turn(self, a: int) -> None:
        """Turn is valid ONLY at a junction with a matching exit; mid-path turn kills (SPEC §3/§4)."""
        cell = self._current_cell()
        want_left = a == int(C.Action.TURN_LEFT)
        valid = bool(cell.get("is_turn")) and self._turn_window_open() and (
            cell.get("can_turn_left") if want_left else cell.get("can_turn_right")
        )
        if valid:
            self._facing = _LEFT_OF[self._facing] if want_left else _RIGHT_OF[self._facing]
            # consuming the turn: clear the junction's turnability so the cell is "taken".
            cell["is_turn"] = False
            cell["can_turn_left"] = False
            cell["can_turn_right"] = False
            self._cells_since_turn = 0
        else:
            # forced turn off a junction -> the run ends (Fall).
            self._kill(1)

    # --------------------------------------------------------- physics tick
    def _advance(self, dt: float, ext_distance: Optional[float] = None) -> None:
        """Advance velocity, transient timers, distance, cell crossings + scoring for ``dt``.

        ``ext_distance`` (replay only) is the real ``hidden.distance`` for this step. When given,
        cell crossings are driven off the recorded distance delta (units = ``cell_length_real``),
        decoupling path_type from the velocity model — the velocity/score fields still evolve via
        the model so their own tiers stay meaningful. ``None`` = procedural: cross on velocity*dt.
        """
        cfg = self.cfg

        # tick down transient flags.
        self._jump_t = max(0.0, self._jump_t - dt)
        self._slide_t = max(0.0, self._slide_t - dt)
        self._strafe_t = max(0.0, self._strafe_t - dt)
        if self._stumble_t > 0.0:
            self._stumble_t = max(0.0, self._stumble_t - dt)

        # Advance the real-scale DistanceTraveled FIRST (it indexes the velocity curve + drives cell
        # crossings): replay echoes the recorded value; procedural integrates the current velocity.
        if ext_distance is not None:
            moved_cells = max(0.0, float(ext_distance) - self._last_ext_distance)
            self._last_ext_distance = float(ext_distance)
            self._distance_real = float(ext_distance)
            cell_len = float(cfg["cell_length_real"])
        else:
            self._distance_real += self._velocity * dt * float(cfg["dist_per_velsec"])
            cell_len = float(cfg["cell_length"])

        # velocity = calibrated curve(distance) — the real AccCurve is distance-gated, not linear
        # (measured 100->210 over ~4k distance, then plateau). Linear accel is the fallback.
        if self._vel_curve_d is not None:
            self._velocity = float(np.interp(self._distance_real, self._vel_curve_d, self._vel_curve_v))
        else:
            self._velocity = min(float(cfg["max_velocity"]),
                                 self._velocity + float(cfg["accel_per_sec"]) * dt)

        # distance covered this tick, scored ~ velocity*dt (SPEC §5).
        moved = self._velocity * dt
        self._distance += moved
        self._score += moved * float(cfg["score_per_distance"])
        if ext_distance is None:
            moved_cells = moved
        self._dist_into_cell += moved_cells

        # cross as many cell boundaries as we covered (usually 0-1 at 10 Hz).
        guard = 0
        while self._dist_into_cell >= cell_len and not self._is_dead:
            guard += 1
            if guard > 64:
                break
            self._dist_into_cell -= cell_len
            self._cross_into_next_cell()

    def _cross_into_next_cell(self) -> None:
        """The player leaves the current cell -> resolve its obstacle/junction, then step forward."""
        leaving = self._current_cell()
        self._resolve_cell_on_exit(leaving)
        if self._is_dead:
            return
        # move onto the next cell.
        self._cell_index += 1
        self._extend_track()
        entered = self._current_cell()
        if entered.get("is_coin"):
            self._score += float(self.cfg["coin_score"])
            entered["is_coin"] = False

    def _resolve_cell_on_exit(self, cell: dict) -> None:
        """Collision/death check for the cell being passed (SPEC §4)."""
        obst = int(cell.get("obstacles", 0))
        # junction: if it's still a live turn (no valid turn was issued) -> Fall.
        if cell.get("is_turn") and (cell.get("can_turn_left") or cell.get("can_turn_right")):
            self._kill(1)   # Fall
            return
        if obst in _MUST_JUMP:
            if self._jump_t <= 0.0:
                self._kill(2 if obst == OBST_TREE_JUMP else 1)  # Tree / Fall(hole)
            return
        if obst in _MUST_SLIDE:
            if self._slide_t <= 0.0:
                self._kill(5 if obst == OBST_FLAME_TOWER else 3)  # Burnt / Slide
            return
        if obst == OBST_TREE_STUMBLE:
            # survivable: slow down (StumbleSlowDown) + brief stumble flag, not death.
            self._velocity *= float(self.cfg["stumble_slowdown"])
            self._stumble_t = float(self.cfg["stumble_duration"])
        # OBST_NARROW / OBST_NONE / OBST_LEVEL_CHANGE: no death in v0.

    def _kill(self, death_type: int) -> None:
        self._is_dead = True
        self._death_type = int(death_type)

    # -------------------------------------------------------------- snapshot
    def _current_cell(self) -> dict:
        if self._cell_index >= len(self._track):
            self._extend_track()
        return self._track[self._cell_index]

    def _nearest_cell(self) -> dict:
        """The element the hook would report as NearestElement (the current/just-ahead cell)."""
        return self._current_cell()

    def _turn_window_open(self) -> bool:
        """Procedural: a junction is turnable only for the first ``turn_window_frac`` of the cell
        (~2 control-steps, matching the real ~217ms window measured from human turns) -> forces the
        policy to turn PROMPTLY instead of lazily (the cause of dying-too-late on the phone). Replay
        keeps the recorded is_turn (it already encodes the real window) so fidelity is untouched."""
        if self.track_source == "replay":
            return True
        cell_len = float(self.cfg["cell_length"])
        return self._dist_into_cell < cell_len * float(self.cfg["turn_window_frac"])

    def _snapshot(self) -> dict:
        """Produce the raw-state dict (constants.STATE_FIELDS order) incl. 1-cell lookahead."""
        cell = self._nearest_cell()
        obst = int(cell.get("obstacles", 0))
        must_jump = obst in _MUST_JUMP
        must_slide = obst in _MUST_SLIDE
        win = self._turn_window_open()
        is_turn = bool(cell.get("is_turn")) and win
        # CanTurnLeft/Right are heading-relative in the game, but the recorded/cell fields are
        # already the post-heading turnability the hook reports -> expose them directly (windowed).
        can_l = bool(cell.get("can_turn_left")) and win
        can_r = bool(cell.get("can_turn_right")) and win

        # 1-cell lookahead: the cell ahead of the nearest (raw, not window-gated -> anticipation).
        self._extend_track()
        nxt = self._track[self._cell_index + 1] if self._cell_index + 1 < len(self._track) else {}
        nobst = int(nxt.get("obstacles", 0))

        on_ground = self._jump_t <= 0.0 and not self._is_dead
        state = {
            "is_dead": bool(self._is_dead),
            "is_jumping": self._jump_t > 0.0,
            "is_sliding": self._slide_t > 0.0,
            "is_on_ground": bool(on_ground),
            "is_stumbling": self._stumble_t > 0.0,
            "has_invincibility": False,
            "has_angel_wings": False,
            "has_vacuum": False,
            "has_boost": False,
            "is_mega_boost": False,
            "is_strafing": self._strafe_t > 0.0,
            "score": int(self._score),
            "velocity": float(self._velocity),
            "path_type": int(cell.get("type", 1)),
            "path_level": int(cell.get("level", 2)),
            "facing": int(self._facing),
            "must_jump": bool(must_jump),
            "must_slide": bool(must_slide),
            "is_game_over": bool(self._is_dead),   # game-over follows death in this model
            "can_turn_left": bool(can_l),
            "can_turn_right": bool(can_r),
            "is_turn": bool(is_turn),
            "distance": float(self._distance_real),
            "next_is_turn": bool(nxt.get("is_turn")),
            "next_can_turn_left": bool(nxt.get("can_turn_left")),
            "next_can_turn_right": bool(nxt.get("can_turn_right")),
            "next_must_jump": nobst in _MUST_JUMP,
            "next_must_slide": nobst in _MUST_SLIDE,
        }
        return state

    # ------------------------------------------- fidelity harness: raw/set state
    def raw_state(self) -> dict:
        """Current 22-field raw-state dict (for the diff harness)."""
        return self._snapshot()

    def set_state(self, raw_state: dict, hidden: Optional[dict] = None) -> None:
        """Force the sim to a given real (raw_state, hidden) — for one-step fidelity tests.

        ``raw_state`` is the 22-field dict; ``hidden`` carries the extra replay state
        (distance, lane, the active-timer remainders, death_type) per SIM_CONTRACT §1.
        Missing/``None`` hidden fields fall back to sensible derivations from raw_state.
        """
        hidden = hidden or {}
        s = raw_state

        self._is_dead = bool(s.get("is_dead", False))
        self._death_type = hidden.get("death_type")
        self._velocity = float(s.get("velocity", self.cfg["start_velocity"]))
        self._facing = int(s.get("facing", DIR_N)) if int(s.get("facing", DIR_N)) > 0 else DIR_N
        self._score = float(s.get("score", self.cfg["start_score"]))

        lane = hidden.get("lane")
        self._lane = int(lane) if lane is not None else 0
        self._distance = float(hidden.get("distance") or 0.0)
        self._distance_real = float(hidden.get("distance") or self.cfg["start_distance"])
        self._ground_height = float(hidden.get("ground_height") or 0.0)

        # Reconstruct transient timers: prefer the recorded remainders, else derive a full
        # duration from the boolean flag (so the flag stays set for one step at least).
        self._jump_t = self._timer_from(hidden.get("slide_t") if False else None,
                                        s.get("is_jumping"), self.cfg["jump_duration"],
                                        explicit=hidden.get("jump_t"))
        self._slide_t = self._timer_from(None, s.get("is_sliding"), self.cfg["slide_duration"],
                                        explicit=hidden.get("slide_t"))
        self._strafe_t = self._timer_from(None, s.get("is_strafing"), self.cfg["strafe_duration"],
                                         explicit=hidden.get("strafe_t"))
        self._stumble_t = self._timer_from(None, s.get("is_stumbling"), self.cfg["stumble_duration"],
                                          explicit=hidden.get("stumble_t"))

        # Seed the current cell from the raw fields so raw_state() echoes it back exactly.
        cell = _element(
            type=int(s.get("path_type", 1)) if int(s.get("path_type", -1)) >= 0 else 1,
            level=int(s.get("path_level", 2)) if int(s.get("path_level", -1)) >= 0 else 2,
            obstacles=self._obstacle_from_flags(s),
            is_turn=bool(s.get("is_turn")),
            can_turn_left=bool(s.get("can_turn_left")),
            can_turn_right=bool(s.get("can_turn_right")),
        )
        # rebuild the track window with this cell as the current one.
        if self.track_source == "replay":
            # The seeded current cell IS injected element 0 (the recorded nearest cell == the
            # front of the recorded track window). So the NEXT injected cell to consume is
            # index 1 -- starting the cursor at 0 duplicates element 0 (observed: the Temple
            # start-piece (type 37) appeared twice, holding path_type=37 for ~11 steps instead
            # of the real ~4).
            self._replay_cursor = 1
            self._track = [cell]
            self._cell_index = 0
            # Anchor cell-crossing to the recorded distance: the player starts partway into the
            # (partial) Temple start-piece, so seed dist_into_cell from the calibrated offset.
            self._last_ext_distance = float(self._distance)
            self._dist_into_cell = float(self.cfg["replay_start_into_cell"])
            self._extend_track()
        else:
            self._track = [cell]
            self._cell_index = 0
            self._dist_into_cell = 0.0
            self._extend_track()

        self._prev_state = self._snapshot()
        self._prev_score = int(self._prev_state["score"])

    @staticmethod
    def _timer_from(_unused, flag, full_duration, explicit=None) -> float:
        if explicit is not None:
            return max(0.0, float(explicit))
        return float(full_duration) if bool(flag) else 0.0

    @staticmethod
    def _obstacle_from_flags(s: dict) -> int:
        """Pick a representative obstacle enum consistent with must_jump/must_slide."""
        if s.get("must_jump"):
            return OBST_HOLE
        if s.get("must_slide"):
            return OBST_TREE_SLIDE
        return OBST_NONE

    def render(self):
        return None

    def close(self):
        return None


# Register on import (SIM_CONTRACT §2). The env truncates itself via max_episode_steps, so
# register without a TimeLimit wrapper (mirrors TempleRun-v0). Guard against double-register
# (this module is imported by __init__ AND runnable as -m) to avoid gymnasium's override warning.
try:
    from gymnasium.envs.registration import register as _register
    from gymnasium.envs.registration import registry as _registry

    if "TempleRunSim-v0" not in _registry:
        _register(
            id="TempleRunSim-v0",
            entry_point="templerun_rl.sim_env:TempleRunSimEnv",
            max_episode_steps=None,
            disable_env_checker=False,
        )
except Exception:  # noqa: BLE001  (re-import / already-registered is fine)
    pass


# =============================================================================
# Smoke test: 1 procedural episode with a reactive policy. Asserts survival > 50
# steps and score > 0, then prints a summary. Run: python -m templerun_rl.sim_env
# =============================================================================
def _reactive_policy(state: dict, last_action: int) -> int:
    """A simple hand-coded reactive controller (jump/slide/turn on the cue)."""
    if state.get("must_jump"):
        return int(C.Action.JUMP)
    if state.get("must_slide"):
        return int(C.Action.SLIDE)
    if state.get("is_turn"):
        if state.get("can_turn_left"):
            return int(C.Action.TURN_LEFT)
        if state.get("can_turn_right"):
            return int(C.Action.TURN_RIGHT)
    return int(C.Action.NOOP)


def _smoke_test() -> None:
    env = TempleRunSimEnv(track_source="procedural", seed=7)
    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs), "obs not in observation_space"
    assert obs.shape == (O.OBS_LEN,), f"obs shape {obs.shape} != ({O.OBS_LEN},)"

    state = info["raw_state"]
    last_action = int(C.Action.NOOP)
    total_reward = 0.0
    steps = 0
    start_score = state["score"]
    max_score = start_score
    deaths = 0

    for _ in range(400):
        action = _reactive_policy(state, last_action)
        obs, reward, terminated, truncated, info = env.step(action)
        last_action = action
        state = info["raw_state"]
        total_reward += reward
        steps += 1
        max_score = max(max_score, state["score"])
        if terminated or truncated:
            deaths += int(terminated)
            break

    score_gain = max_score - start_score
    print("=" * 60)
    print("TempleRunSim-v0 smoke test (procedural, reactive policy)")
    print("=" * 60)
    print(f"  steps survived   : {steps}")
    print(f"  terminated/death : {bool(deaths)}  (death_type={info.get('death_type')})")
    print(f"  start score      : {start_score}")
    print(f"  max score        : {max_score}   (gain {score_gain})")
    print(f"  total reward     : {total_reward:+.3f}")
    print(f"  final velocity   : {state['velocity']:.1f}")
    print(f"  obs space        : {env.observation_space}")
    print(f"  action space     : {env.action_space}")

    # also verify replay mode + set_state round-trips (used by the fidelity harness).
    sample_track = [
        _element(type=1, level=2, obstacles=OBST_HOLE),
        _element(type=1, level=2, obstacles=OBST_TREE_SLIDE),
        _element(type=3, level=2, is_turn=True, can_turn_left=True),
    ]
    renv = TempleRunSimEnv(track_source=sample_track, seed=1)
    robs, rinfo = renv.reset()
    renv.set_state(rinfo["raw_state"], {"distance": 0.0, "lane": 0})
    rs = renv.raw_state()
    assert set(rs.keys()) == set(C.STATE_FIELDS), "raw_state keys != STATE_FIELDS"
    print(f"  replay/set_state : OK (raw_state has all {len(C.STATE_FIELDS)} fields)")

    assert steps > 50, f"reactive policy died too early ({steps} steps)"
    assert score_gain > 0, f"no score accrued (gain={score_gain})"
    print("\nSMOKE TEST PASSED  (survived >50 steps, scored >0)")


if __name__ == "__main__":
    _smoke_test()
