"""Batched GPU port of ``TempleRunSimEnv`` — thousands of parallel envs as torch tensors.

GOAL: byte-identical *mechanics* to ``sim_env.py`` (the device-calibrated reference), at
GPU-batched throughput. The CPU sim stays the source of truth: every mechanic here mirrors
its exact order of operations, and ``scripts/sim_equivalence.py`` diffs the two step-by-step
on fixed tracks + statistically on procedural rollouts. If the two ever disagree, the CPU
sim wins — fix THIS file.

What is ported (procedural/training mode, the full mechanic stack):
  velocity curve + stumble recovery, analog-tilt lateral lag (+ strafe exec latency),
  turn model (direct window / hook swipe-queue with late-cp gate / approach buffer + TTL /
  lane carry / tail risk / linger + double-turn stumble / turn-on-straight kill),
  jump model (exec latency + jitter, drop prob, strafing refusal, gap-lip refusal),
  obstacles (holes, wide gaps + lip grace, step-up crevasse apex parabola, narrow strips +
  bands, slides + jump-into-slide, stumble 2-strike, cliff-edge riding), zone transitions,
  compound stacked sequences, difficulty ramp, coins, curriculum deep starts, per-episode
  domain randomization, and the exact obs (obs.py layout) + reward (reward.py shaping).

NOT ported (stays on the CPU sim, by design): replay mode (``track_source`` list +
``ext_distance`` / ``set_state`` fidelity replays) — the fidelity harness runs there.
Dead config keys the CPU sim defines but never reads (``turn_while_strafing_kills``,
``strafe_off_edge_kills``, ``dr_turn_window_frac`` jitter, ``lane_recenter_time``…) are
not implemented here either: the port replicates *effective* behavior.

Interfaces:
  - ``TorchTempleRunVecEnv`` — an SB3 ``VecEnv`` with ``num_envs`` batched envs on ``device``.
  - ``TorchSimGymAdapter`` — a single-env gymnasium wrapper (batch=1, CPU) exposing the same
    API surface the tooling uses (``reset/step/raw_state``, ``info['raw_state']``), so the
    fidelity scorecard can run against the torch sim via a registry alias.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from . import constants as C
from . import obs as O
from .sim_env import (  # single source of truth for knobs + enums
    DEFAULT_SIM_CONFIG,
    OBST_NONE, OBST_HOLE, OBST_NARROW, OBST_TREE_STUMBLE, OBST_TREE_JUMP,
    OBST_TREE_SLIDE, OBST_FLAME_TOWER, OBST_LEVEL_CHANGE,
)

# facing rotation lookup (PathDirection: None0 N1 E2 S3 W4)
_LEFT_LUT = torch.tensor([0, 4, 1, 2, 3], dtype=torch.int64)
_RIGHT_LUT = torch.tensor([0, 2, 3, 4, 1], dtype=torch.int64)

A_SL, A_SR, A_JUMP, A_SLIDE, A_TL, A_TR, A_NOOP = 0, 1, 2, 3, 4, 5, 6

TRACK_LEN = 13          # ring size incl. current = CPU sim's lookahead(12)+1 (same distance-gated
                        # generation horizon; the obs only reads indices 0-2)
QUEUE_LEN = 16          # compound-sequence FIFO capacity (max real length ~10)


def _merged_cfg(config: Optional[dict]) -> dict:
    cfg = dict(DEFAULT_SIM_CONFIG)
    if config:
        for k, v in config.items():
            if k == "obstacle_type_weights" and isinstance(v, dict):
                w = dict(DEFAULT_SIM_CONFIG["obstacle_type_weights"])
                w.update(v)
                cfg[k] = w
            else:
                cfg[k] = v
    return cfg


class TorchTempleRunVecEnv(VecEnv):
    """Batched Temple Run sim on ``device``. See module docstring."""

    def __init__(
        self,
        num_envs: int,
        config: Optional[dict] = None,
        device: str = "cuda",
        seed: int = 0,
        step_dt: float = C.DEFAULT_STEP_DT,
        max_episode_steps: int = C.DEFAULT_MAX_EPISODE_STEPS,
        reward_scale: float = C.DEFAULT_REWARD_SCALE,
        reward_mode: str = C.DEFAULT_REWARD_MODE,
        death_penalty: float = C.DEFAULT_DEATH_PENALTY,
        action_cost: float = 0.0,
        reaction_bonus: float = 0.0,
        turn_spam_penalty: float = 0.0,
        idle_action_penalty: float = 0.0,
        redundant_action_penalty: float = 0.0,
        strafe_penalty: float = 0.0,
        narrow_hold_bonus: float = 0.0,
        aim_center_weight: float = 0.0,
        velocity_drop_penalty: float = 0.0,
        off_center_penalty: float = 0.0,
        slide_jump_penalty: float = 0.0,
        action_history_len: int = 0,
        frame_stack: int = 1,
        injected_track: Optional[list] = None,   # test hook: fixed track (list of cell dicts)
    ):
        self.dev = torch.device(device)
        self.cfg = _merged_cfg(config)
        self.B = int(num_envs)
        self.step_dt = float(step_dt)
        self.max_episode_steps = int(max_episode_steps)
        self.reward_scale = float(reward_scale)
        self.reward_mode = str(reward_mode)
        self.death_penalty = float(death_penalty)
        self.action_cost = float(action_cost)
        self.reaction_bonus = float(reaction_bonus)
        self.turn_spam_penalty = float(turn_spam_penalty)
        self.idle_action_penalty = float(idle_action_penalty)
        self.redundant_action_penalty = float(redundant_action_penalty)
        self.strafe_penalty = float(strafe_penalty)
        self.narrow_hold_bonus = float(narrow_hold_bonus)
        self.aim_center_weight = float(aim_center_weight)
        self.velocity_drop_penalty = float(velocity_drop_penalty)
        self.off_center_penalty = float(off_center_penalty)
        self.slide_jump_penalty = float(slide_jump_penalty)
        self.K = int(action_history_len)
        self.FS = max(1, int(frame_stack))

        obs_space = O.make_observation_space(self.FS, self.K)
        act_space = gym.spaces.Discrete(C.N_ACTIONS)
        VecEnv.__init__(self, self.B, obs_space, act_space)
        self.render_mode = None

        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(int(seed))

        cfg = self.cfg
        d, B = self.dev, self.B
        # float64 for per-env scalars: the CPU reference sim is float64, and borderline events
        # (cell-crossing/turn-window thresholds) must not diverge from accumulated float32 drift.
        # Cost is negligible (elementwise [B] ops); the [B,L] track is int/bool, the obs float32.
        f32 = dict(dtype=torch.float64, device=d)
        i64 = dict(dtype=torch.int64, device=d)
        bl = dict(dtype=torch.bool, device=d)

        # ---- static config tensors -----------------------------------------
        vc = cfg.get("velocity_curve") or []
        self.curve_d = torch.tensor([p[0] for p in vc], **f32) if vc else None
        self.curve_v = torch.tensor([p[1] for p in vc], **f32) if vc else None
        ow = cfg["obstacle_type_weights"]
        self.obst_types = torch.tensor(sorted(ow.keys()), **i64)
        self.obst_w = torch.tensor([float(ow[int(t)]) for t in self.obst_types], **f32)
        self.obst_slide_mask = torch.tensor(
            [int(t) in (OBST_TREE_SLIDE, OBST_FLAME_TOWER) for t in self.obst_types], **bl)
        lw = cfg["narrow_len_weights"]
        self.nlen_vals = torch.tensor(sorted(int(k) for k in lw.keys()), **i64)
        self.nlen_w = torch.tensor([float(lw[int(k)]) for k in self.nlen_vals], **f32)
        self.LEFT_LUT = _LEFT_LUT.to(d)
        self.RIGHT_LUT = _RIGHT_LUT.to(d)
        self.must_jump_lut = torch.zeros(8, **bl)
        for t in (OBST_HOLE, OBST_TREE_JUMP, OBST_TREE_STUMBLE, OBST_LEVEL_CHANGE):
            self.must_jump_lut[t] = True
        self.must_slide_lut = torch.zeros(8, **bl)
        for t in (OBST_TREE_SLIDE, OBST_FLAME_TOWER):
            self.must_slide_lut[t] = True

        # injected fixed track (test hook)
        self.injected = None
        if injected_track is not None:
            self.injected = self._track_tensor_from_list(injected_track)
        self.inj_cursor = torch.zeros(B, **i64)

        # ---- track ring: index 0 = current cell ----------------------------
        L = TRACK_LEN
        self.t_type = torch.ones(B, L, **i64)
        self.t_level = torch.full((B, L), int(cfg["default_path_level"]), **i64)
        self.t_obst = torch.zeros(B, L, **i64)
        self.t_turn = torch.zeros(B, L, **bl)
        self.t_canl = torch.zeros(B, L, **bl)
        self.t_canr = torch.zeros(B, L, **bl)
        self.t_coin = torch.zeros(B, L, **bl)
        self.t_wide = torch.zeros(B, L, **bl)
        self.t_stepup = torch.zeros(B, L, **bl)
        self.t_nlane = torch.zeros(B, L, **i64)   # 0 = none, -1/+1 = side strips

        # ---- compound FIFO ---------------------------------------------------
        Q = QUEUE_LEN
        self.q_type = torch.zeros(B, Q, **i64)
        self.q_level = torch.zeros(B, Q, **i64)
        self.q_obst = torch.zeros(B, Q, **i64)
        self.q_nlane = torch.zeros(B, Q, **i64)
        self.q_len = torch.zeros(B, **i64)

        # ---- generation state ----------------------------------------------
        self.since_obst = torch.zeros(B, **i64)
        self.since_turn = torch.zeros(B, **i64)
        self.since_bonus = torch.zeros(B, **i64)
        self.since_zone = torch.zeros(B, **i64)
        self.zone_level = torch.full((B,), int(cfg["default_path_level"]), **i64)
        self.b2b = torch.zeros(B, **i64)
        self.pend_gap = torch.zeros(B, **i64)
        self.pend_gap_obst = torch.zeros(B, **i64)
        self.pend_gap_type = torch.ones(B, **i64)
        self.pend_gap_level = torch.full((B,), 2, **i64)
        self.pend_gap_stepup = torch.zeros(B, **bl)
        self.pend_nar = torch.zeros(B, **i64)
        self.pend_nar_lane = torch.zeros(B, **i64)
        self.pend_nar_type = torch.full((B,), 17, **i64)
        # jump->slide / jump->narrow tail after a wide-gap: counter = clear cells then a PAYLOAD (slide
        # cell OR a narrow strip). pay = payload length (1=slide, strip=narrow); nlane 0=slide, +/-1=narrow.
        self.pend_postgap = torch.zeros(B, **i64)
        self.pend_postgap_obst = torch.zeros(B, **i64)
        self.pend_postgap_type = torch.ones(B, **i64)
        self.pend_postgap_level = torch.full((B,), 2, **i64)
        self.pend_postgap_pay = torch.zeros(B, **i64)
        self.pend_postgap_nlane = torch.zeros(B, **i64)

        # ---- player state ----------------------------------------------------
        self.velocity = torch.full((B,), float(cfg["start_velocity"]), **f32)
        self.lane = torch.zeros(B, **f32)
        self.lat_cmd = torch.zeros(B, **f32)
        self.lat_target = torch.zeros(B, **f32)
        self.slat = int(cfg.get("strafe_exec_lat_frames", 0))
        self.lat_buf = torch.zeros(B, max(1, self.slat), **f32)
        self.dist_real = torch.full((B,), float(cfg["start_distance"]), **f32)
        self.distance = torch.zeros(B, **f32)
        self.dist_in_cell = torch.zeros(B, **f32)
        self.cell_counter = torch.zeros(B, **f32)   # absolute cell index (float: math w/ cp)
        self.facing = torch.ones(B, **i64)
        self.is_dead = torch.zeros(B, **bl)
        self.death_type = torch.zeros(B, **i64)
        self.jump_t = torch.zeros(B, **f32)
        self.slide_t = torch.zeros(B, **f32)
        self.strafe_t = torch.zeros(B, **f32)
        self.stumble_t = torch.zeros(B, **f32)
        self.stumble_kill_t = torch.full((B,), float(cfg["stumble_kill_window"]), **f32)
        self.stumble_count = torch.zeros(B, **i64)
        self.stumble_vel = torch.zeros(B, **f32)
        self.stumble_vel_on = torch.zeros(B, **bl)
        self.takeoff_pos = torch.zeros(B, **f32)
        self.has_takeoff = torch.zeros(B, **bl)
        self.pend_jump = torch.zeros(B, **i64)
        self.cliff_frames = torch.zeros(B, **i64)
        self.post_linger = torch.zeros(B, **i64)
        self.linger_l = torch.zeros(B, **bl)
        self.linger_r = torch.zeros(B, **bl)
        self.pend_turn = torch.zeros(B, **i64)      # 0 none / 1 left / 2 right
        self.pend_turn_ttl = torch.zeros(B, **i64)
        self.gap_grace = torch.zeros(B, **bl)
        self.turn_landed = torch.zeros(B, **bl)
        self.score = torch.full((B,), float(cfg["start_score"]), **f32)
        self.elapsed = torch.zeros(B, **i64)

        # per-episode DR values (filled at reset; = base when DR off)
        self.dr = bool(cfg.get("domain_rand", False))
        self.jump_dur = torch.full((B,), float(cfg["jump_duration"]), **f32)
        self.slide_dur = torch.full((B,), float(cfg["slide_duration"]), **f32)
        self.obst_prob = torch.full((B,), float(cfg["obstacle_prob"]), **f32)
        self.cell_len = torch.full((B,), float(cfg["cell_length"]), **f32)
        self.turn_p = torch.full((B,), float(cfg["turn_prob"]), **f32)
        self.turn_p_late = torch.full((B,), float(cfg["turn_prob_late"]), **f32)
        self.vel_scale = torch.ones(B, **f32)
        # per-episode generation knobs (profile-mixable; = base cfg unless a profile overrides).
        self.min_turns_base = torch.full((B,), float(cfg["min_space_between_turns"]), **f32)
        self.min_turns_late = torch.full((B,), float(cfg["min_space_between_turns_late"]), **f32)
        self.narrows_after = torch.full((B,), float(cfg["narrows_after_distance"]), **f32)
        self.zone_after = torch.full((B,), float(cfg.get("zone_after_distance", 0.0)), **f32)
        self.zone_every = torch.full((B,), int(cfg["zone_change_every_cells"]), **i64)
        self.min_obst = torch.full((B,), int(cfg["min_space_between_obstacles"]), **i64)
        self.compound_p = torch.full((B,), float(cfg.get("compound_prob", 0.0)), **f32)
        self.wide_gap_p = torch.full((B,), float(cfg["wide_gap_prob"]), **f32)
        # EPISODE PROFILES (per-episode single-trap focus mix — user-designed, 2026-07-06):
        # cfg["episode_profiles"] = [{"weight": w, "overrides": {knob: value}}, ...]. At each env
        # reset one profile is sampled; its overrides replace the per-env generation knobs for that
        # EPISODE ONLY. Guarantees constant exposure to rare mechanics (a standard episode shows the
        # first narrow at 1600m — an early policy NEVER gets there) while the full-task profile keeps
        # every skill practiced (no sequential-phase atrophy). None/absent = behavior unchanged.
        self._profiles = cfg.get("episode_profiles") or None
        self._prof_w = None
        if self._profiles:
            w = torch.tensor([float(p.get("weight", 1.0)) for p in self._profiles],
                             dtype=torch.float32, device=d)
            self._prof_w = w / w.sum()

        # episode bookkeeping
        self.ep_ret = torch.zeros(B, **f32)
        self.ep_len = torch.zeros(B, **i64)
        self.last_action = torch.full((B,), A_NOOP, **i64)
        self.act_hist = torch.full((B, max(1, self.K)), A_NOOP, **i64)
        self.frames = torch.zeros(B, self.FS, O.OBS_LEN, dtype=torch.float32, device=d)
        self.prev = None          # previous snapshot (dict of tensors) for shaping
        self.prev_dist = torch.zeros(B, **f32)
        self.prev_score = torch.zeros(B, **f32)
        self._actions = None

        self._reset_mask(torch.ones(B, **bl))

    # ------------------------------------------------------------------ utils
    def _rand(self, n=None):
        return torch.rand(self.B if n is None else n, generator=self.g, device=self.dev)

    def _track_tensor_from_list(self, cells: list) -> dict:
        n = len(cells)
        def col(key, default, dtype):
            return torch.tensor([c.get(key, default) if isinstance(c.get(key, default), (int, float, bool))
                                 else default for c in cells], dtype=dtype, device=self.dev)
        nlane = [0 if c.get("narrow_lane") is None else int(c.get("narrow_lane")) for c in cells]
        return {
            "n": n,
            "type": col("type", 1, torch.int64),
            "level": col("level", 2, torch.int64),
            "obst": col("obstacles", 0, torch.int64),
            "turn": col("is_turn", False, torch.bool),
            "canl": col("can_turn_left", False, torch.bool),
            "canr": col("can_turn_right", False, torch.bool),
            "coin": col("is_coin", False, torch.bool),
            "wide": col("wide_gap", False, torch.bool),
            "stepup": col("stepup", False, torch.bool),
            "nlane": torch.tensor(nlane, dtype=torch.int64, device=self.dev),
        }

    def _interp_vel(self, dist: torch.Tensor) -> torch.Tensor:
        if self.curve_d is None:
            return torch.clamp(self.velocity + float(self.cfg["accel_per_sec"]) * self.step_dt,
                               max=float(self.cfg["max_velocity"]))
        idx = torch.searchsorted(self.curve_d, dist.contiguous())
        idx = idx.clamp(1, len(self.curve_d) - 1)
        d0, d1 = self.curve_d[idx - 1], self.curve_d[idx]
        v0, v1 = self.curve_v[idx - 1], self.curve_v[idx]
        w = ((dist - d0) / (d1 - d0).clamp_min(1e-9)).clamp(0.0, 1.0)
        return v0 + (v1 - v0) * w

    # ------------------------------------------------------------- reset logic
    def _reset_mask(self, m: torch.Tensor) -> None:
        """Reset the envs selected by mask ``m`` (vectorized _reset_internal_state + DR + curriculum)."""
        if not bool(m.any()):
            return
        cfg = self.cfg
        B = self.B
        f = torch.float64

        # --- domain randomization (per episode) ---
        def base(k):
            return float(cfg[k])
        if self.dr:
            def U(key):
                lo, hi = cfg[key]
                return torch.empty(B, dtype=f, device=self.dev).uniform_(float(lo), float(hi), generator=self.g)
            self.jump_dur = torch.where(m, U("dr_jump_duration"), self.jump_dur)
            self.slide_dur = torch.where(m, U("dr_slide_duration"), self.slide_dur)
            self.obst_prob = torch.where(m, U("dr_obstacle_prob"), self.obst_prob)
            self.cell_len = torch.where(m, U("dr_cell_length"), self.cell_len)
            tp = U("dr_turn_prob")
            self.turn_p = torch.where(m, tp, self.turn_p)
            self.turn_p_late = torch.where(m, tp.clamp_min(base("turn_prob_late")), self.turn_p_late)
            self.vel_scale = torch.where(m, U("dr_velocity_scale"), self.vel_scale)
        else:
            for t, k in ((self.jump_dur, "jump_duration"), (self.slide_dur, "slide_duration"),
                         (self.obst_prob, "obstacle_prob"), (self.cell_len, "cell_length"),
                         (self.turn_p, "turn_prob"), (self.turn_p_late, "turn_prob_late")):
                t[m] = base(k)
            self.vel_scale[m] = 1.0
        # generation knobs back to base for the resetting envs, then profile overrides on top
        self.min_turns_base[m] = base("min_space_between_turns")
        self.min_turns_late[m] = base("min_space_between_turns_late")
        self.narrows_after[m] = base("narrows_after_distance")
        self.zone_after[m] = float(cfg.get("zone_after_distance", 0.0))
        self.zone_every[m] = int(cfg["zone_change_every_cells"])
        self.min_obst[m] = int(cfg["min_space_between_obstacles"])
        self.compound_p[m] = float(cfg.get("compound_prob", 0.0))
        self.wide_gap_p[m] = base("wide_gap_prob")
        if self._profiles:
            self._apply_profiles(m)

        # --- player state ---
        self.velocity[m] = base("start_velocity")
        self.facing[m] = 1
        self.lane[m] = 0.0
        self.lat_cmd[m] = 0.0
        self.lat_target[m] = 0.0
        self.lat_buf[m] = 0.0
        self.distance[m] = 0.0
        self.dist_real[m] = base("start_distance")
        # curriculum deep starts
        cmax = float(cfg.get("curriculum_max_start_dist", 0.0))
        if cmax > 0.0 and self.injected is None:
            frac = float(cfg.get("curriculum_deep_start_frac", 0.5))
            pw = max(0.1, float(cfg.get("curriculum_dist_power", 1.0)))
            deep = m & (self._rand() < frac)
            u = self._rand() ** pw
            dstart = base("start_distance") + u * cmax
            self.dist_real = torch.where(deep, dstart, self.dist_real)
            if self.curve_d is not None:
                self.velocity = torch.where(deep, self._interp_vel(self.dist_real), self.velocity)
        self.dist_in_cell[m] = 0.0
        self.cell_counter[m] = 0.0
        self.is_dead[m] = False
        self.death_type[m] = 0
        self.jump_t[m] = 0.0
        self.slide_t[m] = 0.0
        self.strafe_t[m] = 0.0
        self.stumble_t[m] = 0.0
        self.stumble_kill_t[m] = base("stumble_kill_window")
        self.stumble_count[m] = 0
        self.stumble_vel_on[m] = False
        self.has_takeoff[m] = False
        self.pend_jump[m] = 0
        self.cliff_frames[m] = 0
        self.post_linger[m] = 0
        self.pend_turn[m] = 0
        self.pend_turn_ttl[m] = 0
        self.gap_grace[m] = False
        self.turn_landed[m] = False
        self.score[m] = base("start_score")
        self.elapsed[m] = 0
        self.ep_ret[m] = 0.0
        self.ep_len[m] = 0
        self.last_action[m] = A_NOOP
        self.act_hist[m] = A_NOOP
        self.frames[m] = 0.0

        # --- generation state ---
        self.since_obst[m] = 0
        self.since_turn[m] = 0
        self.since_bonus[m] = 0
        self.since_zone[m] = 0
        self.zone_level[m] = int(cfg["default_path_level"])
        self.b2b[m] = 0
        self.pend_gap[m] = 0
        self.pend_gap_stepup[m] = False
        self.pend_nar[m] = 0
        self.pend_postgap[m] = 0
        self.pend_postgap_pay[m] = 0
        self.q_len[m] = 0
        self.inj_cursor[m] = 0

        # --- track: cell 0 = plain straight start, then fill the ring ---
        for t, v in ((self.t_type, int(cfg["path_type_straight"])), (self.t_level, int(cfg["default_path_level"])),
                     (self.t_obst, 0), (self.t_nlane, 0)):
            t[m] = v
        for t in (self.t_turn, self.t_canl, self.t_canr, self.t_coin, self.t_wide, self.t_stepup):
            t[m] = False
        for i in range(1, TRACK_LEN):
            self._gen_cell(m, i)

        # snapshot for prev/reward baselines
        snap = self._snapshot()
        if self.prev is None:
            self.prev = snap
        else:
            for k in self.prev:
                self.prev[k] = torch.where(m, snap[k], self.prev[k])
        self.prev_dist = torch.where(m, snap["distance"], self.prev_dist)
        self.prev_score = torch.where(m, snap["score"], self.prev_score)
        # seed the frame stack with the reset frame
        obs0 = self._base_obs(snap)
        self.frames[m] = obs0[m].unsqueeze(1).expand(-1, self.FS, -1)

    def _apply_profiles(self, m: torch.Tensor) -> None:
        """Sample one episode profile per resetting env; apply its knob overrides (profile wins
        over DR). Supported override keys map onto the per-env generation tensors below."""
        key_map = {
            "obstacle_prob": self.obst_prob,
            "turn_prob": self.turn_p,
            "turn_prob_late": self.turn_p_late,
            "min_space_between_turns": self.min_turns_base,
            "min_space_between_turns_late": self.min_turns_late,
            "narrows_after_distance": self.narrows_after,
            "zone_after_distance": self.zone_after,
            "zone_change_every_cells": self.zone_every,
            "min_space_between_obstacles": self.min_obst,
            "compound_prob": self.compound_p,
            "wide_gap_prob": self.wide_gap_p,
        }
        idx = torch.multinomial(self._prof_w.expand(self.B, -1), 1, generator=self.g).squeeze(1)
        for i, prof in enumerate(self._profiles):
            pm = m & (idx == i)
            if not bool(pm.any()):
                continue
            for k, v in (prof.get("overrides") or {}).items():
                if k not in key_map:
                    raise ValueError(f"episode_profiles override '{k}' not supported "
                                     f"(supported: {sorted(key_map)})")
                t = key_map[k]
                t[pm] = int(v) if t.dtype == torch.int64 else float(v)

    # ------------------------------------------------------------ generation
    def _gen_cell(self, m: torch.Tensor, slot: int) -> None:
        """Generate ONE new cell for masked envs into track index ``slot`` (mirrors
        ``_next_cell_procedural`` exactly — counters, pendings, queue, zone, obstacles, coins)."""
        if not bool(m.any()):
            return
        cfg = self.cfg
        B = self.B
        dev = self.dev

        if self.injected is not None:
            self._gen_cell_injected(m, slot)
            return

        # counters tick on every generated cell
        self.since_obst = torch.where(m, self.since_obst + 1, self.since_obst)
        self.since_turn = torch.where(m, self.since_turn + 1, self.since_turn)
        self.since_bonus = torch.where(m, self.since_bonus + 1, self.since_bonus)
        self.since_zone = torch.where(m, self.since_zone + 1, self.since_zone)

        # defaults
        o_type = torch.full((B,), int(cfg["path_type_straight"]), dtype=torch.int64, device=dev)
        o_level = self.zone_level.clone()
        o_obst = torch.zeros(B, dtype=torch.int64, device=dev)
        o_turn = torch.zeros(B, dtype=torch.bool, device=dev)
        o_canl = torch.zeros(B, dtype=torch.bool, device=dev)
        o_canr = torch.zeros(B, dtype=torch.bool, device=dev)
        o_coin = torch.zeros(B, dtype=torch.bool, device=dev)
        o_wide = torch.zeros(B, dtype=torch.bool, device=dev)
        o_stepup = torch.zeros(B, dtype=torch.bool, device=dev)
        o_nlane = torch.zeros(B, dtype=torch.int64, device=dev)

        # 1) compound FIFO pop
        srcQ = m & (self.q_len > 0)
        if bool(srcQ.any()):
            o_type = torch.where(srcQ, self.q_type[:, 0], o_type)
            o_level = torch.where(srcQ, self.q_level[:, 0], o_level)
            o_obst = torch.where(srcQ, self.q_obst[:, 0], o_obst)
            o_nlane = torch.where(srcQ, self.q_nlane[:, 0], o_nlane)
            for q in (self.q_type, self.q_level, self.q_obst, self.q_nlane):
                q[srcQ] = torch.roll(q[srcQ], -1, dims=1)
            self.q_len = torch.where(srcQ, self.q_len - 1, self.q_len)
            self.since_obst = torch.where(srcQ, torch.zeros_like(self.since_obst), self.since_obst)

        # 2) wide-gap continuation
        srcG = m & ~srcQ & (self.pend_gap > 0)
        if bool(srcG.any()):
            self.pend_gap = torch.where(srcG, self.pend_gap - 1, self.pend_gap)
            self.since_obst = torch.where(srcG, torch.zeros_like(self.since_obst), self.since_obst)
            o_type = torch.where(srcG, self.pend_gap_type, o_type)
            o_level = torch.where(srcG, self.pend_gap_level, o_level)
            o_obst = torch.where(srcG, self.pend_gap_obst, o_obst)
            o_wide = torch.where(srcG, torch.ones_like(o_wide), o_wide)
            last = srcG & (self.pend_gap == 0) & self.pend_gap_stepup
            o_stepup = torch.where(last, torch.ones_like(o_stepup), o_stepup)
            self.pend_gap_stepup = torch.where(last, torch.zeros_like(self.pend_gap_stepup),
                                               self.pend_gap_stepup)

        # 2.5) jump->slide / jump->narrow tail: leading clear cells then a PAYLOAD (slide cell OR narrow
        # strip), after the wide-gap continuation. counter > pay -> clear; counter <= pay -> payload.
        srcP = m & ~srcQ & ~srcG & (self.pend_postgap > 0)
        if bool(srcP.any()):
            is_clear = srcP & (self.pend_postgap > self.pend_postgap_pay)
            is_pay = srcP & ~is_clear
            is_slide = is_pay & (self.pend_postgap_nlane == 0)
            is_narrow = is_pay & (self.pend_postgap_nlane != 0)
            o_type = torch.where(is_clear,
                                 torch.full_like(o_type, int(cfg["path_type_straight"])), o_type)
            o_type = torch.where(is_slide, self.pend_postgap_type, o_type)
            o_obst = torch.where(is_slide, self.pend_postgap_obst, o_obst)
            narrow_pt = torch.where(self.pend_postgap_nlane < 0,
                                    torch.full_like(o_type, 19), torch.full_like(o_type, 22))
            o_type = torch.where(is_narrow, narrow_pt, o_type)
            o_nlane = torch.where(is_narrow, self.pend_postgap_nlane, o_nlane)
            o_level = torch.where(srcP, self.pend_postgap_level, o_level)
            self.pend_postgap = torch.where(srcP, self.pend_postgap - 1, self.pend_postgap)
            self.since_obst = torch.where(srcP, torch.zeros_like(self.since_obst), self.since_obst)

        # 3) narrow-strip continuation
        srcN = m & ~srcQ & ~srcG & ~srcP & (self.pend_nar > 0)
        if bool(srcN.any()):
            self.pend_nar = torch.where(srcN, self.pend_nar - 1, self.pend_nar)
            self.since_obst = torch.where(srcN, torch.zeros_like(self.since_obst), self.since_obst)
            o_type = torch.where(srcN, self.pend_nar_type, o_type)
            o_level = torch.where(srcN, self.zone_level, o_level)
            o_nlane = torch.where(srcN, self.pend_nar_lane, o_nlane)

        rem = m & ~srcQ & ~srcG & ~srcP & ~srcN
        if bool(rem.any()):
            diff = ((self.dist_real - float(cfg["start_distance"]))
                    / max(1.0, float(cfg["ramp_distance"]))).clamp(0.0, 1.0)
            min_t = self.min_turns_base * (1.0 - diff) + self.min_turns_late * diff
            turn_p = self.turn_p * (1.0 - diff) + self.turn_p_late * diff

            # junction decision
            t_ok = rem & (self.since_turn.float() >= min_t) & (self._rand() < turn_p)
            self.since_turn = torch.where(t_ok, torch.zeros_like(self.since_turn), self.since_turn)
            roll = self._rand()
            left_only = t_ok & (roll < 0.4)
            right_only = t_ok & (roll >= 0.4) & (roll < 0.8)
            both = t_ok & (roll >= 0.8)
            o_turn = torch.where(t_ok, torch.ones_like(o_turn), o_turn)
            o_canl = torch.where(left_only | both, torch.ones_like(o_canl), o_canl)
            o_canr = torch.where(right_only | both, torch.ones_like(o_canr), o_canr)
            o_type = torch.where(left_only | right_only,
                                 torch.full_like(o_type, int(cfg["path_type_turn"])), o_type)
            o_type = torch.where(both, torch.full_like(o_type, int(cfg["path_type_t_junction"])), o_type)

            # zone transition (priority obstacle)
            z_ok = (rem & ~t_ok
                    & (self.since_turn >= int(cfg["min_clear_after_turn"]))
                    & (self.since_obst >= self.min_obst)
                    & (self.since_zone >= self.zone_every)
                    & (self.dist_real >= self.zone_after))
            if bool(z_ok.any()):
                lo, hi = int(cfg["zone_min_level"]), int(cfg["zone_max_level"])
                prev_zone = self.zone_level.clone()
                up = z_ok & (self.zone_level <= lo)
                down = z_ok & ~up & (self.zone_level >= hi)
                mid = z_ok & ~up & ~down
                mid_down = mid & (self._rand() < 0.65)
                nz = self.zone_level + up.long() - down.long() - mid_down.long() + (mid & ~mid_down).long()
                self.zone_level = torch.where(z_ok, nz, self.zone_level)
                self.since_zone = torch.where(z_ok, torch.zeros_like(self.since_zone), self.since_zone)
                self.since_obst = torch.where(z_ok, torch.zeros_like(self.since_obst), self.since_obst)
                self.b2b = torch.where(z_ok, torch.zeros_like(self.b2b), self.b2b)
                o_obst = torch.where(z_ok, torch.full_like(o_obst, OBST_LEVEL_CHANGE), o_obst)
                o_level = torch.where(z_ok, self.zone_level, o_level)
                o_type = torch.where(z_ok, torch.full_like(o_type, 35), o_type)
                o_wide = torch.where(z_ok, torch.ones_like(o_wide), o_wide)
                self.pend_gap = torch.where(z_ok, torch.ones_like(self.pend_gap), self.pend_gap)
                self.pend_gap_obst = torch.where(z_ok, torch.full_like(o_obst, OBST_LEVEL_CHANGE),
                                                 self.pend_gap_obst)
                self.pend_gap_type = torch.where(z_ok, torch.full_like(o_type, 35), self.pend_gap_type)
                self.pend_gap_level = torch.where(z_ok, self.zone_level, self.pend_gap_level)
                self.pend_gap_stepup = torch.where(z_ok, self.zone_level > prev_zone, self.pend_gap_stepup)

            # normal obstacle placement
            ob_ok = (rem & ~t_ok & (o_obst == OBST_NONE)
                     & (self.since_turn >= int(cfg["min_clear_after_turn"]))
                     & (self.since_obst >= self.min_obst))
            place = ob_ok & (self._rand() < self.obst_prob) \
                    & ~(self.b2b >= int(cfg["max_back_to_back_obstacles"]))
            # b2b: placed -> +1; any other non-turn generated-here cell -> 0 (mirrors CPU if/elif)
            nonturn_reset = rem & ~t_ok & ~place & ~z_ok
            self.b2b = torch.where(place, self.b2b + 1,
                                   torch.where(nonturn_reset, torch.zeros_like(self.b2b), self.b2b))
            self.since_obst = torch.where(place, torch.zeros_like(self.since_obst), self.since_obst)

            nar = place & (self.dist_real >= self.narrows_after) \
                  & (self._rand() < float(cfg["narrow_prob"]))
            if bool(nar.any()):
                lane_a = torch.where(self._rand() < 0.5,
                                     torch.full((B,), -1, dtype=torch.int64, device=dev),
                                     torch.ones(B, dtype=torch.int64, device=dev))
                strip = self.nlen_vals[torch.multinomial(
                    self.nlen_w.expand(B, -1), 1, generator=self.g).squeeze(1)]
                comp = nar & (self.dist_real >= float(cfg.get("compound_after_distance", 0.0))) \
                       & (self._rand() < self.compound_p)
                ncomp = nar & ~comp
                ptype = torch.where(lane_a < 0, torch.full((B,), 19, dtype=torch.int64, device=dev),
                                    torch.full((B,), 22, dtype=torch.int64, device=dev))
                o_nlane = torch.where(nar, lane_a, o_nlane)
                o_type = torch.where(nar, ptype, o_type)
                # non-compound: queue strip continuation
                self.pend_nar = torch.where(ncomp, strip - 1, self.pend_nar)
                self.pend_nar_lane = torch.where(ncomp, lane_a, self.pend_nar_lane)
                self.pend_nar_type = torch.where(ncomp, ptype, self.pend_nar_type)
                if bool(comp.any()):
                    self._queue_compound(comp, lane_a, o_level, strip)

            holewide = torch.zeros(B, dtype=torch.bool, device=dev)   # long-hole wide-gap START mask
            obs_sel = place & ~nar
            if bool(obs_sel.any()):
                w = self.obst_w.expand(B, -1).clone()
                early = (self.dist_real < float(cfg["slides_after_distance"])).unsqueeze(1)
                w = torch.where(early & self.obst_slide_mask.unsqueeze(0), torch.zeros_like(w), w)
                typ = self.obst_types[torch.multinomial(w, 1, generator=self.g).squeeze(1)]
                o_obst = torch.where(obs_sel, typ, o_obst)
                holewide = obs_sel & (typ == OBST_HOLE) & (self._rand() < self.wide_gap_p)
                if bool(holewide.any()):
                    o_wide = torch.where(holewide, torch.ones_like(o_wide), o_wide)
                    o_type = torch.where(holewide, torch.full_like(o_type, 12), o_type)
                    self.pend_gap = torch.where(holewide,
                                                torch.full_like(self.pend_gap, int(cfg["wide_gap_extra_cells"])),
                                                self.pend_gap)
                    self.pend_gap_obst = torch.where(holewide, torch.full_like(o_obst, OBST_HOLE),
                                                     self.pend_gap_obst)
                    self.pend_gap_type = torch.where(holewide, torch.full_like(o_type, 12), self.pend_gap_type)
                    self.pend_gap_level = torch.where(holewide, o_level, self.pend_gap_level)
                    self.pend_gap_stepup = torch.where(holewide, torch.zeros_like(self.pend_gap_stepup),
                                                       self.pend_gap_stepup)

            # JUMP->SLIDE / JUMP->STRAFE compounds: after a wide-gap START (crevasse z_ok / long-hole
            # holewide), sometimes queue a SLIDE tail (jumpslide) or a side-NARROW tail (jumpnarrow) within
            # the jump airtime -> airborne-at-slide death / strafe-too-late narrow fall. Each rng gated on
            # its prob>0 FIRST (stream/equiv safe). A wide-gap gets AT MOST one tail (slide rolled first).
            wide_start0 = (z_ok | holewide)
            queued = torch.zeros(B, dtype=torch.bool, device=dev)
            jsp = float(cfg.get("jumpslide_prob", 0.0))
            if jsp > 0.0:
                ws = wide_start0 & (self.dist_real >= float(cfg.get("jumpslide_after_distance", 0.0)))
                js = ws & (self._rand() < jsp)
                if bool(js.any()):
                    gap = int(cfg.get("jumpslide_gap_cells", 1))
                    is_flame = self._rand() < float(cfg.get("jumpslide_flame_frac", 0.5))
                    sob = torch.where(is_flame, torch.full_like(self.pend_postgap, OBST_FLAME_TOWER),
                                      torch.full_like(self.pend_postgap, OBST_TREE_SLIDE))
                    spt = torch.where(is_flame, torch.full_like(self.pend_postgap, 31),
                                      torch.full_like(self.pend_postgap, int(cfg["path_type_straight"])))
                    self.pend_postgap = torch.where(js, torch.full_like(self.pend_postgap, gap + 1),
                                                    self.pend_postgap)
                    self.pend_postgap_pay = torch.where(js, torch.ones_like(self.pend_postgap_pay),
                                                        self.pend_postgap_pay)
                    self.pend_postgap_nlane = torch.where(js, torch.zeros_like(self.pend_postgap_nlane),
                                                          self.pend_postgap_nlane)
                    self.pend_postgap_obst = torch.where(js, sob, self.pend_postgap_obst)
                    self.pend_postgap_type = torch.where(js, spt, self.pend_postgap_type)
                    self.pend_postgap_level = torch.where(js, o_level, self.pend_postgap_level)
                    queued = queued | js
            jnp = float(cfg.get("jumpnarrow_prob", 0.0))
            if jnp > 0.0:
                ws = (wide_start0 & ~queued
                      & (self.dist_real >= float(cfg.get("jumpnarrow_after_distance", 0.0))))
                jn = ws & (self._rand() < jnp)
                if bool(jn.any()):
                    gap = int(cfg.get("jumpnarrow_gap_cells", 1))
                    strip = max(1, int(cfg.get("jumpnarrow_strip_cells", 2)))
                    lane = torch.where(self._rand() < 0.5,
                                       torch.full_like(self.pend_postgap_nlane, -1),
                                       torch.ones_like(self.pend_postgap_nlane))
                    self.pend_postgap = torch.where(jn, torch.full_like(self.pend_postgap, gap + strip),
                                                    self.pend_postgap)
                    self.pend_postgap_pay = torch.where(jn, torch.full_like(self.pend_postgap_pay, strip),
                                                        self.pend_postgap_pay)
                    self.pend_postgap_nlane = torch.where(jn, lane, self.pend_postgap_nlane)
                    self.pend_postgap_level = torch.where(jn, o_level, self.pend_postgap_level)

            # coin
            coin = rem & ~t_ok & (o_obst == OBST_NONE) \
                   & (self.since_bonus >= int(cfg["min_spaces_between_bonus"])) & (self._rand() < 0.5)
            o_coin = torch.where(coin, torch.ones_like(o_coin), o_coin)
            self.since_bonus = torch.where(coin, torch.zeros_like(self.since_bonus), self.since_bonus)

        # write the slot
        self.t_type[:, slot] = torch.where(m, o_type, self.t_type[:, slot])
        self.t_level[:, slot] = torch.where(m, o_level, self.t_level[:, slot])
        self.t_obst[:, slot] = torch.where(m, o_obst, self.t_obst[:, slot])
        self.t_turn[:, slot] = torch.where(m, o_turn, self.t_turn[:, slot])
        self.t_canl[:, slot] = torch.where(m, o_canl, self.t_canl[:, slot])
        self.t_canr[:, slot] = torch.where(m, o_canr, self.t_canr[:, slot])
        self.t_coin[:, slot] = torch.where(m, o_coin, self.t_coin[:, slot])
        self.t_wide[:, slot] = torch.where(m, o_wide, self.t_wide[:, slot])
        self.t_stepup[:, slot] = torch.where(m, o_stepup, self.t_stepup[:, slot])
        self.t_nlane[:, slot] = torch.where(m, o_nlane, self.t_nlane[:, slot])

    def _gen_cell_injected(self, m: torch.Tensor, slot: int) -> None:
        inj = self.injected
        idx = self.inj_cursor.clamp(max=inj["n"] - 1)
        valid = m & (self.inj_cursor < inj["n"])
        neutral = m & ~valid
        self.t_type[:, slot] = torch.where(valid, inj["type"][idx], self.t_type[:, slot])
        self.t_level[:, slot] = torch.where(valid, inj["level"][idx], self.t_level[:, slot])
        self.t_obst[:, slot] = torch.where(valid, inj["obst"][idx], self.t_obst[:, slot])
        self.t_turn[:, slot] = torch.where(valid, inj["turn"][idx], self.t_turn[:, slot])
        self.t_canl[:, slot] = torch.where(valid, inj["canl"][idx], self.t_canl[:, slot])
        self.t_canr[:, slot] = torch.where(valid, inj["canr"][idx], self.t_canr[:, slot])
        self.t_coin[:, slot] = torch.where(valid, inj["coin"][idx], self.t_coin[:, slot])
        self.t_wide[:, slot] = torch.where(valid, inj["wide"][idx], self.t_wide[:, slot])
        self.t_stepup[:, slot] = torch.where(valid, inj["stepup"][idx], self.t_stepup[:, slot])
        self.t_nlane[:, slot] = torch.where(valid, inj["nlane"][idx], self.t_nlane[:, slot])
        if bool(neutral.any()):
            self.t_type[:, slot] = torch.where(neutral, torch.full_like(self.t_type[:, slot],
                                               int(self.cfg["path_type_straight"])), self.t_type[:, slot])
            self.t_level[:, slot] = torch.where(neutral, torch.full_like(self.t_level[:, slot],
                                                int(self.cfg["default_path_level"])), self.t_level[:, slot])
            self.t_obst[:, slot] = torch.where(neutral, torch.zeros_like(self.t_obst[:, slot]),
                                               self.t_obst[:, slot])
            for t in (self.t_turn, self.t_canl, self.t_canr, self.t_coin, self.t_wide, self.t_stepup):
                t[:, slot] = torch.where(neutral, torch.zeros_like(t[:, slot]), t[:, slot])
            self.t_nlane[:, slot] = torch.where(neutral, torch.zeros_like(self.t_nlane[:, slot]),
                                                self.t_nlane[:, slot])
        self.inj_cursor = torch.where(m, self.inj_cursor + 1, self.inj_cursor)

    def _queue_compound(self, m: torch.Tensor, lane_a: torch.Tensor, level: torch.Tensor,
                        strip_a: torch.Tensor) -> None:
        """Vectorized ``_queue_compound``: restA(strip_a-1) -> barrier -> land cells -> stripB(opposite)."""
        cfg = self.cfg
        B = self.B
        dev = self.dev
        land = max(1, int(cfg.get("compound_land_cells", 1)))
        strip_b = self.nlen_vals[torch.multinomial(
            self.nlen_w.expand(B, -1), 1, generator=self.g).squeeze(1)]
        lane_b = -lane_a
        restA = (strip_a - 1).clamp_min(0)
        b_barrier = restA                       # index of the barrier cell
        b_land_end = restA + 1 + land           # first stripB index
        b_end = (b_land_end + strip_b).clamp(max=QUEUE_LEN)

        idx = torch.arange(QUEUE_LEN, device=dev).unsqueeze(0)      # [1, Q]
        in_restA = idx < restA.unsqueeze(1)
        is_barrier = idx == b_barrier.unsqueeze(1)
        in_land = (idx > b_barrier.unsqueeze(1)) & (idx < b_land_end.unsqueeze(1))
        in_stripB = (idx >= b_land_end.unsqueeze(1)) & (idx < b_end.unsqueeze(1))

        ptA = torch.where(lane_a < 0, 19, 22).unsqueeze(1)
        ptB = torch.where(lane_b < 0, 19, 22).unsqueeze(1)
        q_type = torch.where(in_restA, ptA, torch.where(in_stripB, ptB,
                             torch.full((B, QUEUE_LEN), 2, dtype=torch.int64, device=dev)))
        q_obst = torch.where(is_barrier,
                             torch.full((B, QUEUE_LEN), OBST_TREE_JUMP, dtype=torch.int64, device=dev),
                             torch.zeros(B, QUEUE_LEN, dtype=torch.int64, device=dev))
        q_nlane = torch.where(in_restA, lane_a.unsqueeze(1),
                              torch.where(in_stripB, lane_b.unsqueeze(1),
                                          torch.zeros(B, QUEUE_LEN, dtype=torch.int64, device=dev)))
        q_level = level.unsqueeze(1).expand(-1, QUEUE_LEN)

        mm = m.unsqueeze(1)
        self.q_type = torch.where(mm, q_type, self.q_type)
        self.q_obst = torch.where(mm, q_obst, self.q_obst)
        self.q_nlane = torch.where(mm, q_nlane, self.q_nlane)
        self.q_level = torch.where(mm, q_level, self.q_level)
        self.q_len = torch.where(m, b_end, self.q_len)

    # --------------------------------------------------------------- actions
    def _kill(self, m: torch.Tensor, dtype: int) -> None:
        self.is_dead = self.is_dead | m
        self.death_type = torch.where(m & (self.death_type == 0),
                                      torch.full_like(self.death_type, int(dtype)), self.death_type)

    def _kill_t(self, m: torch.Tensor, dtypes: torch.Tensor) -> None:
        self.is_dead = self.is_dead | m
        self.death_type = torch.where(m & (self.death_type == 0), dtypes, self.death_type)

    def _stumble(self, m: torch.Tensor) -> None:
        m = m & ~self.is_dead
        if not bool(m.any()):
            return
        die = m & ((self.stumble_t > 0.0) | (self.stumble_kill_t < float(self.cfg["stumble_kill_window"])))
        self._kill(die, 1)
        ok = m & ~die
        self.velocity = torch.where(ok, self.velocity * float(self.cfg["stumble_slowdown"]), self.velocity)
        self.stumble_vel = torch.where(ok, self.velocity, self.stumble_vel)
        self.stumble_vel_on = self.stumble_vel_on | ok
        self.lane = torch.where(ok, torch.zeros_like(self.lane), self.lane)
        self.strafe_t = torch.where(ok, torch.zeros_like(self.strafe_t), self.strafe_t)
        self.stumble_t = torch.where(ok, torch.full_like(self.stumble_t, float(self.cfg["stumble_duration"])),
                                     self.stumble_t)
        self.stumble_kill_t = torch.where(ok, torch.zeros_like(self.stumble_kill_t), self.stumble_kill_t)
        self.stumble_count = torch.where(ok, self.stumble_count + 1, self.stumble_count)

    def _cp(self) -> torch.Tensor:
        return (self.dist_in_cell / self.cell_len.clamp_min(1e-6)).clamp(0.0, 1.0)

    def _long_pos(self) -> torch.Tensor:
        return self.cell_counter + self._cp()

    def _apply_action(self, a: torch.Tensor, alive: torch.Tensor) -> None:
        cfg = self.cfg
        # --- JUMP ---
        can_jump = alive & (a == A_JUMP) & (self.jump_t <= 0.0) & (self.slide_t <= 0.0)
        p_drop = float(cfg.get("jump_drop_prob", 0.0))
        if p_drop > 0.0:
            can_jump = can_jump & ~(self._rand() < p_drop)
        p_ref = float(cfg.get("jump_refuse_strafing_prob", 0.6))
        refuse = can_jump & ((self.lat_target - self.lane).abs() > 1.8) & (self._rand() < p_ref)
        can_jump = can_jump & ~refuse
        lip = can_jump & self.t_wide[:, 0] & (self._cp() > float(cfg.get("gap_jump_lip_cp", 0.2)))
        can_jump = can_jump & ~lip
        lat = int(cfg.get("jump_exec_lat_frames", 0))
        if lat > 0:
            sched = can_jump & (self.pend_jump <= 0)
            jit = int(cfg.get("jump_exec_lat_jitter", 0))
            if jit > 0:
                delay = lat + torch.randint(-jit, jit + 1, (self.B,), generator=self.g, device=self.dev)
            else:
                delay = torch.full((self.B,), lat, dtype=torch.int64, device=self.dev)
            self.pend_jump = torch.where(sched, delay.clamp_min(1), self.pend_jump)
        else:
            self.jump_t = torch.where(can_jump, self.jump_dur, self.jump_t)
            lp = self._long_pos()
            self.takeoff_pos = torch.where(can_jump, lp, self.takeoff_pos)
            self.has_takeoff = self.has_takeoff | can_jump

        # --- SLIDE ---
        can_slide = alive & (a == A_SLIDE) & (self.slide_t <= 0.0) & (self.jump_t <= 0.0)
        self.slide_t = torch.where(can_slide, self.slide_dur, self.slide_t)

        # --- STRAFE / NOOP (analog tilt targets) ---
        edge = float(cfg["lane_edge"])
        self.lat_cmd = torch.where(alive & (a == A_SL), torch.full_like(self.lat_cmd, -edge), self.lat_cmd)
        self.lat_cmd = torch.where(alive & (a == A_SR), torch.full_like(self.lat_cmd, +edge), self.lat_cmd)
        self.lat_cmd = torch.where(alive & (a == A_NOOP), torch.zeros_like(self.lat_cmd), self.lat_cmd)

        # --- TURN ---
        t_mask = alive & ((a == A_TL) | (a == A_TR))
        if bool(t_mask.any()):
            self._attempt_turn(t_mask, a == A_TL)

    def _attempt_turn(self, m: torch.Tensor, want_left: torch.Tensor) -> None:
        cfg = self.cfg
        # over-rotation during the linger -> stumble
        linger_hit = m & (self.post_linger > 0)
        self._stumble(linger_hit)
        m = m & ~linger_hit
        if not bool(m.any()):
            return
        cell_turn = self.t_turn[:, 0]
        side_ok = torch.where(want_left, self.t_canl[:, 0], self.t_canr[:, 0])
        cp = self._cp()

        rem = m
        # hook swipe-queue (16j+)
        if cfg.get("hook_turn_buffer"):
            hb = rem & cell_turn & side_ok
            # decision cp -> PROCESSING cp: byte arrives turn_send_lat_frames later (velocity-scaled);
            # at high speed an on-junction decision lands past the deadline -> dead-letter (mirror of CPU).
            slat = float(cfg.get("turn_send_lat_frames", 0.0))
            proc_cp = cp + (self.velocity * slat * float(self.step_dt) / self.cell_len.clamp_min(1e-6)
                            if slat > 0.0 else 0.0)
            fire = hb & (proc_cp <= float(cfg.get("turn_buffer_late_cp", 0.56)))
            if bool(fire.any()):
                self._execute_turn(fire, want_left,
                                   torch.full_like(cp, float(cfg["turn_center_cprog"])),
                                   from_buffer=torch.ones_like(fire))
            rem = rem & ~hb    # fired OR silent late miss
        # direct window
        early, late = float(cfg["turn_exec_early"]), float(cfg["turn_exec_late"])
        direct = rem & cell_turn & (cp >= early) & (cp <= late) & side_ok
        if bool(direct.any()):
            self._execute_turn(direct, want_left, cp, from_buffer=torch.zeros_like(direct))
        rem = rem & ~direct
        # approach-cell latch
        if cfg.get("turn_approach_buffer"):
            nx_ok = torch.where(want_left, self.t_canl[:, 1], self.t_canr[:, 1])
            app = rem & ~cell_turn & self.t_turn[:, 1] & nx_ok
            self.pend_turn = torch.where(app, torch.where(want_left, torch.ones_like(self.pend_turn),
                                                          torch.full_like(self.pend_turn, 2)), self.pend_turn)
            self.pend_turn_ttl = torch.where(app, torch.full_like(self.pend_turn_ttl,
                                             int(cfg.get("turn_buffer_ttl", 3))), self.pend_turn_ttl)
            rem = rem & ~app
        # mistimed near a junction -> stumble; on a plain straight -> fatal
        near = rem & (cell_turn | self.t_turn[:, 1])
        self._stumble(near)
        self._kill(rem & ~near, 1)

    def _execute_turn(self, m: torch.Tensor, want_left: torch.Tensor, swipe_cp: torch.Tensor,
                      from_buffer: torch.Tensor) -> None:
        cfg = self.cfg
        carry = torch.where(from_buffer, float(cfg.get("turn_lane_carry_buffered", 1.0)),
                            float(cfg["turn_lane_carry"]))
        lat_frames = float(cfg.get("turn_exec_lat_frames", 0.0))
        if lat_frames > 0.0:
            swipe_cp = torch.where(~from_buffer,
                                   swipe_cp + self.velocity * lat_frames * self.step_dt / self.cell_len,
                                   swipe_cp)
        sign = torch.where(want_left, 1.0, -1.0)
        offset = torch.where(from_buffer, torch.zeros_like(swipe_cp),
                             sign * (swipe_cp - float(cfg["turn_center_cprog"]))
                             * float(cfg["turn_offset_scale"]) + self.lane * carry)
        # facing flips for the whole exec mask (even a fatal landing flips it — CPU order)
        newf = torch.where(want_left, self.LEFT_LUT[self.facing], self.RIGHT_LUT[self.facing])
        self.facing = torch.where(m, newf, self.facing)
        die1 = m & (offset.abs() > float(cfg["turn_land_kill"]))
        self._kill(die1, 1)
        surv = m & ~die1
        pmax = float(cfg.get("turn_tail_pmax", 0.20))
        if pmax > 0.0:
            lo = float(cfg.get("turn_tail_cp_lo", 0.50))
            hi = float(cfg.get("turn_tail_cp_hi", 0.64))
            p = ((swipe_cp - lo) / max(1e-6, hi - lo) * pmax).clamp(max=pmax)
            die2 = surv & (swipe_cp > lo) & (self._rand() < p)
            self._kill(die2, 1)
            surv = surv & ~die2
        self.lane = torch.where(surv, offset, self.lane)
        self.turn_landed = self.turn_landed | surv
        for t in (self.t_turn, self.t_canl, self.t_canr):
            t[:, 0] = torch.where(surv, torch.zeros_like(t[:, 0]), t[:, 0])
        self.since_turn = torch.where(surv, torch.zeros_like(self.since_turn), self.since_turn)
        self.pend_turn = torch.where(surv, torch.zeros_like(self.pend_turn), self.pend_turn)
        n = int(cfg.get("turn_linger_frames", 0))
        if n > 0:
            self.post_linger = torch.where(surv, torch.full_like(self.post_linger, n), self.post_linger)
            self.linger_l = torch.where(surv, want_left, self.linger_l)
            self.linger_r = torch.where(surv, ~want_left, self.linger_r)

    # ---------------------------------------------------------------- physics
    def _advance(self, alive: torch.Tensor) -> None:
        cfg = self.cfg
        dt = self.step_dt

        # pending (latency-buffered) jump
        pj = alive & (self.pend_jump > 0)
        self.pend_jump = torch.where(pj, self.pend_jump - 1, self.pend_jump)
        fire = pj & (self.pend_jump == 0) & (self.jump_t <= 0.0) & (self.slide_t <= 0.0)
        self.jump_t = torch.where(fire, self.jump_dur, self.jump_t)
        lp = self._long_pos()
        self.takeoff_pos = torch.where(fire, lp, self.takeoff_pos)
        self.has_takeoff = self.has_takeoff | fire

        # transient timers
        self.jump_t = torch.where(alive, (self.jump_t - dt).clamp_min(0.0), self.jump_t)
        self.slide_t = torch.where(alive, (self.slide_t - dt).clamp_min(0.0), self.slide_t)
        self.strafe_t = torch.where(alive, (self.strafe_t - dt).clamp_min(0.0), self.strafe_t)

        # strafe-command latency -> active lateral target
        if self.slat > 0:
            popped = self.lat_buf[:, 0].clone()
            self.lat_buf = torch.roll(self.lat_buf, -1, dims=1)
            self.lat_buf[:, -1] = torch.where(alive, self.lat_cmd, self.lat_buf[:, -1])
            self.lat_target = torch.where(alive, popped, self.lat_target)
        else:
            self.lat_target = torch.where(alive, self.lat_cmd, self.lat_target)

        # analog-tilt lateral lag (skipped on the turn-landing frame)
        edge = float(cfg["lane_edge"])
        tl = self.turn_landed
        self.turn_landed = torch.zeros_like(self.turn_landed)
        alpha = 1.0 - math.exp(-dt / max(1e-3, float(cfg["lat_tau"])))
        lag = alive & ~tl
        new_lane = (self.lane + (self.lat_target - self.lane) * alpha).clamp(-edge, edge)
        self.lane = torch.where(lag, new_lane, self.lane)

        self.stumble_kill_t = torch.where(alive, self.stumble_kill_t + dt, self.stumble_kill_t)
        self.stumble_t = torch.where(alive, (self.stumble_t - dt).clamp_min(0.0), self.stumble_t)

        # wall hit (reachable via turn-landing overshoot) — cliff level kills, else stumble
        wh = alive & (self.lane.abs() > edge)
        wh_cliff = wh & (self.t_level[:, 0] >= 3)
        self._kill(wh_cliff, 1)
        wh_st = wh & ~wh_cliff
        self._stumble(wh_st)
        self.lane = torch.where(wh_st, self.lane.clamp(-edge, edge), self.lane)

        # cliff edge-riding (sustained full tilt at altitude). NOTE: like the CPU sim, the
        # rest of _advance is NOT re-gated on is_dead — a runner killed by _apply_action this
        # step still advances (timers/lane/velocity/distance) exactly one frame. Only the
        # crossing loop and the post-loop obstacle checks stop for the dead.
        ce = alive & (self.t_level[:, 0] >= 3) \
             & (self.lane.abs() >= float(cfg.get("cliff_edge_lane", 6.3)))
        self.cliff_frames = torch.where(ce, self.cliff_frames + 1,
                                        torch.where(alive, torch.zeros_like(self.cliff_frames),
                                                    self.cliff_frames))
        self._kill(ce & (self.cliff_frames >= int(cfg.get("cliff_edge_kill_frames", 6))), 1)

        self.post_linger = torch.where(alive & (self.post_linger > 0), self.post_linger - 1,
                                       self.post_linger)

        # distance / velocity (unguarded on is_dead — CPU order)
        self.dist_real = torch.where(alive,
                                     self.dist_real + self.velocity * dt * float(cfg["dist_per_velsec"]),
                                     self.dist_real)
        curve_v = self._interp_vel(self.dist_real)
        sv = alive & self.stumble_vel_on
        nsv = (self.stumble_vel + float(cfg["stumble_recover_rate"]) * dt).minimum(curve_v)
        self.stumble_vel = torch.where(sv, nsv, self.stumble_vel)
        self.velocity = torch.where(sv, nsv, torch.where(alive, curve_v, self.velocity))
        self.stumble_vel_on = self.stumble_vel_on & ~(sv & (nsv >= curve_v))
        self.velocity = torch.where(alive, self.velocity * self.vel_scale, self.velocity)

        moved = self.velocity * dt
        self.distance = torch.where(alive, self.distance + moved, self.distance)
        self.score = torch.where(alive, self.score + moved * float(cfg["score_per_distance"]), self.score)
        self.dist_in_cell = torch.where(alive, self.dist_in_cell + moved, self.dist_in_cell)

        # cell crossings
        for _ in range(8):
            m = (~self.is_dead) & alive & (self.dist_in_cell >= self.cell_len)
            if not bool(m.any()):
                break
            self.dist_in_cell = torch.where(m, self.dist_in_cell - self.cell_len, self.dist_in_cell)
            self._cross(m)

        # jump-into-slide-bar (continuous while airborne on a must-slide cell)
        alive3 = alive & ~self.is_dead
        oc = self.t_obst[:, 0]
        jis = alive3 & (self.jump_t > 0.0) & self.must_slide_lut[oc]
        self._kill_t(jis, torch.where(oc == OBST_FLAME_TOWER, 5, 3))

        # wide-gap lip deadline (entered grounded; still grounded past the grace -> fell in)
        alive3 = alive & ~self.is_dead
        g = alive3 & self.gap_grace
        g_air = g & (self.jump_t > 0.0)
        self.gap_grace = self.gap_grace & ~g_air
        g_kill = g & ~g_air & ((self.dist_in_cell / self.cell_len.clamp_min(1e-6))
                               >= float(cfg["gap_entry_grace"]))
        self._kill_t(g_kill, torch.where(self.t_obst[:, 0] == OBST_HOLE, 1, 2))
        self.gap_grace = self.gap_grace & ~g_kill

    def _cross(self, m: torch.Tensor) -> None:
        # missed mandatory turn (checked on exit)
        missed = m & self.t_turn[:, 0] & (self.t_canl[:, 0] | self.t_canr[:, 0])
        self._kill(missed, 1)
        mm = m & ~missed
        if not bool(mm.any()):
            return
        self.gap_grace = self.gap_grace & ~mm
        self.cell_counter = torch.where(mm, self.cell_counter + 1.0, self.cell_counter)
        # shift the ring left; generate the new horizon cell
        mm2 = mm.unsqueeze(1)
        for t in (self.t_type, self.t_level, self.t_obst, self.t_turn, self.t_canl, self.t_canr,
                  self.t_coin, self.t_wide, self.t_stepup, self.t_nlane):
            shifted = torch.roll(t, -1, dims=1)
            t.copy_(torch.where(mm2, shifted, t))
        self._gen_cell(mm, TRACK_LEN - 1)

        # approach-buffer latch fires on entering the junction
        pt = mm & (self.pend_turn > 0)
        if bool(pt.any()):
            wl = self.pend_turn == 1
            nx_ok = torch.where(wl, self.t_canl[:, 0], self.t_canr[:, 0])
            fire = pt & self.t_turn[:, 0] & nx_ok
            if bool(fire.any()):
                self._execute_turn(fire, wl,
                                   torch.full_like(self.lane, float(self.cfg["turn_center_cprog"])),
                                   from_buffer=torch.ones_like(fire))
            ttl_dec = pt & ~fire
            self.pend_turn_ttl = torch.where(ttl_dec, self.pend_turn_ttl - 1, self.pend_turn_ttl)
            expired = ttl_dec & (self.pend_turn_ttl <= 0)
            self.pend_turn = torch.where(expired, torch.zeros_like(self.pend_turn), self.pend_turn)

        self._resolve_enter(mm & ~self.is_dead)

    def _resolve_enter(self, m: torch.Tensor) -> None:
        cfg = self.cfg
        # narrow band on every strip-cell entry
        nl = self.t_nlane[:, 0]
        nb = m & (nl != 0)
        die = nb & ((self.lane - nl.float() * float(cfg["narrow_lane_center"])).abs()
                    > float(cfg["narrow_lane_tol"]))
        self._kill(die, 1)
        m = m & ~self.is_dead

        obst = self.t_obst[:, 0]
        mj = self.must_jump_lut[obst]
        # apex check fires on the crevasse LEDGE cell: step-UP always; ALL crevasse last-cells when
        # crevasse_apex_all (step-DOWN/high-ground also demand near-apex timing). Mirror of CPU ref.
        apex_mask = self.t_stepup[:, 0]
        if cfg.get("crevasse_apex_all") or cfg.get("crevasse_graze_worldy"):
            last_crev = (obst == OBST_LEVEL_CHANGE) & self.t_wide[:, 0] & ~self.t_wide[:, 1]
            apex_mask = apex_mask | last_crev
        # step-up crevasse: velocity-dependent apex check at the raised ledge
        su = m & mj & apex_mask
        if bool(su.any()):
            su_nair = su & ((self.jump_t <= 0.0) | ~self.has_takeoff)
            self._kill(su_nair, 2)
            su_air = su & ~su_nair
            span = self.jump_dur * self.velocity / self.cell_len.clamp_min(1e-6)
            d = self.cell_counter - self.takeoff_pos
            frac = d / span.clamp_min(1e-6)
            height = 1.0 - (2.0 * frac - 1.0) ** 2
            if cfg.get("crevasse_graze_worldy"):
                # WORLD-Y graze (device-decoded): arc height = jump_peak*parabola; cap top 15 @lvl>=3,
                # 0 @lvl<3. Graze (stumble) iff arc < cap; deep clip -> Ledge kill. Mirror of CPU ref.
                arc_y = float(cfg["crevasse_jump_peak"]) * height
                cap_hi = float(cfg["crevasse_cap_height_hi"])
                cap_h = torch.where(self.t_level[:, 0] >= 3, torch.full_like(arc_y, cap_hi),
                                    torch.zeros_like(arc_y))
                band = float(cfg["crevasse_graze_band"])
                kill = su_air & (arc_y < cap_h - band)
                self._kill(kill, 2)
                graze = su_air & ~kill & (arc_y < cap_h)
                self._stumble(graze)     # chronic graze -> non-fatal x0.85 slowdown
            else:
                mh = float(cfg["stepup_min_height"])
                low = su_air & (height < mh)
                band = float(cfg.get("stepup_clip_band", 0.0))
                if band > 0.0:
                    clip = low & (height >= mh - band)
                    self._stumble(clip)      # marginal graze -> non-fatal slowdown (device 2026-07-06)
                    low = low & ~clip
                self._kill(low, 2)
            m = m & ~self.is_dead

        # plain must-jump (1-cell hole / wide gap with lip grace) — exclude apex (crevasse ledge) cells
        pj = m & mj & ~apex_mask
        need = torch.where(self.t_wide[:, 0], float(cfg["wide_gap_min_airtime"]), 0.0)
        fail = pj & (self.jump_t <= need)
        grace_ok = fail & self.t_wide[:, 0] & (float(cfg["gap_entry_grace"]) > 0.0)
        self.gap_grace = self.gap_grace | grace_ok
        die = fail & ~grace_ok
        self._kill_t(die, torch.where(obst == OBST_HOLE, 1, 2))
        m = m & ~self.is_dead
        # WIDE-GAP LANDING MODEL (mirror of the CPU ref): airborne here, but does the arc LAND past
        # the gap? Fires on the LAST gap cell (next cell not wide). A short arc lands in the far gap
        # cell -> clips the far wall (device dt4 Ledge). span = a*v + b (device fit). Default OFF.
        if cfg.get("wide_gap_landing_model"):
            # HOLES ONLY (obst==OBST_HOLE): the span fit is flat-hole physics; LevelChange crevasses
            # have raised-ledge apex physics -> applying the hole span rewarded early crevasse jumps.
            last_gap = (m & self.t_wide[:, 0] & (obst == OBST_HOLE) & ~self.t_stepup[:, 0]
                        & ~self.t_wide[:, 1] & (self.jump_t > 0.0) & self.has_takeoff)
            span = float(cfg["wide_gap_span_a"]) * self.velocity + float(cfg["wide_gap_span_b"])
            d = self.cell_counter - self.takeoff_pos
            short = last_gap & (span < d + 1.0 - float(cfg["wide_gap_land_grace"]))
            self._kill(short, 2)
            m = m & ~self.is_dead

        # must-slide: not sliding OR airborne -> death
        ms = m & self.must_slide_lut[obst] & ((self.slide_t <= 0.0) | (self.jump_t > 0.0))
        self._kill_t(ms, torch.where(obst == OBST_FLAME_TOWER, 5, 3))
        m = m & ~self.is_dead

        # coin
        cm = m & self.t_coin[:, 0]
        self.score = torch.where(cm, self.score + float(cfg["coin_score"]), self.score)
        self.t_coin[:, 0] = self.t_coin[:, 0] & ~cm

    # --------------------------------------------------------------- snapshot
    def _snapshot(self) -> dict:
        cfg = self.cfg
        cp = self._cp()
        win = (cp >= float(cfg["turn_exec_early"])) & (cp <= float(cfg["turn_exec_late"]))
        raw_turn = self.t_turn[:, 0]
        is_turn = raw_turn & win
        can_l = self.t_canl[:, 0] & win
        can_r = self.t_canr[:, 0] & win
        lingering = self.post_linger > 0
        is_turn = torch.where(lingering, torch.ones_like(is_turn), is_turn)
        can_l = torch.where(lingering, self.linger_l, can_l)
        can_r = torch.where(lingering, self.linger_r, can_r)

        obst = self.t_obst[:, 0]
        nobst = self.t_obst[:, 1]
        on_ground = (self.jump_t <= 0.0) & ~self.is_dead
        neg1 = torch.full_like(self.t_level[:, 1], -1)
        return {
            "is_dead": self.is_dead,
            "is_jumping": self.jump_t > 0.0,
            "is_sliding": self.slide_t > 0.0,
            "is_on_ground": on_ground,
            "is_stumbling": self.stumble_t > 0.0,
            "is_strafing": (self.lat_target - self.lane).abs() > 0.5,
            "score": torch.floor(self.score),
            "velocity": self.velocity.clone(),
            "path_type": self.t_type[:, 0].clone(),
            "path_level": self.t_level[:, 0].clone(),
            "facing": self.facing.clone(),
            "must_jump": self.must_jump_lut[obst],
            "must_slide": self.must_slide_lut[obst],
            "is_game_over": self.is_dead,
            "can_turn_left": can_l,
            "can_turn_right": can_r,
            "is_turn": is_turn,
            "distance": self.dist_real.clone(),
            "next_is_turn": self.t_turn[:, 1].clone(),
            "next_can_turn_left": self.t_canl[:, 1].clone(),
            "next_can_turn_right": self.t_canr[:, 1].clone(),
            "next_must_jump": self.must_jump_lut[nobst],
            "next_must_slide": self.must_slide_lut[nobst],
            "next_wide_gap": self.t_wide[:, 1].clone(),
            "next_ground_change_higher": nobst == OBST_LEVEL_CHANGE,
            "next_path_level": torch.where(raw_turn, neg1, self.t_level[:, 1]),
            "next2_path_level": torch.where(raw_turn | self.t_turn[:, 1], neg1, self.t_level[:, 2]),
            # 2-cell OBSTACLE lookahead (chained-obstacle fix) — same junction dropout as next2_path_level.
            "next2_must_jump": self.must_jump_lut[self.t_obst[:, 2]] & ~(raw_turn | self.t_turn[:, 1]),
            "next2_must_slide": self.must_slide_lut[self.t_obst[:, 2]] & ~(raw_turn | self.t_turn[:, 1]),
            "narrow_target_lane": self.t_nlane[:, 1].clone(),    # 0 == None
            "cell_progress": cp,
            "lane": self.lane.clone(),
            "stumble_kill_timer": self.stumble_kill_t.clone(),
            "stumble_count": self.stumble_count.clone(),
        }

    def _base_obs(self, s: dict) -> torch.Tensor:
        """Vectorized obs.state_to_obs — [B, OBS_LEN], identical layout/normalization."""
        B = self.B
        out = torch.zeros(B, O.OBS_LEN, dtype=torch.float32, device=self.dev)
        for i, f in enumerate(O.OBS_BOOL_FIELDS):
            out[:, i] = s[f].float()
        b = len(O.OBS_BOOL_FIELDS)
        v = s["velocity"]
        out[:, b + 0] = ((v - O.VEL_MIN) / (O.VEL_MAX - O.VEL_MIN)).clamp(0.0, 1.0)
        pt = s["path_type"]
        out[:, b + 1] = torch.where(pt < 0, 0.0,
                        torch.where(pt <= 2, 0.16,
                        torch.where(pt <= 7, 0.33,
                        torch.where(pt <= 16, 0.50,
                        torch.where(pt <= 30, 0.66, 0.83)))))
        pl = s["path_level"]
        out[:, b + 2] = torch.where(pl < 0, 0.0, pl.float().clamp(0.0, O.PATH_LEVEL_MAX) / O.PATH_LEVEL_MAX)
        fc = s["facing"]
        out[:, b + 3] = torch.where(fc < 0, 0.0, fc.float().clamp(0.0, O.FACING_MAX) / O.FACING_MAX)
        cp = s["cell_progress"]
        out[:, b + 4] = cp.clamp(0.0, 1.0)
        lane_norm = lambda x: ((x - O.LANE_OBS_MIN) / (O.LANE_OBS_MAX - O.LANE_OBS_MIN)).clamp(0.0, 1.0)  # noqa: E731
        out[:, b + 5] = lane_norm(s["lane"])
        ntl = s["narrow_target_lane"]
        out[:, b + 6] = lane_norm(ntl.float() * O.NARROW_CENTER)   # ntl==0 (None) -> center 0.5, like CPU
        out[:, b + 7] = (cp + O.DCPROG_PER_VEL * v).clamp(0.0, 1.0)
        span = O.JUMP_DURATION * v / O.CELL_LENGTH
        raw = (O.LEDGE_CELLS_AHEAD - cp) - 0.5 * span
        pj = ((raw + 0.5) / 2.0).clamp(0.0, 1.0)
        out[:, b + 8] = torch.where(s["next_ground_change_higher"], pj,
                                    torch.full_like(pj, O.PJ_NONE))
        npl = s["next_path_level"]
        out[:, b + 9] = torch.where(npl < 0, 0.0, npl.float().clamp(0.0, O.PATH_LEVEL_MAX) / O.PATH_LEVEL_MAX)
        n2 = s["next2_path_level"]
        out[:, b + 10] = torch.where(n2 < 0, 0.0, n2.float().clamp(0.0, O.PATH_LEVEL_MAX) / O.PATH_LEVEL_MAX)
        return out

    def _full_obs(self) -> torch.Tensor:
        parts = [self.frames.reshape(self.B, -1)]
        if self.K > 0:
            oh = torch.nn.functional.one_hot(self.act_hist, C.N_ACTIONS).float()
            parts.append(oh.reshape(self.B, -1))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    # ---------------------------------------------------------------- shaping
    def _shaping(self, a: torch.Tensor, snap: dict) -> torch.Tensor:
        if not any((self.action_cost, self.reaction_bonus, self.turn_spam_penalty,
                    self.idle_action_penalty, self.redundant_action_penalty, self.strafe_penalty,
                    self.narrow_hold_bonus, self.velocity_drop_penalty, self.off_center_penalty,
                    self.slide_jump_penalty)):
            return torch.zeros(self.B, dtype=torch.float64, device=self.dev)
        p = self.prev
        r = torch.zeros(self.B, dtype=torch.float64, device=self.dev)
        is_j, is_s = a == A_JUMP, a == A_SLIDE
        is_strafe = (a == A_SL) | (a == A_SR)
        is_t = (a == A_TL) | (a == A_TR)

        if self.slide_jump_penalty:
            r = r - self.slide_jump_penalty * (is_j & (p["must_slide"] | p["next_must_slide"]
                                                       | p["is_sliding"])).float()
        if self.velocity_drop_penalty:
            drop = p["velocity"] - snap["velocity"]
            r = r - torch.where(drop > 3.0, self.velocity_drop_penalty * drop, torch.zeros_like(drop))

        ptv = p["path_type"]
        ntl = p["narrow_target_lane"]
        on_narrow = ((ptv >= 17) & (ptv <= 30)) | (ntl != 0)
        if self.action_cost:
            pay = (a != A_NOOP) & ~(is_strafe & on_narrow)
            r = r - self.action_cost * pay.float()
        if self.narrow_hold_bonus:
            has_t = on_narrow & (ntl != 0)
            sign = torch.where(ntl > 0, 1.0, -1.0)
            closeness = (p["lane"] * sign / 3.0).clamp(0.0, 1.0)
            r = r + self.narrow_hold_bonus * torch.where(
                has_t, closeness, (on_narrow & (ntl == 0)).float())
        if self.off_center_penalty:
            off = (p["lane"].abs() - 1.5).clamp_min(0.0)
            r = r - self.off_center_penalty * torch.where(on_narrow, torch.zeros_like(off), off)
        if self.reaction_bonus:
            jm = (p["must_jump"] | p["next_must_jump"]) & is_j
            sm = ~jm & (p["must_slide"] | p["next_must_slide"]) & is_s
            tm = ~jm & ~sm & p["is_turn"]
            # CPU quirk replicated: `prev.get("cell_progress") or 0.44` maps cp==0 -> 0.44
            cp0 = torch.where(p["cell_progress"] == 0.0, 0.44, p["cell_progress"])
            cp_land = cp0 + O.DCPROG_PER_VEL * p["velocity"]
            aim = (1.0 - self.aim_center_weight * (cp_land - 0.44).abs() / 0.16).clamp_min(0.0)
            side_ok = ((a == A_TL) & p["can_turn_left"]) | ((a == A_TR) & p["can_turn_right"])
            r = r + self.reaction_bonus * (jm | sm).float()
            r = r + self.reaction_bonus * aim * (tm & side_ok).float()
        if self.turn_spam_penalty:
            last_t = (self.last_action == A_TL) | (self.last_action == A_TR)
            r = r - self.turn_spam_penalty * (is_t & last_t).float()
        if self.idle_action_penalty:
            ij = is_j & ~p["must_jump"] & ~p["next_must_jump"]
            isl = ~ij & is_s & ~p["must_slide"] & ~p["next_must_slide"]
            it = ~ij & ~isl & is_t & ~p["is_turn"] & ~p["next_is_turn"]
            r = r - self.idle_action_penalty * (ij | isl | it).float()
        if self.redundant_action_penalty:
            rj = is_j & p["must_jump"] & ~p["next_must_jump"]
            rs = ~rj & is_s & p["must_slide"] & ~p["next_must_slide"]
            r = r - self.redundant_action_penalty * (rj | rs).float()
        dock = self.strafe_penalty if self.strafe_penalty else self.idle_action_penalty
        if dock:
            r = r - dock * (is_strafe & (ntl == 0)).float()
        return r

    # ------------------------------------------------- tensor-native fast path
    def reset_tensor(self) -> torch.Tensor:
        """reset() but returning the obs as a device tensor (no numpy)."""
        self._reset_mask(torch.ones(self.B, dtype=torch.bool, device=self.dev))
        return self._full_obs()

    def step_tensor(self, a: torch.Tensor):
        """GPU-resident step for the fast PPO loop: no numpy, no per-env info dicts.

        Returns (obs, reward, done, timeout, terminal_obs, ep_ret, ep_len, ep_dist):
          obs          [B, obs_dim] float32 — post-auto-reset (rollout continues seamlessly)
          reward/done/timeout [B] — timeout = truncated-not-terminated (bootstrap marker)
          terminal_obs [B, obs_dim] — pre-reset obs, valid only where done (value bootstrap)
          ep_*         [B] — completed-episode stats, valid only where done
        """
        a = a.reshape(self.B).to(self.dev, dtype=torch.int64)
        alive = ~self.is_dead
        self._apply_action(a, alive)
        self._advance(alive)

        snap = self._snapshot()
        if self.reward_mode == "distance":
            reward = (snap["distance"] - self.prev_dist).clamp_min(0.0) * self.reward_scale
        else:
            reward = (snap["score"] - self.prev_score) * self.reward_scale
        reward = reward + self._shaping(a, snap)
        terminated = snap["is_dead"]
        reward = reward - self.death_penalty * terminated.float()

        self.prev_dist = snap["distance"].clone()
        self.prev_score = snap["score"].clone()
        self.prev = snap
        self.last_action = a.clone()
        self.elapsed = self.elapsed + 1
        truncated = (self.elapsed >= self.max_episode_steps) & ~terminated
        done = terminated | truncated

        if self.K > 0:
            self.act_hist = torch.roll(self.act_hist, -1, dims=1)
            self.act_hist[:, -1] = a
        self.frames = torch.roll(self.frames, -1, dims=1)
        self.frames[:, -1] = self._base_obs(snap)

        self.ep_ret = self.ep_ret + reward
        self.ep_len = self.ep_len + 1

        term_obs = self._full_obs()
        ep_ret = self.ep_ret.clone()
        ep_len = self.ep_len.clone()
        ep_dist = snap["distance"].clone()
        if bool(done.any()):
            self.term_snap = {k: v.clone() for k, v in snap.items()}
            self._reset_mask(done)
            obs = self._full_obs()
        else:
            obs = term_obs
        return obs, reward, done, truncated, term_obs, ep_ret, ep_len, ep_dist

    # ----------------------------------------------------------- VecEnv API
    def reset(self):
        self._reset_mask(torch.ones(self.B, dtype=torch.bool, device=self.dev))
        return self._full_obs().cpu().numpy()

    def step_async(self, actions):
        self._actions = torch.as_tensor(np.asarray(actions), dtype=torch.int64,
                                        device=self.dev).reshape(self.B)

    def step_wait(self):
        a = self._actions
        alive = ~self.is_dead
        self._apply_action(a, alive)
        self._advance(alive)

        snap = self._snapshot()
        if self.reward_mode == "distance":
            reward = (snap["distance"] - self.prev_dist).clamp_min(0.0) * self.reward_scale
        else:
            reward = (snap["score"] - self.prev_score) * self.reward_scale
        reward = reward + self._shaping(a, snap)
        terminated = snap["is_dead"]
        reward = reward - self.death_penalty * terminated.float()

        self.prev_dist = snap["distance"].clone()
        self.prev_score = snap["score"].clone()
        self.prev = snap
        self.last_action = a.clone()
        self.elapsed = self.elapsed + 1
        truncated = (self.elapsed >= self.max_episode_steps) & ~terminated
        done = terminated | truncated

        # action history + frame stack
        if self.K > 0:
            self.act_hist = torch.roll(self.act_hist, -1, dims=1)
            self.act_hist[:, -1] = a
        base = self._base_obs(snap)
        self.frames = torch.roll(self.frames, -1, dims=1)
        self.frames[:, -1] = base

        self.ep_ret = self.ep_ret + reward
        self.ep_len = self.ep_len + 1

        obs_t = self._full_obs()
        rewards = reward.cpu().numpy()
        dones = done.cpu().numpy()
        infos = [{} for _ in range(self.B)]
        if bool(done.any()):
            # stash the terminal snapshot BEFORE auto-reset (the gym adapter reads it)
            self.term_snap = {k: v.clone() for k, v in snap.items()}
            idxs = torch.nonzero(done, as_tuple=False).squeeze(1)
            term_obs = obs_t[idxs].cpu().numpy()
            ep_r = self.ep_ret[idxs].cpu().numpy()
            ep_l = self.ep_len[idxs].cpu().numpy()
            tr = truncated[idxs].cpu().numpy()
            dt = self.death_type[idxs].cpu().numpy()
            di = snap["distance"][idxs].cpu().numpy()
            for j, i in enumerate(idxs.cpu().numpy()):
                infos[i]["terminal_observation"] = term_obs[j]
                infos[i]["TimeLimit.truncated"] = bool(tr[j])
                infos[i]["episode"] = {"r": float(ep_r[j]), "l": int(ep_l[j])}
                infos[i]["death_type"] = int(dt[j])
                infos[i]["distance"] = float(di[j])
            self._reset_mask(done)
            obs_t = self._full_obs()
        return obs_t.cpu().numpy(), rewards, dones, infos

    def close(self):
        return None

    def get_attr(self, attr_name, indices=None):
        n = self.B if indices is None else len(indices)
        return [getattr(self, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError(f"env_method({method_name}) not supported on the batched sim")

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.B if indices is None else len(indices)
        return [False] * n

    def seed(self, seed=None):
        if seed is not None:
            self.g.manual_seed(int(seed))
        return [seed] * self.B


class TorchSimGymAdapter(gym.Env):
    """Single-env gymnasium adapter over a batch-1 CPU ``TorchTempleRunVecEnv``.

    API-compatible with ``TempleRunSimEnv`` for procedural use (reset/step/raw_state,
    ``info['raw_state']`` dicts) so the scorecard/oracle tooling can drive the torch sim
    through the ``TempleRunSim-v0`` registry alias. Replay/set_state are NOT supported.
    """

    metadata = {"render_modes": []}

    def __init__(self, track_source: Any = "procedural", config: Optional[dict] = None,
                 seed: Optional[int] = None, device: str = "cpu", **kwargs):
        injected = None
        if not isinstance(track_source, str):
            injected = list(track_source)
        kw = dict(kwargs)
        kw.pop("action_delay", None)   # unsupported (always 0 in procedural tooling)
        self.venv = TorchTempleRunVecEnv(1, config=config, device=device,
                                         seed=0 if seed is None else int(seed),
                                         injected_track=injected, **kw)
        self.observation_space = self.venv.observation_space
        self.action_space = self.venv.action_space
        self.cfg = self.venv.cfg

    def _state_dict(self, snap: Optional[dict] = None) -> dict:
        s = self.venv._snapshot() if snap is None else snap
        out = {}
        for k, v in s.items():
            x = v[0].item()
            if v.dtype == torch.bool:
                out[k] = bool(x)
            elif k in ("score", "path_type", "path_level", "facing", "stumble_count",
                       "next_path_level", "next2_path_level"):
                out[k] = int(x)
            else:
                out[k] = float(x)
        ntl = int(s["narrow_target_lane"][0].item())
        out["narrow_target_lane"] = None if ntl == 0 else ntl
        return out

    def raw_state(self) -> dict:
        return self._state_dict()

    # --- CPU-sim internals compatibility (fidelity_scorecard.check_stumble pokes these) ---
    @property
    def _velocity(self) -> float:
        return float(self.venv.velocity[0].item())

    @property
    def _is_dead(self) -> bool:
        return bool(self.venv.is_dead[0].item())

    def _stumble(self) -> None:
        self.venv._stumble(torch.ones(1, dtype=torch.bool, device=self.venv.dev))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.venv.seed(seed)
        obs = self.venv.reset()
        st = self._state_dict()
        return obs[0], {"raw_state": st, "score": st["score"], "distance": st["distance"]}

    def step(self, action: int):
        obs, rew, done, infos = self.venv.step(np.array([int(action)]))
        info = infos[0]
        # NOTE: the vec env auto-resets; expose the TERMINAL state like a plain gym env would.
        if done[0]:
            term_obs = info.get("terminal_observation")
            trunc = bool(info.get("TimeLimit.truncated", False))
            raw = self._state_dict(self.venv.term_snap)
            out_info = {"raw_state": raw, "death_type": info.get("death_type"),
                        "distance": raw.get("distance", 0.0)}
            return (term_obs if term_obs is not None else obs[0]), float(rew[0]), \
                not trunc, trunc, out_info
        st = self._state_dict()
        return obs[0], float(rew[0]), False, False, \
            {"raw_state": st, "score": st["score"], "distance": st["distance"],
             "death_type": None}

    def close(self):
        self.venv.close()


try:
    from gymnasium.envs.registration import register as _register
    from gymnasium.envs.registration import registry as _registry

    if "TempleRunSimTorch-v0" not in _registry:
        _register(id="TempleRunSimTorch-v0",
                  entry_point="templerun_rl.sim_env_torch:TorchSimGymAdapter",
                  max_episode_steps=None, disable_env_checker=True)
except Exception:  # noqa: BLE001
    pass
