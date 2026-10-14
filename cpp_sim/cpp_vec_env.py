"""ctypes front-end for the C++ port of the Temple Run sim (``cpp_sim/sim.cpp``).

Two entry points:

``CppTempleRunVecEnv``
    Drop-in replacement for ``templerun_rl.sim_env_torch.TorchTempleRunVecEnv``
    as ``scripts/train/train_fast.py`` uses it: ``reset_tensor()``, ``step_tensor()``,
    ``approach_turn_hint()``, ``observation_space``/``action_space``/``num_envs``,
    plus the SB3 ``VecEnv`` methods.  The simulation runs on the CPU; the tensors
    handed back live on whatever ``device`` you ask for.

``CppSimGymAdapter``
    Single-env ``gymnasium.Env`` with the same contract as
    ``templerun_rl.sim_env.TempleRunSimEnv`` (``info["raw_state"]`` dicts,
    ``terminated``/``truncated``), which is what ``scripts/validate/sim_equivalence.py``
    drives for the deterministic tier.

NOTE ON GLOBAL CONFIG: the shared library keeps ONE global config block.  Each
wrapper remembers its own config and re-installs it whenever it becomes the
active env, so several differently-configured envs can coexist in one process
(they just may not be stepped concurrently from multiple threads).
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Any, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gymnasium as gym  # noqa: E402

from templerun_rl import constants as C  # noqa: E402
from templerun_rl import obs as O  # noqa: E402
from templerun_rl.sim_env import DEFAULT_SIM_CONFIG, parse_type_reject  # noqa: E402

try:  # SB3 is only needed for the VecEnv base class the trainer expects
    from stable_baselines3.common.vec_env import VecEnv as _VecEnvBase
except Exception:  # noqa: BLE001
    _VecEnvBase = object


# ---------------------------------------------------------------- library ---
def _lib_path() -> str:
    for name in ("sim.dll", "sim.so", "libsim.so", "sim.dylib"):
        p = os.path.join(_HERE, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "cpp_sim shared library not found -- run:  sh cpp_sim/build.sh")


_LIB = None


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    L = ctypes.CDLL(_lib_path())
    d, i, u64, cp, vp = (ctypes.c_double, ctypes.c_int, ctypes.c_ulonglong,
                         ctypes.c_char_p, ctypes.c_void_p)
    pd, pf, pi, pu8 = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ubyte))
    L.sim_set_pj_lead.argtypes = [ctypes.c_double]
    for n in ("sim_state_size", "sim_obs_len", "sim_obs_base", "sim_raw_len"):
        getattr(L, n).argtypes = []
        getattr(L, n).restype = i
    L.sim_cfg_reset.argtypes = []
    L.sim_cfg_set.argtypes = [cp, d]
    L.sim_cfg_set.restype = i
    L.sim_cfg_get.argtypes = [cp]
    L.sim_cfg_get.restype = d
    L.sim_cfg_set_vel_curve.argtypes = [pd, pd, i]
    L.sim_cfg_set_ramp_table.argtypes = [pd, i]
    L.sim_cfg_set_obst_weights.argtypes = [pi, pd, i]
    L.sim_cfg_set_narrow_len_weights.argtypes = [pi, pd, i]
    L.sim_cfg_set_type_reject.argtypes = [pi, pi, pd, i]
    L.sim_cfg_clear_profiles.argtypes = []
    L.sim_profile_key_id.argtypes = [cp]
    L.sim_profile_key_id.restype = i
    L.sim_cfg_add_profile.argtypes = [d, i, pi, pd]
    L.sim_cfg_add_profile.restype = i
    L.sim_set_env_params.argtypes = [d, i, d, i, d, i, i, i]
    L.sim_set_shaping.argtypes = [pd, i]
    L.sim_set_track.argtypes = [pi, i]
    L.sim_init.argtypes = [vp, i, u64]
    L.sim_reset_all.argtypes = [vp, i, ctypes.c_longlong, pf]
    L.sim_reset_one.argtypes = [vp, i, ctypes.c_longlong, pf]
    L.sim_step.argtypes = [vp, i, pi, i, pf, pd, pu8, pf, pd, pi, pd, pi]
    L.sim_get_raw.argtypes = [vp, i, pd]
    L.sim_get_obs.argtypes = [vp, i, pf]
    L.sim_approach_turn_hint.argtypes = [vp, i, pi]
    L.sim_cascade_hint.argtypes = [vp, i, pi]
    for n, rt in (("sim_get_distance", d), ("sim_get_velocity", d),
                  ("sim_get_death_type", i), ("sim_get_is_dead", i),
                  ("sim_get_draws", u64), ("sim_get_profile_idx", i)):
        getattr(L, n).argtypes = [vp, i]
        getattr(L, n).restype = rt
    L.sim_stumble.argtypes = [vp, i]
    _LIB = L
    return L


# --------------------------------------------------------------- raw state --
# MUST match the R_* enum in sim.cpp.
RAW_FIELDS = (
    "is_dead", "is_jumping", "is_sliding", "is_on_ground", "is_stumbling",
    "is_strafing", "must_jump", "must_slide", "is_game_over",
    "can_turn_left", "can_turn_right", "is_turn",
    "next_is_turn", "next_can_turn_left", "next_can_turn_right",
    "next_must_jump", "next_must_slide", "next_wide_gap",
    "next_ground_change_higher", "next2_must_jump", "next2_must_slide",
    "next3_must_jump", "next3_must_slide",
    "score", "velocity", "path_type", "path_level", "facing", "distance",
    "next_path_level", "next2_path_level",
    "narrow_target_lane", "next2_narrow_lane",
    "cell_progress", "lane", "stumble_kill_timer", "stumble_count",
    # fields past the obs layout -- exposed so device forensics run on C++ rollouts too
    "soft_stepdown", "next_soft_stepdown", "dist_next_must_jump", "dist_next_must_slide",
    "obstacle",
)
_RAW_BOOL = set(RAW_FIELDS[:23])
_RAW_INT = {"score", "path_type", "path_level", "facing", "obstacle",
            "next_path_level", "next2_path_level", "stumble_count"}
_RAW_LANE = {"narrow_target_lane", "next2_narrow_lane"}
_ALWAYS_FALSE = ("has_invincibility", "has_angel_wings", "has_vacuum",
                 "has_boost", "is_mega_boost")

SHAPING_ORDER = ("action_cost", "reaction_bonus", "turn_spam_penalty",
                 "idle_action_penalty", "redundant_action_penalty",
                 "airborne_action_penalty", "jump_hold_penalty", "strafe_penalty",
                 "narrow_hold_bonus", "aim_center_weight", "velocity_drop_penalty",
                 "off_center_penalty", "slide_jump_penalty", "turn_buffer_bonus",
                 "stepdown_bonus_gate", "landing_hazard_penalty",
                 "slide_landing_hazard_penalty",
                 "stepdown_jump_penalty",
                 "margin_bonus",
                 "stack_margin_penalty",
                 "turn_takeoff_penalty")

# config keys handled by dedicated setters (not sim_cfg_set)
_TABLE_KEYS = {"velocity_curve", "difficulty_ramp_table", "obstacle_type_weights",
               "narrow_len_weights", "episode_profiles", "type_reject"}
# list-valued domain-randomisation ranges -> <key>_lo / <key>_hi scalars
_DR_RANGE_KEYS = ("dr_turn_window_frac", "dr_turn_early_frac", "dr_turn_late_frac",
                  "dr_jump_duration", "dr_slide_duration", "dr_velocity_scale",
                  "dr_obstacle_prob", "dr_turn_prob", "dr_cell_length")


def _carr(t, seq):
    seq = list(seq)
    return (t * max(1, len(seq)))(*seq) if seq else (t * 1)()


class _Config:
    """Everything the shared library keeps globally, captured per-env instance."""

    def __init__(self, config: Optional[dict], track_source, step_dt, max_episode_steps,
                 reward_scale, reward_mode, death_penalty, action_history_len,
                 frame_stack, shaping: dict):
        cfg = dict(DEFAULT_SIM_CONFIG)
        if config:
            for k, v in config.items():
                if k == "obstacle_type_weights" and isinstance(v, dict):
                    w = dict(DEFAULT_SIM_CONFIG["obstacle_type_weights"])
                    w.update({int(kk): vv for kk, vv in v.items()})
                    cfg[k] = w
                else:
                    cfg[k] = v
        self.cfg = cfg
        self.track = None
        if not isinstance(track_source, str):
            self.track = [dict(e) for e in track_source]
        self.step_dt = float(step_dt)
        self.max_episode_steps = int(max_episode_steps)
        self.reward_scale = float(reward_scale)
        self.reward_score = 1 if str(reward_mode) != "distance" else 0
        self.death_penalty = float(death_penalty)
        self.K = int(action_history_len)
        self.FS = max(1, int(frame_stack))
        self.shaping = {k: float(shaping.get(k, 0.0) or 0.0) for k in SHAPING_ORDER}
        self._validate()

    def _validate(self):
        """Reject unsupported episode_profiles overrides here (mirrors the torch env,
        which raises ValueError), and surface unknown scalar knobs loudly."""
        L = lib()
        for prof in (self.cfg.get("episode_profiles") or []):
            for k in (prof.get("overrides") or {}):
                if L.sim_profile_key_id(k.encode()) < 0:
                    raise ValueError(f"episode_profiles override '{k}' not supported")

    def install(self):
        L = lib()
        L.sim_cfg_reset()
        cfg = self.cfg
        unknown = []
        for k, v in cfg.items():
            if k in _TABLE_KEYS:
                continue
            if k in _DR_RANGE_KEYS:
                lo, hi = v
                L.sim_cfg_set(f"{k}_lo".encode(), float(lo))
                L.sim_cfg_set(f"{k}_hi".encode(), float(hi))
                continue
            if k == "pair_same_type_types":
                # the C++ sampler HARDCODES the slide types (5, 6) -- that restriction is the model
                # (device: 73% of same-type pairs are repeated slides, holes repeat BELOW chance).
                # Accept the key only when it matches, so a different list fails loudly instead of
                # silently diverging from the Python reference.
                if tuple(int(x) for x in v) != (5, 6):
                    raise ValueError(
                        f"cpp_sim: pair_same_type_types={tuple(v)} unsupported; the C++ sampler is "
                        f"hardcoded to (5, 6). Change sim.cpp too if this is intentional.")
                continue
            if k == "pair_anti_repeat_types":
                # same deal: the C++ anti-repeat branch is hardcoded to (1, 4) = Hole / TreeJump.
                if tuple(int(x) for x in v) != (1, 4):
                    raise ValueError(
                        f"cpp_sim: pair_anti_repeat_types={tuple(v)} unsupported; the C++ sampler "
                        f"is hardcoded to (1, 4). Change sim.cpp too if this is intentional.")
                continue
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            if not isinstance(v, (int, float)):
                unknown.append(k)
                continue
            if not L.sim_cfg_set(k.encode(), float(v)):
                unknown.append(k)
        if unknown:
            raise KeyError(f"cpp_sim: unsupported sim-config key(s): {sorted(unknown)}")

        vc = cfg.get("velocity_curve") or []
        L.sim_cfg_set_vel_curve(_carr(ctypes.c_double, [p[0] for p in vc]),
                                _carr(ctypes.c_double, [p[1] for p in vc]), len(vc))
        rows = []
        for r in (cfg.get("difficulty_ramp_table") or ()):
            rows += [float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                     1.0 if r[4] else 0.0, 1.0 if r[5] else 0.0]
        L.sim_cfg_set_ramp_table(_carr(ctypes.c_double, rows), len(rows) // 6)
        ow = cfg["obstacle_type_weights"]
        L.sim_cfg_set_obst_weights(_carr(ctypes.c_int, list(ow.keys())),
                                   _carr(ctypes.c_double, list(ow.values())), len(ow))
        nl = cfg["narrow_len_weights"]
        L.sim_cfg_set_narrow_len_weights(_carr(ctypes.c_int, list(nl.keys())),
                                         _carr(ctypes.c_double, list(nl.values())), len(nl))
        # type_reject: parsed by the CPU REFERENCE's own parser so the two ports cannot disagree
        # about what a key means (and so its validation -- tree-valued `cur`, out-of-range types --
        # fires identically on both backends). Always installed, so an env without the key clears
        # any matrix a previous env left in the process-global config.
        tr = sorted(parse_type_reject(cfg.get("type_reject")).items())
        L.sim_cfg_set_type_reject(_carr(ctypes.c_int, [k[0] for k, _ in tr]),
                                  _carr(ctypes.c_int, [k[1] for k, _ in tr]),
                                  _carr(ctypes.c_double, [v for _, v in tr]), len(tr))
        L.sim_cfg_clear_profiles()
        for prof in (cfg.get("episode_profiles") or []):
            ov = prof.get("overrides") or {}
            ids, vals = [], []
            for k, v in ov.items():
                kid = L.sim_profile_key_id(k.encode())
                if kid < 0:
                    raise ValueError(f"episode_profiles override '{k}' not supported")
                ids.append(kid)
                vals.append(float(v))
            L.sim_cfg_add_profile(float(prof.get("weight", 1.0)), len(ids),
                                  _carr(ctypes.c_int, ids), _carr(ctypes.c_double, vals))

        L.sim_set_pj_lead(ctypes.c_double(float(getattr(O, 'PJ_LEAD_FRAC', 0.5))))
        L.sim_set_env_params(self.step_dt, self.max_episode_steps, self.reward_scale,
                             self.reward_score, self.death_penalty, self.K, self.FS,
                             1 if O.PJ_VELOCITY_INVARIANT else 0)
        L.sim_set_shaping(_carr(ctypes.c_double, [self.shaping[k] for k in SHAPING_ORDER]),
                          len(SHAPING_ORDER))
        if self.track:
            flat = []
            for e in self.track:
                nl_ = e.get("narrow_lane")
                flat += [int(e.get("type", e.get("path_type", 1))),
                         int(e.get("level", e.get("path_level", 2))),
                         int(e.get("obstacles", 0)),
                         1 if e.get("is_turn") else 0,
                         1 if e.get("can_turn_left") else 0,
                         1 if e.get("can_turn_right") else 0,
                         1 if e.get("is_coin") else 0,
                         1 if e.get("wide_gap") else 0,
                         1 if e.get("stepup") else 0,
                         0 if nl_ is None else int(nl_),
                         1 if e.get("stepdown_low") else 0]
            L.sim_set_track(_carr(ctypes.c_int, flat), len(self.track))
        else:
            L.sim_set_track(_carr(ctypes.c_int, [0]), 0)


_ACTIVE: Any = None


class _Base:
    """Shared plumbing: owns the env-state buffer and keeps the global config in sync."""

    def __init__(self, num_envs: int, conf: _Config, seed: int):
        self._L = lib()
        self.num_envs = int(num_envs)
        self._conf = conf
        self._activate()
        self._ssz = self._L.sim_state_size()
        self.obs_dim = self._L.sim_obs_len()
        self._buf = ctypes.create_string_buffer(self._ssz * self.num_envs)
        self._raw = (ctypes.c_double * self._L.sim_raw_len())()
        B, D = self.num_envs, self.obs_dim
        self._obs = np.zeros((B, D), dtype=np.float32)
        self._tobs = np.zeros((B, D), dtype=np.float32)
        # reward / episode stats stay float64 end-to-end (the C++ computes them in
        # double, and TorchTempleRunVecEnv hands back float64 too)
        self._rew = np.zeros(B, dtype=np.float64)
        self._flg = np.zeros(B, dtype=np.uint8)
        self._epr = np.zeros(B, dtype=np.float64)
        self._epl = np.zeros(B, dtype=np.int32)
        self._epd = np.zeros(B, dtype=np.float64)
        self._dty = np.zeros(B, dtype=np.int32)
        self._act = np.zeros(B, dtype=np.int32)
        self._hint = np.zeros(B, dtype=np.int32)
        self._L.sim_init(self._buf, self.num_envs, ctypes.c_ulonglong(int(seed) & 0xFFFFFFFF))

    # -- global-config bookkeeping ------------------------------------------
    def _activate(self):
        global _ACTIVE
        if _ACTIVE is not self:
            self._conf.install()
            _ACTIVE = self

    def _p(self, arr, t):
        return arr.ctypes.data_as(ctypes.POINTER(t))

    # -- raw helpers ---------------------------------------------------------
    def raw_dict(self, env: int = 0) -> dict:
        self._L.sim_get_raw(self._buf, int(env), self._raw)
        out = {}
        for i, f in enumerate(RAW_FIELDS):
            v = self._raw[i]
            if f in _RAW_BOOL:
                out[f] = bool(v)
            elif f in _RAW_LANE:
                out[f] = None if int(v) == 0 else int(v)
            elif f in _RAW_INT:
                out[f] = int(v)
            else:
                out[f] = float(v)
        for f in _ALWAYS_FALSE:
            out[f] = False
        return out

    def draws(self, env: int = 0) -> int:
        return int(self._L.sim_get_draws(self._buf, int(env)))

    def profile_idx(self, env: int = 0) -> int:
        return int(self._L.sim_get_profile_idx(self._buf, int(env)))

    def _step_raw(self, actions: np.ndarray, auto_reset: int):
        self._activate()
        np.copyto(self._act, actions.astype(np.int32, copy=False))
        self._L.sim_step(self._buf, self.num_envs, self._p(self._act, ctypes.c_int),
                         auto_reset,
                         self._p(self._obs, ctypes.c_float),
                         self._p(self._rew, ctypes.c_double),
                         self._p(self._flg, ctypes.c_ubyte),
                         self._p(self._tobs, ctypes.c_float),
                         self._p(self._epr, ctypes.c_double),
                         self._p(self._epl, ctypes.c_int),
                         self._p(self._epd, ctypes.c_double),
                         self._p(self._dty, ctypes.c_int))


# ======================================================================= vec ==
class CppTempleRunVecEnv(_Base, _VecEnvBase):
    """Batched CPU sim behind the interface ``scripts/train/train_fast.py`` uses."""

    def __init__(self, num_envs: int, config: Optional[dict] = None,
                 device: str = "cpu", seed: int = 0,
                 track_source: Any = "procedural",
                 step_dt: float = C.DEFAULT_STEP_DT,
                 max_episode_steps: int = C.DEFAULT_MAX_EPISODE_STEPS,
                 reward_scale: float = C.DEFAULT_REWARD_SCALE,
                 reward_mode: str = C.DEFAULT_REWARD_MODE,
                 death_penalty: float = C.DEFAULT_DEATH_PENALTY,
                 action_history_len: int = C.DEFAULT_ACTION_HISTORY_LEN,
                 frame_stack: int = C.DEFAULT_FRAME_STACK,
                 injected_track: Any = None,
                 **shaping: Any):
        import torch
        self._torch = torch
        self.dev = torch.device(device)
        if injected_track is not None:
            track_source = injected_track
        conf = _Config(config, track_source, step_dt, max_episode_steps, reward_scale,
                       reward_mode, death_penalty, action_history_len, frame_stack,
                       shaping)
        _Base.__init__(self, num_envs, conf, seed)
        self.cfg = conf.cfg
        self.K, self.FS = conf.K, conf.FS
        self.max_episode_steps = conf.max_episode_steps
        self.reward_mode = "score" if conf.reward_score else "distance"
        self.reward_scale = conf.reward_scale
        self.death_penalty = conf.death_penalty
        obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Discrete(C.N_ACTIONS)
        self.observation_space = obs_space
        self.action_space = act_space
        if _VecEnvBase is not object:
            _VecEnvBase.__init__(self, num_envs, obs_space, act_space)
        self._actions = None
        self._seed_next = None

    # ---- tensor-native fast path (train_fast.py) --------------------------
    def _t(self, a):
        return self._torch.from_numpy(a).to(self.dev)

    def reset_tensor(self):
        self._activate()
        seed = -1 if self._seed_next is None else int(self._seed_next)
        self._seed_next = None
        self._L.sim_reset_all(self._buf, self.num_envs, ctypes.c_longlong(seed),
                              self._p(self._obs, ctypes.c_float))
        return self._t(self._obs.copy())

    def step_tensor(self, a):
        torch = self._torch
        acts = a.reshape(self.num_envs).to("cpu", dtype=torch.int64).numpy()
        self._step_raw(acts, 1)
        terminated = (self._flg & 1) != 0
        over_cap = (self._flg & 2) != 0
        truncated = over_cap & ~terminated
        done = terminated | truncated
        # dtypes deliberately mirror TorchTempleRunVecEnv.step_tensor so this is a
        # drop-in: obs/term_obs float32, rew/ep_ret/ep_dist float64, ep_len int64.
        return (self._t(self._obs.copy()), self._t(self._rew.copy()),
                self._t(done.copy()), self._t(truncated.copy()),
                self._t(self._tobs.copy()), self._t(self._epr.copy()),
                self._t(self._epl.astype(np.int64)), self._t(self._epd.copy()))

    def approach_turn_hint(self):
        self._L.sim_approach_turn_hint(self._buf, self.num_envs,
                                       self._p(self._hint, ctypes.c_int))
        return self._torch.from_numpy(self._hint.astype(np.int64)).to(self.dev)

    def stepdown_walk_hint(self):
        self._L.sim_stepdown_walk_hint(self._buf, self.num_envs,
                                       self._p(self._hint, ctypes.c_int))
        return self._torch.from_numpy(self._hint.astype(np.int64)).to(self.dev)

    def cascade_hint(self):
        self._L.sim_cascade_hint(self._buf, self.num_envs,
                                       self._p(self._hint, ctypes.c_int))
        return self._torch.from_numpy(self._hint.astype(np.int64)).to(self.dev)

    # ---- SB3 VecEnv API ----------------------------------------------------
    def reset(self):
        return np.asarray(self.reset_tensor().cpu().numpy())

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(self.num_envs)

    def step_wait(self):
        self._step_raw(self._actions, 1)
        terminated = (self._flg & 1) != 0
        over_cap = (self._flg & 2) != 0
        truncated = over_cap & ~terminated
        done = terminated | truncated
        infos = [{} for _ in range(self.num_envs)]
        for i in np.nonzero(done)[0]:
            infos[i]["terminal_observation"] = self._tobs[i].copy()
            infos[i]["TimeLimit.truncated"] = bool(truncated[i])
            infos[i]["episode"] = {"r": float(self._epr[i]), "l": int(self._epl[i])}
            infos[i]["death_type"] = int(self._dty[i])
            infos[i]["distance"] = float(self._epd[i])
        return self._obs.copy(), self._rew.copy(), done.copy(), infos

    def close(self):
        return None

    def get_attr(self, attr_name, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [getattr(self, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError(f"env_method({method_name}) on the batched C++ sim")

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [False] * n

    def seed(self, seed=None):
        if seed is not None:
            self._seed_next = int(seed)
        return [seed] * self.num_envs


# ==================================================================== gym ====
class CppSimGymAdapter(gym.Env):
    """Single-env gym adapter with ``TempleRunSimEnv``'s exact step contract."""

    metadata = {"render_modes": []}

    def __init__(self, track_source: Any = "procedural", config: Optional[dict] = None,
                 seed: Optional[int] = None,
                 step_dt: float = C.DEFAULT_STEP_DT,
                 max_episode_steps: int = C.DEFAULT_MAX_EPISODE_STEPS,
                 reward_scale: float = C.DEFAULT_REWARD_SCALE,
                 reward_mode: str = C.DEFAULT_REWARD_MODE,
                 death_penalty: float = C.DEFAULT_DEATH_PENALTY,
                 action_history_len: int = C.DEFAULT_ACTION_HISTORY_LEN,
                 frame_stack: int = C.DEFAULT_FRAME_STACK,
                 device: str = "cpu", **shaping: Any):
        shaping.pop("action_delay", None)
        conf = _Config(config, track_source, step_dt, max_episode_steps, reward_scale,
                       reward_mode, death_penalty, action_history_len, frame_stack,
                       shaping)
        self._core = _Base(1, conf, 0 if seed is None else int(seed))
        self.cfg = conf.cfg
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self._core.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(C.N_ACTIONS)
        self._one = np.zeros(1, dtype=np.int32)

    # -- CPU-sim compatibility shims ----------------------------------------
    def raw_state(self) -> dict:
        return self._core.raw_dict(0)

    @property
    def _velocity(self) -> float:
        return float(self._core._L.sim_get_velocity(self._core._buf, 0))

    @property
    def _is_dead(self) -> bool:
        return bool(self._core._L.sim_get_is_dead(self._core._buf, 0))

    def _stumble(self) -> None:
        self._core._L.sim_stumble(self._core._buf, 0)

    def draws(self) -> int:
        return self._core.draws(0)

    # -- gym ----------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        c = self._core
        c._activate()
        c._L.sim_reset_all(c._buf, 1, ctypes.c_longlong(-1 if seed is None else int(seed)),
                           c._p(c._obs, ctypes.c_float))
        st = c.raw_dict(0)
        return c._obs[0].copy(), {"raw_state": st, "score": st["score"],
                                  "distance": st["distance"]}

    def step(self, action: int):
        c = self._core
        self._one[0] = int(action)
        c._step_raw(self._one, 0)
        st = c.raw_dict(0)
        terminated = bool(c._flg[0] & 1)
        truncated = bool(c._flg[0] & 2)      # CPU sim: elapsed >= cap, not gated on death
        info = {"raw_state": st, "score": st["score"], "distance": st["distance"],
                "death_type": (int(c._dty[0]) if terminated else None)}
        return c._obs[0].copy(), float(c._rew[0]), terminated, truncated, info

    def close(self):
        return None


try:
    from gymnasium.envs.registration import register as _register
    from gymnasium.envs.registration import registry as _registry

    if "TempleRunSimCpp-v0" not in _registry:
        _register(id="TempleRunSimCpp-v0",
                  entry_point="cpp_sim.cpp_vec_env:CppSimGymAdapter",
                  max_episode_steps=None, disable_env_checker=True)
except Exception:  # noqa: BLE001
    pass
