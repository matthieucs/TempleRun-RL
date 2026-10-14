"""Smoke-check slide_landing_hazard: width, index, and that BOTH sims actually fill it."""
import sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "cpp_sim")
from templerun_rl import obs as O
import gymnasium as gym, templerun_rl  # noqa
from cpp_vec_env import CppTempleRunVecEnv
from cpp_vec_env import lib as _cpplib

idx = len(O.OBS_BOOL_FIELDS) + len(O.OBS_SCALAR_FIELDS) + O.OBS_EXTRA_FIELDS.index("slide_landing_hazard")
print("OBS_LEN=%d  slide_landing_hazard at index %d  (last=%s)" % (O.OBS_LEN, idx, idx == O.OBS_LEN - 1))
assert O.OBS_FIELDS[idx] == "slide_landing_hazard"

cfg = json.load(open("scratchpad/_v114_cfg.json"))
cfg.update(dict(domain_rand=False, turn_approach_buffer=True, turn_linger_frames=3, curriculum_max_start_dist=6000.0))

# --- C++ trainer ---
env = CppTempleRunVecEnv(num_envs=64, config=cfg, seed=0, max_episode_steps=2000,
                         action_history_len=4, frame_stack=1)
for k, v in (("turn_approach_buffer", True),):
    assert env.cfg[k] == v, "config not applied!"
print("C++ sim_obs_base=%d" % _cpplib().sim_obs_base())
assert _cpplib().sim_obs_base() == O.OBS_LEN, "C++ OBS_BASE != python OBS_LEN"
o = env.reset(); cols = []
rng = np.random.default_rng(0)
for _ in range(1200):
    a = rng.integers(0, 7, size=64)
    o, _, _, _ = env.step(a)
    cols.append(np.asarray(o)[:, idx].copy())
env.close()
c = np.concatenate(cols)
print("C++   col%d: uniq=%s  rate(1.0)=%.4f  rate(0.5)=%.4f" %
      (idx, np.unique(c), (c == 1.0).mean(), (c == 0.5).mean()))

# --- CPU reference (obs.state_to_obs verbatim) ---
e = gym.make("TempleRunSim-v0", config=cfg, action_history_len=4)
vals = []
for ep in range(6):
    ob, _ = e.reset(seed=ep)
    for _ in range(2000):
        vals.append(ob[idx])
        ob, _, term, trunc, _ = e.step(int(rng.integers(0, 7)))
        if term or trunc: break
v = np.asarray(vals)
print("CPU   col%d: uniq=%s  rate(1.0)=%.4f  rate(0.5)=%.4f" %
      (idx, np.unique(v), (v == 1.0).mean(), (v == 0.5).mean()))
