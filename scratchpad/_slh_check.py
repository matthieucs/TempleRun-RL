"""Does the C++ trainer's obs builder equal obs.state_to_obs -- the encoder the DEVICE uses?

This is the only comparison that matters for deployment: the policy is trained on the C++ vector and
deployed on the obs.py vector, so any divergence is a silent sim-to-real gap. It is a per-frame
recompute from the C++'s OWN raw state, not a track-level equivalence run.
"""
import sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "cpp_sim")
from templerun_rl import obs as O
from cpp_vec_env import CppTempleRunVecEnv

cfg = json.load(open("scratchpad/_v114_cfg.json"))
cfg.update(dict(domain_rand=False, turn_approach_buffer=True, turn_linger_frames=3,
                curriculum_max_start_dist=6000.0))
env = CppTempleRunVecEnv(num_envs=32, config=cfg, seed=7, max_episode_steps=4000,
                         action_history_len=4, frame_stack=1)
assert env.cfg["curriculum_max_start_dist"] == 6000.0
o = env.reset()
rng = np.random.default_rng(1)
worst = np.zeros(O.OBS_LEN); n = 0; hits = 0
for _ in range(3000):
    for e in (0, 5, 11, 23):
        ref = O.state_to_obs(env.raw_dict(e))
        got = np.asarray(o)[e, :O.OBS_LEN]
        worst = np.maximum(worst, np.abs(ref - got))
        n += 1
        hits += int(ref[O.OBS_LEN - 1] == 1.0)
    o, _, _, _ = env.step(rng.integers(0, 7, size=32))
bad = np.nonzero(worst > 1e-6)[0]
print("compared %d frames; slide_landing_hazard fired on %d (%.2f%%)" % (n, hits, 100 * hits / n))
print("max |cpp - obs.py| per column: %.3e" % worst.max())
if len(bad):
    for i in bad:
        print("  MISMATCH %-24s idx %2d  %.6f" % (O.OBS_FIELDS[i], i, worst[i]))
else:
    print("ALL %d COLUMNS MATCH -- trainer obs == device obs" % O.OBS_LEN)
