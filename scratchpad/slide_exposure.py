"""How often does the runner FACE a slide hazard, sim vs device -- on an encoding-independent basis.

path_type counting is ambiguous: TreeSlide rides a STRAIGHT cell (path_type 1/2) with the obstacle in
a separate field that the C++ raw state does not expose, so a path_type census silently drops it on
the sim side. `must_slide` is a BOOLEAN defined identically in both worlds and streamed by both, so
counting its onsets is symmetric by construction. Same for must_jump, as the control.
"""
import sys, json, glob
sys.path.insert(0,"src"); sys.path.insert(0,"cpp_sim")
import numpy as np
from stable_baselines3 import PPO
from cpp_vec_env import CppTempleRunVecEnv

# ---- DEVICE: onsets per 1000 frames (10 Hz), champion-era captures ----
for label, pats in (("DEVICE v107+v111", ["captures/w17_v111_device.jsonl","captures/w17_v107_device.jsonl"]),
                    ("DEVICE pooled",    ["captures/w1*_*device*.jsonl"])):
    js = jf = sf = ff = 0
    for pat in pats:
        for f in glob.glob(pat):
            pj = ps = False
            for line in open(f):
                try: r = json.loads(line)
                except Exception: continue
                if "velocity" not in r: continue
                ff += 1
                mj, ms = bool(r.get("must_jump")), bool(r.get("must_slide"))
                if mj and not pj: jf += 1
                if ms and not ps: sf += 1
                pj, ps = mj, ms
    print("%-20s frames %6d   must_jump onsets %4d (%.1f/1k)   must_slide onsets %4d (%.1f/1k)   slide share %.1f%%"
          % (label, ff, jf, 1000*jf/max(1,ff), sf, 1000*sf/max(1,ff), 100*sf/max(1,jf+sf)))

# ---- SIM ----
m = PPO.load("runs/ppo_v111_obs66_slh.zip", device="cpu")
for name in ("_v108_cfg.json", "_v114_cfg.json"):
    cfg = json.load(open("scratchpad/"+name))
    cfg.update(dict(domain_rand=False, turn_approach_buffer=True, turn_linger_frames=3,
                    curriculum_max_start_dist=0.0))
    NE = 64
    env = CppTempleRunVecEnv(num_envs=NE, config=cfg, seed=0, max_episode_steps=8000,
                             action_history_len=4, frame_stack=1)
    assert env.cfg["curriculum_max_start_dist"] == 0.0
    o = env.reset(); seen = np.zeros(NE, bool)
    pj = np.zeros(NE, bool); ps = np.zeros(NE, bool)
    jf = sf = ff = 0
    for _ in range(4000):
        mj = np.zeros(NE, bool); ms = np.zeros(NE, bool)
        for i in range(NE):
            if seen[i]: continue
            r = env.raw_dict(i)
            mj[i], ms[i] = bool(r.get("must_jump")), bool(r.get("must_slide"))
            ff += 1
        jf += int((mj & ~pj).sum()); sf += int((ms & ~ps).sum())
        pj, ps = mj, ps if False else ms
        a, _ = m.predict(np.asarray(o), deterministic=True)
        o, _, dn, _ = env.step(np.asarray(a).reshape(-1))
        for i in np.nonzero(dn)[0]: seen[i] = True
        if seen.all(): break
    env.close()
    print("%-20s frames %6d   must_jump onsets %4d (%.1f/1k)   must_slide onsets %4d (%.1f/1k)   slide share %.1f%%"
          % ("SIM " + name[:9], ff, jf, 1000*jf/max(1,ff), sf, 1000*sf/max(1,ff), 100*sf/max(1,jf+sf)))
