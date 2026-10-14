"""Sim-vs-device DEATH MIX on a MATCHED basis.

The naive comparison is invalid: the sim rollouts start at a random distance up to 5000 m, so they
spend their whole life in flame/slide territory, while device runs start at 0 and spend their early
metres in a zone where flames do not exist yet. Any flame-share gap would be manufactured by that
alone. So: run the sim from distance 0 like the device, and bucket BOTH sides by the distance at
death. Same statistic, same code, both sides -- the rule today's pair-statistic retraction produced.
"""
import sys, json, glob
sys.path.insert(0, "src"); sys.path.insert(0, "cpp_sim")
from collections import Counter, defaultdict
import numpy as np
from stable_baselines3 import PPO
from cpp_vec_env import CppTempleRunVecEnv

NAMES = {1: "Fall", 2: "Tree", 3: "Slide", 4: "Ledge", 5: "Burnt", 6: "6", 7: "7", 9: "9"}
BANDS = [(0, 1000), (1000, 3000), (3000, 99999)]
def band(d):
    for lo, hi in BANDS:
        if lo <= d < hi: return "%d-%d" % (lo, hi)
    return "?"

# ---------------- SIM, started at 0 like the device ----------------
cfg = json.load(open("scratchpad/_v108_cfg.json"))
cfg.update(dict(domain_rand=False, turn_approach_buffer=True, turn_linger_frames=3,
                curriculum_max_start_dist=0.0))
model = PPO.load("runs/ppo_v111_obs66_slh.zip", device="cpu")
sim = defaultdict(Counter)
for sd in range(8):
    env = CppTempleRunVecEnv(num_envs=128, config=cfg, seed=sd, max_episode_steps=8000,
                             action_history_len=4, frame_stack=1)
    assert env.cfg["curriculum_max_start_dist"] == 0.0
    o = env.reset(); seen = np.zeros(128, bool)
    for _ in range(8005):
        a, _ = model.predict(np.asarray(o), deterministic=True)
        o, _, dn, inf = env.step(np.asarray(a).reshape(-1))
        for i in np.nonzero(dn)[0]:
            if not seen[i]:
                seen[i] = True
                dt = inf[i].get("death_type")
                if dt: sim[band(float(inf[i].get("distance") or 0))][int(dt)] += 1
        if seen.all(): break
    env.close()

# ---------------- DEVICE ----------------
dev = defaultdict(Counter)
for f in glob.glob("captures/*.jsonl"):
    if "human" in f or "menuonly" in f: continue
    ep, last = None, None
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if "velocity" not in r: continue
        if r.get("ep") != ep:
            ep, last = r.get("ep"), None
        if r.get("death_type") and last is None:
            last = 1
            dev[band(float(r.get("distance") or 0))][int(r["death_type"])] += 1

print("%-12s %-8s %s" % ("band", "source", "  ".join("%-7s" % NAMES.get(k, k) for k in (1,2,3,4,5))))
for lo, hi in BANDS:
    b = "%d-%d" % (lo, hi)
    for src, tbl in (("SIM", sim), ("DEVICE", dev)):
        c = tbl[b]; t = sum(c.values())
        if not t: continue
        print("%-12s %-8s %s   n=%d" % (b, src,
              "  ".join("%6.1f%%" % (100*c[k]/t) for k in (1,2,3,4,5)), t))
    print()
