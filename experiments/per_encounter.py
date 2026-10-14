"""PER-ENCOUNTER death rate by hazard type, sim vs device -- the basis that separates LETHALITY from
FREQUENCY. A death-mix SHARE confounds the two: the sim could kill often per crevasse and still show
a small Ledge share if crevasses are rare. This asks the clean question: given that the runner meets
a crevasse (or a flame), how often does it die there?

Encounter = the runner ENTERS a cell of that type (counted once per cell). Device path_type:
31-32 flame, 33-36 crevasse/level-cap, 11-16 hole. Same definition on both sides.
"""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, json, glob
sys.path.insert(0, "src"); sys.path.insert(0, "cpp_sim")
import numpy as np
from stable_baselines3 import PPO
from cpp_vec_env import CppTempleRunVecEnv

def kind(pt):
    pt = int(pt or 0)
    if 31 <= pt <= 32: return "flame"
    if 33 <= pt <= 36: return "crevasse"
    if 11 <= pt <= 16: return "hole"
    return None

# ---------------- DEVICE ----------------
dev_enc, dev_die = {}, {}
for f in glob.glob("captures/*.jsonl"):
    if "human" in f or "menuonly" in f: continue
    ep = None; last_cell = None
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if "velocity" not in r: continue
        if r.get("ep") != ep: ep, last_cell = r.get("ep"), None
        k = kind(r.get("path_type"))
        # ENCOUNTER = a TRANSITION onto a hazard kind, identical to the sim side. Keying on
        # (distance, path_type) counted every FRAME as a fresh encounter -- distance advances each
        # frame -- which inflated the device denominator ~3-4x and understated its death rate.
        if k != last_cell:
            if k: dev_enc[k] = dev_enc.get(k, 0) + 1
            last_cell = k
        if k and r.get("death_type"):
            dev_die[k] = dev_die.get(k, 0) + 1

# ---------------- SIM ----------------
cfg = json.load(open("experiments/_v108_cfg.json"))
cfg.update(dict(domain_rand=False, turn_approach_buffer=True, turn_linger_frames=3,
                curriculum_max_start_dist=0.0))
model = PPO.load("runs/ppo_v111_obs66_slh.zip", device="cpu")
sim_enc, sim_die = {}, {}
NE = 96
for sd in range(4):
    env = CppTempleRunVecEnv(num_envs=NE, config=cfg, seed=sd, max_episode_steps=8000,
                             action_history_len=4, frame_stack=1)
    o = env.reset(); seen = np.zeros(NE, bool)
    prev = [None]*NE      # kind of the cell the env is ON (sticky: only cleared by a new hazard)
    onhaz = [None]*NE     # kind of the MOST RECENT hazard cell, for death attribution
    for _ in range(8005):
        # READ BEFORE STEPPING: after step() an auto-reset has already replaced the raw state, so a
        # post-step read attributes the death to the NEXT episode's first cell (this returned 0
        # deaths for every hazard -- all-zeros is the tell that a harness is broken, not a result).
        for i in range(NE):
            if seen[i]: continue
            k = kind(env.raw_dict(i).get("path_type"))
            if k != prev[i]:
                if k: sim_enc[k] = sim_enc.get(k, 0) + 1
                prev[i] = k
            onhaz[i] = k   # ON this cell right now, or None. MUST match the device rule below
                           # (death counted only if it happens ON the hazard cell). Keeping this
                           # sticky pinned every later death to the last hazard ever seen and
                           # inflated the sim column ~4x.
        a, _ = model.predict(np.asarray(o), deterministic=True)
        o, _, dn, inf = env.step(np.asarray(a).reshape(-1))
        for i in np.nonzero(dn)[0]:
            if not seen[i]:
                seen[i] = True
                if onhaz[i]: sim_die[onhaz[i]] = sim_die.get(onhaz[i], 0) + 1
        if seen.all(): break
    env.close()

print("%-10s %-8s %10s %8s %9s" % ("hazard", "source", "encounters", "deaths", "rate"))
for k in ("crevasse", "flame", "hole"):
    for src, e, d in (("SIM", sim_enc, sim_die), ("DEVICE", dev_enc, dev_die)):
        n, x = e.get(k, 0), d.get(k, 0)
        print("%-10s %-8s %10d %8d %8.2f%%" % (k, src, n, x, 100*x/max(1, n)))
    print()
