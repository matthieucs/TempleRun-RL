"""Dump C++ trainer rollouts in DEVICE CAPTURE FORMAT, so every device forensic runs on sim data.

The CPU-ref dumper (sim_patterns.py) needs ~1500 s for 24 episodes; the tight-chain defect occurs on
~15% of chains, so characterising it needs hundreds of episodes. The C++ trainer produces the same
raw fields ~7x faster and is the world the policy actually trains in.

    python experiments/cpp_rollout_dump.py runs/x.zip --episodes 256 --out experiments/_sim.jsonl
"""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "cpp_sim"))
import numpy as np
from stable_baselines3 import PPO
from cpp_vec_env import CppTempleRunVecEnv

p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("--episodes", type=int, default=128)
p.add_argument("--sim-config", default=""); p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--curriculum-max-start-dist", type=float, default=0.0)
a = p.parse_args()

cfg = {"domain_rand": False, "turn_approach_buffer": True, "turn_linger_frames": 3,
       "curriculum_max_start_dist": a.curriculum_max_start_dist}
if a.sim_config:
    cfg.update(json.loads(a.sim_config))
model = PPO.load(a.model, device="cpu")
NE = a.episodes
env = CppTempleRunVecEnv(num_envs=NE, config=cfg, seed=a.seed, max_episode_steps=8000,
                         action_history_len=4, frame_stack=1)
assert env.cfg["turn_approach_buffer"] is True
obs = env.reset()
done_once = np.zeros(NE, bool)
rows = []
for _ in range(8005):
    act, _ = model.predict(np.asarray(obs), deterministic=True)
    act = np.asarray(act).reshape(-1)
    for i in range(NE):
        if done_once[i]:
            continue
        r = env.raw_dict(i)
        r["ep"] = int(i)
        r["action"] = int(act[i])
        rows.append(r)
    obs, _, dn, inf = env.step(act)
    for i in np.nonzero(dn)[0]:
        if not done_once[i]:
            done_once[i] = True
            if rows and inf[i].get("death_type"):
                # mark the LAST row of that episode as the death frame
                for k in range(len(rows) - 1, -1, -1):
                    if rows[k]["ep"] == i:
                        rows[k]["death_type"] = int(inf[i]["death_type"])
                        rows[k]["is_dead"] = True
                        break
    if done_once.all():
        break
env.close()
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote %d frames / %d episodes -> %s" % (len(rows), NE, a.out))
