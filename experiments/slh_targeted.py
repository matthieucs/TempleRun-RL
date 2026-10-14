"""TARGETED metric for v117: the slide-type death RATE, not the median.

The cue addresses ~3% of deaths. eval_paired resolves ~2.5% of the MEDIAN -- a different unit and a
far blunter one, so a real fix to this mode is invisible there BY CONSTRUCTION. This counts the
mode itself over hundreds of deaths, which is the discipline CLAUDE.md already prescribes
("PREFER RATE METRICS ... they separate cleanly where medians cannot").

DeathType: 1 Fall, 2 Tree, 3 Slide, 5 Burnt. 3 and 5 are the airborne-into-slide family.
Matched env seeds per arm so track luck is a common random number.
"""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "cpp_sim")
from collections import Counter
import numpy as np
from stable_baselines3 import PPO
from cpp_vec_env import CppTempleRunVecEnv

cfg = json.load(open("experiments/_v108_cfg.json"))
cfg.update(dict(domain_rand=False, turn_approach_buffer=True, turn_linger_frames=3,
                curriculum_max_start_dist=5000.0))
ARMS = [("v111", "runs/ppo_v111_obs66_slh.zip"),
        ("v117", "runs/ppo_v117_slh_6000640_steps.zip")]
SEEDS = range(12)

for name, ck in ARMS:
    model = PPO.load(ck, device="cpu")
    mix, dist = Counter(), []
    for sd in SEEDS:
        env = CppTempleRunVecEnv(num_envs=128, config=cfg, seed=sd, max_episode_steps=8000,
                                 action_history_len=4, frame_stack=1)
        assert env.cfg["curriculum_max_start_dist"] == 5000.0
        o = env.reset(); done_once = np.zeros(128, bool)
        for _ in range(8005):
            a, _ = model.predict(np.asarray(o), deterministic=True)
            o, _, dn, inf = env.step(np.asarray(a).reshape(-1))
            for i in np.nonzero(dn)[0]:
                if not done_once[i]:
                    done_once[i] = True
                    dt = inf[i].get("death_type")
                    if dt is not None: mix[int(dt)] += 1
                    dist.append(float(inf[i].get("distance") or 0))
            if done_once.all(): break
        env.close()
    tot = sum(mix.values())
    slide = mix[3] + mix[5]
    print("%-5s n=%4d deaths  slide-type(3+5) %4d = %5.2f%%   median %5.0f   mix %s"
          % (name, tot, slide, 100*slide/max(1,tot), np.median(dist), dict(mix.most_common())))
