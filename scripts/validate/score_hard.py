#!/usr/bin/env python3
"""Score a trained model on the HARD held-out sim config (tight turn window, dense turns, fast ramp).

The normal sim is saturated by a small net (0 deaths / cap), so it can't compare architectures.
This config keeps real headroom (oracle survives but loses ~30% to the tight window) so a more
capable policy scores measurably higher. Used by the net-size sweep.

    python scripts/validate/score_hard.py runs/ppo_sim_zones_final.zip --episodes 60
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from templerun_rl.sim_env import TempleRunSimEnv  # noqa: E402

HARD_CONFIG = {
    "domain_rand": True,
    "dr_turn_window_frac": [0.28, 0.42],
    "turn_prob": 0.30, "turn_prob_late": 0.6,
    "min_space_between_turns": 8,
    "ramp_distance": 2500.0,
}


def score(model_path: str, episodes: int = 60, cap: int = 2000, seed0: int = 90000):
    m = PPO.load(model_path)
    # obs shape from the sidecar (frame_stack & action_history_len); fallback assumes frame_stack=1.
    import json
    from pathlib import Path
    import templerun_rl.obs as O
    side = Path(str(model_path).replace(".zip", "") + ".json")
    if side.exists():
        d = json.loads(side.read_text()); fs, K = int(d.get("frame_stack", 1)), int(d.get("action_history_len", 0))
    else:
        fs, K = 1, max(0, (int(m.observation_space.shape[0]) - O.OBS_LEN) // 7)
    e = TempleRunSimEnv(max_episode_steps=cap, config=HARD_CONFIG, frame_stack=fs, action_history_len=K)
    steps, dists, deaths, dt = [], [], 0, Counter()
    for ep in range(episodes):
        o, i = e.reset(seed=seed0 + ep)
        s = 0
        while True:
            a, _ = m.predict(o, deterministic=True)
            o, r, t, tr, info = e.step(int(a))
            s += 1
            if t:
                deaths += 1
                dt[info["death_type"]] += 1
                break
            if tr:
                break
        steps.append(s)
        dists.append(info["distance"])
    return {
        "median_steps": float(np.median(steps)),
        "mean_steps": float(np.mean(steps)),
        "median_dist": float(np.median(dists)),
        "deaths": deaths,
        "episodes": episodes,
        "death_types": dict(dt),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=60)
    args = p.parse_args()
    r = score(args.model, args.episodes)
    print(f"{args.model}")
    print(f"  median_steps={r['median_steps']:.0f} mean_steps={r['mean_steps']:.0f} "
          f"median_dist={r['median_dist']:.0f} deaths={r['deaths']}/{r['episodes']} "
          f"dtypes={r['death_types']}  (1=turn 2=tree 3=slide 5=flame)")


if __name__ == "__main__":
    main()
