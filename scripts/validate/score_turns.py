#!/usr/bin/env python3
"""Per-TURN reliability + max-distance scorer — the metric that gates the 10,000m goal.

A 10,000m run crosses ~100+ junctions, so per-turn reliability must be ~99.99% (95% => dead in ~20
turns). This runs many episodes on a TURN-DENSE domain-randomized config, counts junctions FACED vs
turn-deaths, and reports the per-turn failure rate + reliability %, plus the distance distribution.

Obs shape (frame_stack, action_history_len) is read from the model's sidecar ``<model>.json`` if
present (written by train_sim), else inferred assuming frame_stack=1.

    python scripts/validate/score_turns.py runs/ppo_hist4.zip --episodes 200
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import templerun_rl.obs as O  # noqa: E402
from templerun_rl.sim_env import TempleRunSimEnv  # noqa: E402

# Turn-stress with a PHONE-REALISTIC TIGHT window: the easy default window (DR 0.40-0.62) is solved
# at 100% by the current model in-sim, but the phone's is_turn is ~1 frame, so the gap shows only
# under a tight window. dr_turn_window_frac [0.22,0.40] makes the sim's turn timing match the phone
# (oracle ~70% here) -> per-turn reliability becomes discriminating and a proxy for on-device turns.
TURN_CONFIG = {
    "domain_rand": True,
    "dr_turn_window_frac": [0.22, 0.40],
    "turn_prob": 0.35, "turn_prob_late": 0.6,
    "min_space_between_turns": 7, "min_space_between_turns_late": 6,
    "ramp_distance": 2500.0,
}


def obs_cfg(model_path: str, model) -> tuple[int, int]:
    side = Path(str(model_path).replace(".zip", "") + ".json")
    if side.exists():
        d = json.loads(side.read_text())
        return int(d.get("frame_stack", 1)), int(d.get("action_history_len", 0))
    # fallback: assume no frame stacking -> K from the remaining width.
    dim = int(model.observation_space.shape[0])
    return 1, max(0, (dim - O.OBS_LEN) // 7)


def score(model_path: str, episodes: int = 200, cap: int = 6000, seed0: int = 30000):
    m = PPO.load(model_path)
    fs, K = obs_cfg(model_path, m)
    e = TempleRunSimEnv(max_episode_steps=cap, config=TURN_CONFIG,
                        frame_stack=fs, action_history_len=K)
    junctions = turn_deaths = other_deaths = 0
    dists, reached_10k = [], 0
    for ep in range(episodes):
        o, i = e.reset(seed=seed0 + ep)
        st = i["raw_state"]
        prev_turn = bool(st.get("is_turn"))
        while True:
            a, _ = m.predict(o, deterministic=True)
            o, r, t, tr, info = e.step(int(a))
            st = info["raw_state"]
            cur = bool(st.get("is_turn"))
            if cur and not prev_turn:
                junctions += 1            # a new junction came into view
            prev_turn = cur
            if t:
                if info["death_type"] == 1:
                    turn_deaths += 1
                else:
                    other_deaths += 1
                break
            if tr:
                break
        d = float(info["distance"])
        dists.append(d)
        if d >= 10000:
            reached_10k += 1
    rate = turn_deaths / max(1, junctions)
    return {
        "fs": fs, "K": K, "junctions": junctions, "turn_deaths": turn_deaths,
        "other_deaths": other_deaths, "per_turn_fail": rate, "per_turn_reliab_pct": 100 * (1 - rate),
        "median_dist": float(np.median(dists)), "max_dist": float(np.max(dists)),
        "reached_10k": reached_10k, "episodes": episodes,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--cap", type=int, default=6000)
    a = p.parse_args()
    r = score(a.model, a.episodes, a.cap)
    print(f"{a.model}  [fs={r['fs']} K={r['K']}]")
    print(f"  junctions faced : {r['junctions']}")
    print(f"  turn-deaths     : {r['turn_deaths']}  (other deaths {r['other_deaths']})")
    print(f"  PER-TURN reliab : {r['per_turn_reliab_pct']:.3f}%   (fail rate {r['per_turn_fail']:.5f})")
    print(f"  distance        : median {r['median_dist']:.0f}  max {r['max_dist']:.0f}  "
          f"reached 10k: {r['reached_10k']}/{r['episodes']}")


if __name__ == "__main__":
    main()
