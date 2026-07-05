#!/usr/bin/env python
"""Offline gate for turn-timing retrains (v29+): swipe-cp distribution + death breakdown.

Runs a checkpoint under the DEVICE-TRUE sim config (hook_turn_buffer + late gate + approach buffer +
jump exec lat 1/jitter 0) and reports:
  - swipe-cp distribution of junction-cell TL/TR actions (med / p90) + approach-cell swipe fraction
    (approach swipes = the desired early behavior the late gate is meant to teach)
  - deaths bucketed by mechanic (missed-turn / crevasse / narrow / hole / obstacle / cliff-edge / other)
  - median + best distance over N episodes

Usage: .venv/bin/python scratchpad/eval_swipe_deaths.py runs/ckpt.zip [more.zip ...] [--episodes 30]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gymnasium as gym  # noqa: E402
import templerun_rl  # noqa: F401, E402

DEVICE_TRUE = {
    "hook_turn_buffer": True,
    "turn_approach_buffer": True,
    # turn_buffer_late_cp: default 0.50
    "jump_exec_lat_frames": 1,
    "jump_exec_lat_jitter": 0,   # device-measured: exactly 1 frame, zero jitter
    "domain_rand": False,
}

CREVASSE_TYPES = set(range(33, 37))
HOLE_TYPES = set(range(11, 17))
NARROW_TYPES = set(range(17, 31))
TURN_TYPES = set(range(3, 11))


def classify_death(env, death_type):
    cell = env._track[env._cell_index]
    pt = int(cell.get("type", 0))
    plvl = int(cell.get("level", 2))
    if pt in CREVASSE_TYPES:
        return "crevasse"
    if cell.get("is_turn") or pt in TURN_TYPES:
        return "missed_turn"
    if pt in NARROW_TYPES:
        return "narrow"
    if pt in HOLE_TYPES:
        return "hole"
    if death_type == 1 and plvl >= 3:
        return "cliff_edge"
    if death_type in (2, 3, 5):
        return "obstacle"
    return f"other(dt={death_type},pt={pt})"


def eval_model(path, episodes, seed0=20000):
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    env = gym.make("TempleRunSim-v0", max_episode_steps=2000,
                   action_history_len=4, frame_stack=1,
                   config=dict(DEVICE_TRUE), seed=seed0)
    core = env.unwrapped

    dists, deaths, junc_cps, approach_swipes, junc_swipes = [], {}, [], 0, 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            if a in (4, 5):
                cell = core._track[core._cell_index]
                if cell.get("is_turn"):
                    junc_cps.append(core._cell_progress())
                    junc_swipes += 1
                else:
                    nxt = core._track[core._cell_index + 1] if core._cell_index + 1 < len(core._track) else {}
                    if nxt.get("is_turn"):
                        approach_swipes += 1
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            if term:
                cause = classify_death(core, core._death_type)
                deaths[cause] = deaths.get(cause, 0) + 1
        dists.append(info.get("distance", core._distance_real))
    env.close()

    cps = np.array(junc_cps) if junc_cps else np.array([np.nan])
    tot_sw = junc_swipes + approach_swipes
    return {
        "median": float(np.median(dists)), "best": float(np.max(dists)),
        "deaths": deaths, "n_deaths": sum(deaths.values()), "episodes": episodes,
        "swipe_cp_med": float(np.nanmedian(cps)), "swipe_cp_p90": float(np.nanpercentile(cps, 90)),
        "approach_frac": (approach_swipes / tot_sw) if tot_sw else float("nan"),
        "n_junction_swipes": junc_swipes, "n_approach_swipes": approach_swipes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()
    for m in args.models:
        r = eval_model(m, args.episodes)
        name = Path(m).stem
        print(f"\n=== {name} ({r['episodes']} eps, device-true cfg) ===")
        print(f"  distance: median {r['median']:.0f}  best {r['best']:.0f}  deaths {r['n_deaths']}/{r['episodes']}")
        print(f"  junction swipe-cp: med {r['swipe_cp_med']:.2f}  p90 {r['swipe_cp_p90']:.2f} "
              f"(n={r['n_junction_swipes']})")
        print(f"  approach-cell swipes: {r['n_approach_swipes']} ({r['approach_frac']*100:.0f}% of turn swipes)")
        print(f"  deaths by cause: {dict(sorted(r['deaths'].items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
