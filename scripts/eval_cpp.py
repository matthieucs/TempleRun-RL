#!/usr/bin/env python3
"""Score checkpoints in the C++ sim -- THE SIMULATOR THE POLICIES ARE ACTUALLY TRAINED IN.

    python scripts/eval_cpp.py runs/a.zip runs/b.zip --episodes 128 --seed 0 \
        --sim-config "$(cat scratchpad/_v116_cfg.json)"

WHY THIS REPLACES eval_ckpt_torch.py. That script evaluates in `TorchTempleRunVecEnv`, the RETIRED
torch sim. It is not gated against the C++ trainer and is documented to drift -- and it silently
ignores anything added to the C++/CPU sims since. Concretely, on 2026-07-28 it had ZERO
implementation of `airborne_slide_kill_prob`, so v115 (trained WITH that death rule) was being
judged in a world WITHOUT it. Three arms were called that day on measurements taken in the wrong
simulator. Evaluate where you train.

Output format is line-compatible with the old script so scripts/eval_paired.py parses either.

ONE EPISODE PER ENV, and that is deliberate: with auto-reset, harvesting "the first N finished
episodes" oversamples SHORT ones and biases the median down. Using num_envs == episodes and taking
each env's FIRST episode gives N independent, unbiased samples.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp_sim"))

from stable_baselines3 import PPO            # noqa: E402
from cpp_vec_env import CppTempleRunVecEnv   # noqa: E402


def score(ckpt: str, episodes: int, seed: int, cfg: dict, max_steps: int) -> dict:
    model = PPO.load(ckpt, device="cpu")
    side = Path(str(ckpt).replace(".zip", "") + ".json")
    K, FS = 4, 1
    if side.exists():
        d = json.loads(side.read_text())
        K, FS = int(d.get("action_history_len", 4)), int(d.get("frame_stack", 1))

    env = CppTempleRunVecEnv(num_envs=episodes, config=cfg, seed=seed,
                             max_episode_steps=max_steps,
                             action_history_len=K, frame_stack=FS)
    obs = env.reset()
    dist = [None] * episodes          # first-episode distance per env
    deaths: Counter = Counter()
    for _ in range(max_steps + 5):
        act, _ = model.predict(np.asarray(obs), deterministic=True)
        obs, _rew, done, infos = env.step(np.asarray(act).reshape(-1))
        for i in np.nonzero(done)[0]:
            if dist[i] is None:                       # record ONLY the first episode
                dist[i] = float(infos[i].get("distance") or 0.0)
                dt = infos[i].get("death_type")
                if dt is not None:
                    deaths[int(dt)] += 1
        if all(d is not None for d in dist):
            break
    env.close()
    got = [d for d in dist if d is not None]
    if not got:
        return {}
    return {"median": st.median(got), "mean": st.mean(got), "max": max(got),
            "n": len(got), "deaths": dict(deaths.most_common())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=8000)
    p.add_argument("--sim-config", default="")
    # MUST mirror train_fast.py's base cfg. These are TRAINER FLAGS that land in the config, not in
    # the --sim-config JSON, so an evaluator that passes only the JSON silently runs a different
    # world: without turn_approach_buffer the pre-swipe never latches and v111 dies at the FIRST
    # junction (~109 m instead of ~3000).
    p.add_argument("--turn-approach-buffer", action="store_true", default=True)
    p.add_argument("--no-turn-approach-buffer", dest="turn_approach_buffer",
                   action="store_false")
    p.add_argument("--turn-linger", type=int, default=3)
    p.add_argument("--curriculum-max-start-dist", type=float, default=0.0)
    p.add_argument("--domain-rand", action="store_true", default=False)
    a = p.parse_args()

    cfg = {"domain_rand": a.domain_rand,
           "turn_approach_buffer": a.turn_approach_buffer,
           "turn_linger_frames": a.turn_linger,
           "curriculum_max_start_dist": a.curriculum_max_start_dist}
    if a.sim_config:
        cfg.update(json.loads(a.sim_config))
    print("[cfg] %d keys, %d episode_profiles, turn_approach_buffer=%s"
          % (len(cfg), len(cfg.get("episode_profiles") or []), cfg["turn_approach_buffer"]))
    for c in a.checkpoints:
        r = score(c, a.episodes, a.seed, cfg, a.max_episode_steps)
        if not r:
            print("%-46s NO EPISODES COMPLETED" % Path(c).name)
            continue
        print("%-42s median=%6.0f  mean=%6.0f  max=%6.0f  deaths=%s"
              % (Path(c).name, r["median"], r["mean"], r["max"], r["deaths"]), flush=True)


if __name__ == "__main__":
    main()
