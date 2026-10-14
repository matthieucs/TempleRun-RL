#!/usr/bin/env python3
"""Measure how strongly a policy RESPONDS to one observation column -- the leading indicator that
decides whether a new cue is being learned, long before any distance metric can.

    python scripts/analysis/cue_response.py runs/ppo_v117_slh_2002944_steps.zip \
        --field slide_landing_hazard --action 2 --sim-config "$(cat experiments/_v108_cfg.json)"

WHY THIS EXISTS. Distance is a lagging, noisy verdict: a 10M arm costs ~6 minutes but its median is
only resolvable to ~2.5% with 8 paired seeds, and a cue can be completely dead while the median drifts
around inside the noise. The response is a DIRECT read: collect states the policy actually visits,
flip the one column, and measure how much the action probability moves. It answers "did the policy
learn to use this input" without any reference to outcome.

REFERENCE SCALE, from this project's own history (all measured with this same counterfactual):
    landing_hazard, mature v111 ............ -0.01163   the strongest cue ever measured here
    landing_hazard, mature v90 ............. -0.00670
    landing_hazard, eroded v104 ............ -0.00050   present but not load-bearing
    horizon obs, after 40M steps ........... -0.00005   DEAD -- reverted after three arms
A useful bar for "this is being learned" is |dP| >= 0.0005 by 2M steps, i.e. an order of magnitude
above the horizon obs's terminal value.

⚠️ USE ON-DISTRIBUTION STATES. Flipping a column on random probe vectors reads ~0 and means nothing:
the network is only sensitive to a feature in the region of state space it actually occupies. This
script therefore rolls the policy out in the sim and flips columns on the visited states.

⚠️ THE SIGN MATTERS AS MUCH AS THE SIZE. For a hazard-at-my-landing-cell cue the correct response is
NEGATIVE (see the hazard -> jump LESS). A positive response of the same magnitude means the policy
learned the cue backwards, which is worse than not learning it.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp_sim"))

import torch                                  # noqa: E402
from stable_baselines3 import PPO             # noqa: E402
from templerun_rl import obs as O             # noqa: E402
from cpp_vec_env import CppTempleRunVecEnv    # noqa: E402


def action_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
        return model.policy.get_distribution(t).distribution.probs.cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--field", default="slide_landing_hazard")
    p.add_argument("--action", type=int, default=2, help="2 = JUMP")
    p.add_argument("--on", type=float, default=1.0, help="value meaning 'hazard present'")
    p.add_argument("--off", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--sim-config", default="")
    p.add_argument("--curriculum-max-start-dist", type=float, default=5000.0)
    a = p.parse_args()

    idx = O.OBS_FIELDS.index(a.field)
    model = PPO.load(a.checkpoint, device="cpu")

    cfg = {"domain_rand": False, "turn_approach_buffer": True, "turn_linger_frames": 3,
           "curriculum_max_start_dist": a.curriculum_max_start_dist}
    if a.sim_config:
        cfg.update(json.loads(a.sim_config))
    env = CppTempleRunVecEnv(num_envs=a.num_envs, config=cfg, seed=a.seed,
                             max_episode_steps=8000, action_history_len=4, frame_stack=1)
    assert env.cfg["turn_approach_buffer"] is True, "base cfg not applied"

    obs = env.reset()
    deltas, base_p, n_on = [], [], 0
    for _ in range(a.steps):
        x = np.asarray(obs, dtype=np.float32).copy()
        n_on += int((x[:, idx] == a.on).sum())
        hi = x.copy(); hi[:, idx] = a.on
        lo = x.copy(); lo[:, idx] = a.off
        d = action_probs(model, hi)[:, a.action] - action_probs(model, lo)[:, a.action]
        deltas.append(d)
        base_p.append(action_probs(model, lo)[:, a.action])
        act, _ = model.predict(x, deterministic=True)
        obs, _, _, _ = env.step(np.asarray(act).reshape(-1))
    env.close()

    d = np.concatenate(deltas)
    b = np.concatenate(base_p)
    n = len(d)
    print("%s   field=%s (idx %d)   action=%d   n=%d states"
          % (Path(a.checkpoint).name, a.field, idx, a.action, n))
    print("  base P(action)            %8.5f" % b.mean())
    print("  dP when cue %.1f -> %.1f     %+8.5f   (s.e. %.5f)"
          % (a.off, a.on, d.mean(), d.std() / n ** 0.5))
    print("  max |dP| over states      %8.5f" % np.abs(d).max())
    print("  states naturally at cue=%.1f: %d (%.2f%%)" % (a.on, n_on, 100 * n_on / n))
    verdict = ("LEARNED" if abs(d.mean()) >= 5e-4 else "DEAD -- below the horizon-obs bar")
    print("  reference: v111 landing_hazard -0.01163 | eroded -0.00050 | horizon obs -0.00005")
    print("  VERDICT: %s%s" % (verdict, "" if d.mean() <= 0 else "  ⚠️ SIGN IS POSITIVE (backwards)"))


if __name__ == "__main__":
    main()
