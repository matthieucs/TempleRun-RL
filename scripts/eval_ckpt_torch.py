#!/usr/bin/env python3
"""Fast batched checkpoint eval on the torch sim, device-true config — the quick verdict
tool between training and the full eval_swipe_deaths/device gates.

    python scripts/eval_ckpt_torch.py runs/ppo_best.zip runs/ppo_v41_torch256_final.zip ...
    (--episodes N, --seed S; K/frame-stack read from the model's obs shape: 58 -> K=4)

Reports per checkpoint: median/mean/max distance + death-type mix over N fresh episodes.
Same config for every model -> directly comparable columns.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import templerun_rl  # noqa: F401,E402
from templerun_rl.sim_env_torch import TorchTempleRunVecEnv  # noqa: E402
import templerun_rl.obs as O  # noqa: E402
import templerun_rl.constants as C  # noqa: E402

DEVICE_TRUE_CFG = {
    "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True,
    "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1,
    "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "domain_rand": False,
    "turn_approach_buffer": True, "turn_linger_frames": 3,
    "curriculum_max_start_dist": 0.0,
    "stepup_clip_band": 0.25,   # device-true since 2026-07-06 (non-fatal marginal clip)
    "wide_gap_landing_model": True,   # device-true since 2026-07-06 (2-cell holes: arc must land past
    "wide_gap_land_grace": 0.35,      # the far edge; short arc clips the far wall -> dt4 Ledge)
    "crevasse_apex_all": True,        # device-true since 2026-07-06 (ALL crevasses need near-apex at the
                                      # ledge; early jump -> graze/stumble slowdown, was unmodelled for step-down)
    "turn_send_lat_frames": 0.4,      # device-true since 2026-07-06 (on-junction turn byte arrives ~late;
                                      # at speed it dead-letters -> off-track. Policy must pre-swipe approach)
}


def infer_k_fs(obs_dim: int) -> tuple[int, int]:
    """Solve obs_dim = FS*OBS_LEN + K*7 for small FS/K (unique for the shapes in use)."""
    for fs in (1, 2, 3, 4):
        rem = obs_dim - fs * O.OBS_LEN
        if rem >= 0 and rem % C.N_ACTIONS == 0 and rem // C.N_ACTIONS <= 8:
            return rem // C.N_ACTIONS, fs
    raise ValueError(f"cannot infer K/frame_stack from obs dim {obs_dim}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=24680)
    p.add_argument("--max-episode-steps", type=int, default=4000)
    p.add_argument("--sim-config", default="", help="JSON dict merged into DEVICE_TRUE_CFG (e.g. the "
                   "crevasse-fix knobs) so a model is evaluated in the env it was trained on")
    p.add_argument("--land-span", default="",
                   help="A,B overriding obs.LAND_SPAN_A/B. Pass '0.0204,-0.94' for any model trained "
                        "before the 2026-07-27 span correction (v90, v91*) — otherwise landing_hazard "
                        "is computed with a law they never trained under.")
    args = p.parse_args()

    if args.land_span:
        O.LAND_SPAN_A, O.LAND_SPAN_B = (float(x) for x in args.land_span.split(","))
        print(f"[land-span pinned] {O.LAND_SPAN_A}, {O.LAND_SPAN_B}")

    import json as _json
    eval_cfg = dict(DEVICE_TRUE_CFG)
    if args.sim_config:
        eval_cfg.update(_json.loads(args.sim_config))
        print(f"[eval cfg override] {args.sim_config}")

    for ck in args.checkpoints:
        m = PPO.load(ck, device="cpu")
        K, FS = infer_k_fs(int(m.observation_space.shape[0]))
        B = args.episodes
        v = TorchTempleRunVecEnv(B, config=eval_cfg, device="cpu", seed=args.seed,
                                 max_episode_steps=args.max_episode_steps,
                                 reward_mode="distance", action_history_len=K, frame_stack=FS)
        obs = v.reset()
        done_once = np.zeros(B, dtype=bool)
        dist = np.zeros(B)
        deaths = Counter()
        while not done_once.all():
            a, _ = m.predict(obs, deterministic=True)
            obs, r, d, infos = v.step(a)
            for i in np.nonzero(d)[0]:
                if not done_once[i]:
                    done_once[i] = True
                    dist[i] = infos[i].get("distance", 0.0)
                    if not infos[i].get("TimeLimit.truncated"):
                        deaths[int(infos[i].get("death_type") or 0)] += 1
        print(f"{Path(ck).name:<42} median={np.median(dist):6.0f}  mean={dist.mean():6.0f}  "
              f"max={dist.max():6.0f}  deaths={dict(deaths)}", flush=True)


if __name__ == "__main__":
    main()
