#!/usr/bin/env python3
"""Calibrate crevasse_cap_arc_offset + crevasse_takeoff_std so a v51-like policy grazes ~the device
rate (~32%) at level-3 caps in-sim, ~0% at level-2, with LOW kill. v55 used a fixed offset only ->
sharp all-or-nothing; the takeoff-std spreads the descending-arc clearance -> a clean partial rate.

    python scratchpad/sweep_crevasse_variance.py [model.zip]
"""
import sys
sys.path.insert(0, "src")
from collections import defaultdict
from stable_baselines3 import PPO
from templerun_rl.sim_env import TempleRunSimEnv, _MUST_JUMP, OBST_LEVEL_CHANGE
import templerun_rl.obs as O
import numpy as np

# device-true config (matches the training / device-eval cfg) + the crevasse world-Y graze model ON.
BASE = {"jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True, "jump_drop_prob": 0.1,
        "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0,
        "stepup_clip_band": 0.25, "wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
        "crevasse_graze_worldy": True, "turn_approach_buffer": True, "turn_linger_frames": 3,
        "curriculum_max_start_dist": 0.0}

# centered on the analytic estimate (offset ~0.4 puts the cap on the descending arc at clearance ~+2;
# std ~0.27 = the measured device takeoff_cp std -> ~30% of crossings dip below the lvl-3 cap).
GRID = [(off, std) for off in (0.3, 0.4, 0.5) for std in (0.2, 0.27, 0.35)]


def run(model_path, cfg, n_eps=60, seed0=1000):
    m = PPO.load(model_path, device="cpu")
    K = max(0, (int(m.observation_space.shape[0]) - O.OBS_LEN) // 7)
    env = TempleRunSimEnv(config=cfg, max_episode_steps=4000, reward_mode="distance",
                          action_history_len=K, frame_stack=1)
    counts = defaultdict(lambda: {"clear": 0, "graze": 0, "kill": 0})
    orig = env._resolve_obstacle_on_enter

    def patched(cell):
        obst = int(cell.get("obstacles", 0)); lvl = int(cell.get("level", 2))
        nxt = env._track[env._cell_index + 1] if env._cell_index + 1 < len(env._track) else {}
        is_cap = (obst == OBST_LEVEL_CHANGE and cell.get("wide_gap") and not nxt.get("wide_gap")) or bool(cell.get("stepup"))
        dead0 = env._is_dead; st0 = env._stumble_t
        orig(cell)
        if is_cap:
            if env._is_dead and not dead0:
                counts[lvl]["kill"] += 1
            elif env._stumble_t > st0:
                counts[lvl]["graze"] += 1
            else:
                counts[lvl]["clear"] += 1
    env._resolve_obstacle_on_enter = patched
    for ep in range(n_eps):
        o, i = env.reset(seed=seed0 + ep); hist = []

        def obsv():
            base = O.state_to_obs(env._snapshot())
            if K:
                h = [int(a) for a in hist[-K:]]; h = [6] * (K - len(h)) + h
                oh = np.zeros(K * 7, dtype=np.float32)
                for j, a in enumerate(h):
                    oh[j * 7 + a] = 1.0
                return np.concatenate([base, oh])
            return base
        obs = obsv(); done = False; steps = 0
        while not done and steps < 4000:
            a, _ = m.predict(obs, deterministic=True); a = int(a)
            o, r, t, tr, info = env.step(a); hist.append(a); obs = obsv()
            done = t or tr; steps += 1
    env.close()
    return counts


def rate(c):
    n = c["clear"] + c["graze"] + c["kill"]
    return n, (100 * c["graze"] / n if n else 0), (100 * c["kill"] / n if n else 0)


mp = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_v51_next2_best.zip"
print(f"model={mp}  TARGET: lvl3 graze~32% kill-low, lvl2 graze~0%\n")
print(f"{'offset':>6} {'std':>5} | {'lvl3 n':>6} {'graze%':>6} {'kill%':>5} | {'lvl2 n':>6} {'graze%':>6} {'kill%':>5}")
for off, std in GRID:
    cfg = dict(BASE, crevasse_cap_arc_offset=off, crevasse_takeoff_std=std)
    c = run(mp, cfg, n_eps=40)
    n3, g3, k3 = rate(c[3]); n2, g2, k2 = rate(c[2])
    print(f"{off:>6.1f} {std:>5.1f} | {n3:>6} {g3:>6.0f} {k3:>5.0f} | {n2:>6} {g2:>6.0f} {k2:>5.0f}")
