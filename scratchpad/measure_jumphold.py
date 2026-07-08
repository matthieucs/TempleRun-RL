#!/usr/bin/env python3
"""Measure the HELD-JUMP rate for policies in the sim: fraction of JUMP presses that immediately
FOLLOW another JUMP (consecutive) = the hold that produces the lethal high-ground landing-frame
re-jump. Lower = the jump_hold_penalty worked. Also reports median distance (crevasse-fix cfg)."""
import sys, statistics as S
sys.path.insert(0, "src")
from stable_baselines3 import PPO
from templerun_rl.sim_env import TempleRunSimEnv
import templerun_rl.obs as O
import numpy as np

CFG = {"jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True, "jump_drop_prob": 0.1,
       "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0,
       "stepup_clip_band": 0.25, "wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
       "crevasse_graze_worldy": True, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2,
       "turn_approach_buffer": True, "turn_linger_frames": 3, "curriculum_max_start_dist": 0.0}
JUMP = 2


def run(mp, n_eps=60, seed0=1000):
    m = PPO.load(mp, device="cpu")
    K = max(0, (int(m.observation_space.shape[0]) - O.OBS_LEN) // 7)
    env = TempleRunSimEnv(config=CFG, max_episode_steps=4000, reward_mode="distance",
                          action_history_len=K, frame_stack=1)
    jumps = held = rejump = 0
    WIN = 6
    dists = []
    for ep in range(n_eps):
        o, i = env.reset(seed=seed0 + ep); hist = []
        since_jump = 99

        def obsv():
            base = O.state_to_obs(env._snapshot())
            if K:
                h = [int(a) for a in hist[-K:]]; h = [6] * (K - len(h)) + h
                oh = np.zeros(K * 7, dtype=np.float32)
                for j, a in enumerate(h):
                    oh[j * 7 + a] = 1.0
                return np.concatenate([base, oh])
            return base
        obs = obsv(); done = False; steps = 0; last = 6
        while not done and steps < 4000:
            a, _ = m.predict(obs, deterministic=True); a = int(a)
            if a == JUMP:
                jumps += 1
                if last == JUMP:
                    held += 1
                if since_jump < WIN:          # windowed re-jump (un-gameable by gaps)
                    rejump += 1
            since_jump = 0 if a == JUMP else min(99, since_jump + 1)
            hist.append(a); last = a
            o, r, t, tr, info = env.step(a); obs = obsv()
            done = t or tr; steps += 1
        dists.append(float(env._prev_state.get("distance", 0.0)))
    env.close()
    return jumps, held, rejump, dists


for mp in sys.argv[1:] or ["runs/ppo_v59_crevasse_1503232_steps.zip"]:
    jumps, held, rejump, dists = run(mp)
    held_pct = 100 * held / jumps if jumps else 0
    rej_pct = 100 * rejump / jumps if jumps else 0
    print(f"{mp.split('/')[-1]:<40} rejump(win6)%={rej_pct:5.1f}%  held%={held_pct:5.1f}%  "
          f"median_dist={S.median(dists):.0f}")
