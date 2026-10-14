#!/usr/bin/env python3
"""Why does the sim refuse GROUNDED delay-1 post-landing jumps (0/52)? Log the guard state at every
post-landing JUMP attempt: _jump_t, _slide_t, _pending_jump_frames, strafing gap, wide_gap+cp, drop."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, numpy as np  # noqa: F401
from stable_baselines3 import PPO
from collections import Counter

CFG = {"jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True,
       "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1,
       "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25,
       "wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
       "crevasse_graze_worldy": True, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2}

model = PPO.load("runs/ppo_v70_obs64_pturn.zip", device="cpu")
env = gym.make("TempleRunSim-v0", config=CFG, action_history_len=4, frame_stack=1,
               turn_approach_buffer=True, turn_linger_frames=3).unwrapped
reasons = Counter()
outcomes = Counter()
for epi in range(30):
    obs, info = env.reset(seed=2000 + epi)
    done = False
    step = 0
    frames_since_land = 99
    prev_jumping = False
    while not done and step < 3000:
        act = int(model.predict(obs, deterministic=True)[0])
        # PRE-STEP guard state (what _apply_action will see)
        if act == 2 and frames_since_land <= 3:
            jt, st_, pend = env._jump_t, env._slide_t, getattr(env, "_pending_jump_frames", 0)
            gap = abs(env._lat_target - env._lane)
            cell = env._track[env._cell_index]
            cp = env._cell_progress()
            if jt > 0:
                reasons["airborne(_jump_t>0)"] += 1
            elif st_ > 0:
                reasons["sliding"] += 1
            elif pend > 0:
                reasons["pending_jump_already_queued"] += 1
            elif gap > 1.8:
                reasons["strafing_gap>1.8 (60% refuse)"] += 1
            elif cell.get("wide_gap") and cp > 0.2:
                reasons["gap_lip"] += 1
            else:
                reasons["no_guard (drop 10% only)"] += 1
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        s = info["raw_state"]
        jumping = bool(s.get("is_jumping"))
        if prev_jumping and not jumping and not s.get("is_dead"):
            frames_since_land = 0
        else:
            frames_since_land += 1
        prev_jumping = jumping
        step += 1
print("post-landing (<=3 frames) JUMP attempt guard states:")
for k, v in reasons.most_common():
    print(f"  {k}: {v}")
