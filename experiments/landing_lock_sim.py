#!/usr/bin/env python3
"""Sim-side landing-lock measurement (mirror of landing_lock_forensic): roll v70 in the CPU sim
(device-true cfg) and compute the post-landing JUMP fire-rate table. Device measured: delay0 25%,
delay1 4%, delay2 0%. If the sim fires ~100%, the instant re-jump is a FALSE AFFORDANCE in-sim."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, numpy as np  # noqa: F401
from stable_baselines3 import PPO
from collections import defaultdict

CFG = {"jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True,
       "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1,
       "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25,
       "wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
       "crevasse_graze_worldy": True, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2}

model = PPO.load("runs/ppo_v70_obs64_pturn.zip", device="cpu")  # v70-exact on the current obs64 pipeline
env = gym.make("TempleRunSim-v0", config=CFG, action_history_len=4, frame_stack=1,
               turn_approach_buffer=True, turn_linger_frames=3).unwrapped
# Mirror of the DEVICE measurement exactly: build a frame log (state AFTER each step + the action
# that produced it), then run the same table logic as landing_lock_forensic.py.
rows = []
for epi in range(40):
    obs, info = env.reset(seed=1000 + epi)
    done = False
    step = 0
    while not done and step < 3000:
        act = int(model.predict(obs, deterministic=True)[0])
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        s = info["raw_state"]
        rows.append({"ep": epi, "action": act, "is_jumping": bool(s.get("is_jumping")),
                     "is_sliding": bool(s.get("is_sliding")), "is_dead": bool(s.get("is_dead"))})
        step += 1

stats = defaultdict(lambda: [0, 0])
last_land = None
for i in range(1, len(rows) - 2):
    prev, fr = rows[i - 1], rows[i]
    if prev["ep"] != fr["ep"]:
        last_land = None
        continue
    if prev["is_jumping"] and not fr["is_jumping"] and not fr["is_dead"]:
        last_land = i
    if last_land is None:
        continue
    delay = i - last_land
    if delay > 6:
        continue
    if fr["action"] == 2 and not fr["is_jumping"] and not fr["is_sliding"]:
        fired = rows[i + 1]["is_jumping"] or rows[i + 2]["is_jumping"]
        stats[delay][1] += 1
        stats[delay][0] += 1 if fired else 0
print("SIM post-landing JUMP fire rate by delay (frames @10Hz since touchdown):")
for d in sorted(stats):
    f, n = stats[d]
    print(f"  delay {d}: fired {f}/{n} = {100*f/max(n,1):.0f}%")
