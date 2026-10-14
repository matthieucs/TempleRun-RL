#!/usr/bin/env python3
"""Smoke test: slidegap_prob=1.0 -> every slide obstacle is followed by gap_cells clear + a hole,
in BOTH sims. Prints the first few slide->hole layouts found in each."""
import sys
sys.path.insert(0, "src")
import numpy as np

CFG = {"slidegap_prob": 1.0, "slidegap_after_distance": 0.0, "slides_after_distance": 0.0,
       "obstacle_prob": 0.8}

# ---- CPU ----
from templerun_rl.sim_env import TempleRunSimEnv, OBST_TREE_SLIDE, OBST_FLAME_TOWER, OBST_HOLE
env = TempleRunSimEnv(config=CFG)
env.reset(seed=7)
# force-generate cells and inspect the track buffer
for _ in range(400):
    env._track.append(env._next_cell_procedural())
tr = env._track
found = 0
for i, c in enumerate(tr):
    if c.get("obstacles") in (OBST_TREE_SLIDE, OBST_FLAME_TOWER) and i + 3 < len(tr):
        nxt = [(tr[i+k].get("type"), tr[i+k].get("obstacles"), tr[i+k].get("wide_gap")) for k in (1, 2, 3)]
        tag = "SLIDEGAP" if tr[i+3].get("obstacles") == OBST_HOLE else "plain"
        print(f"CPU slide@{i}: next3={nxt}  -> {tag}")
        found += 1
        if found >= 6:
            break

# ---- torch ----
import torch
from templerun_rl.sim_env_torch import TorchTempleRunVecEnv
te = TorchTempleRunVecEnv(num_envs=4, device="cpu", config=CFG)
te.reset()
for _ in range(300):
    acts = torch.full((4,), 6, dtype=torch.int64)
    te.step(acts)
# scan env 0's track ring for slide->x->x->hole
tt, to, tw = te.t_type, te.t_obst, te.t_wide
n_slots = tt.shape[1]
hits = 0
for b in range(4):
    for s in range(n_slots - 3):
        if int(to[b, s]) in (5, 6):
            trip = [(int(tt[b, s+k]), int(to[b, s+k]), bool(tw[b, s+k])) for k in (1, 2, 3)]
            tag = "SLIDEGAP" if int(to[b, s+3]) == 1 else "plain"
            print(f"TORCH env{b} slide@slot{s}: next3={trip} -> {tag}")
            hits += 1
            if hits >= 8:
                break
    if hits >= 8:
        break
print("smoke OK")
