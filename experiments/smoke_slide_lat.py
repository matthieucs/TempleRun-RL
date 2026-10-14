#!/usr/bin/env python3
"""Smoke: slide_send_lat_frames should KILL a policy that slides AT flame entry (react-cp) but SPARE
one that pre-slides on the approach cell. A scripted 'react' policy (slide only when must_slide is
the CURRENT cell) should die more with lat=1 than lat=0; a 'pre-slide' policy (slide when
next_must_slide) should be unaffected."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, numpy as np  # noqa: F401

def run(pre_slide, slat, seeds=60):
    cfg = {"slide_send_lat_frames": slat, "slides_after_distance": 0.0, "obstacle_prob": 0.6,
           "obstacle_type_weights": {1: 1, 5: 3, 6: 3}}  # slide-heavy
    env = gym.make("TempleRunSim-v0", config=cfg).unwrapped
    dists = []
    for s in range(seeds):
        obs, info = env.reset(seed=s)
        done = False
        while not done:
            st = info["raw_state"]
            if pre_slide:
                act = 3 if st.get("next_must_slide") else 6
            else:
                act = 3 if st.get("must_slide") else 6
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        dists.append(float(info["raw_state"].get("distance") or 0))
    return int(np.median(dists))

for pre in (False, True):
    tag = "PRE-SLIDE(approach)" if pre else "REACT(on-cell)"
    row = "  ".join(f"lat{s}={run(pre, s):5}" for s in (0, 1, 2, 3))
    print(f"{tag:22} {row}")
