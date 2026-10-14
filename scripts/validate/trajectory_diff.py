#!/usr/bin/env python3
"""MULTI-FRAME sim-vs-device comparator — the cheap fidelity tool replay_diff couldn't be.

replay_diff is ONE-STEP (real state -> action -> compare next), blind to arc/chain/recovery
mechanics because it can't carry the recorded track. This tool RECONSTRUCTS the track from a
capture (each frame's nearest cell; a cell_progress wrap = a new cell entered), then REPLAYS the
recorded action sequence through the sim on that exact track and overlays the trajectories.

For each device DEATH it answers the load-bearing question: does the sim ALSO die there? A device
death the sim SURVIVES is a pinpointed fidelity gap (the mechanic that lets a bad play live in-sim).
It also reports velocity divergence (sim vs device) over the run — where the sim's physics drift.

    python scripts/validate/trajectory_diff.py captures/v46_device.jsonl
    python scripts/validate/trajectory_diff.py captures/v46_device.jsonl --ep 4 --verbose

No phone, no policy, no training — just a capture the sim already knows how to consume.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from templerun_rl.sim_env import TempleRunSimEnv, _MUST_JUMP, _MUST_SLIDE  # noqa: E402


def pt_family(pt):
    pt = pt if isinstance(pt, int) else -1
    if pt < 0:
        return "off"
    if pt <= 2:
        return "straight"
    if pt <= 10:
        return "junction"
    if pt <= 16:
        return "hole"
    if pt <= 30:
        return "narrow"
    return "cap"


def load_eps(path):
    eps = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "cell_progress" in r and "ep" in r:
            eps.setdefault(r["ep"], []).append(r)
    return eps


def reconstruct_track(frames):
    """Ordered list of cells the player crossed = the nearest cell sampled at each NEW cell
    (detected by a cell_progress DROP = a boundary crossing). Each cell dict matches the sim's
    replay element schema. The obstacle enum is inferred from must_jump/must_slide + path_type."""
    cells = []
    prev_cp = -1.0
    for f in frames:
        cp = float(f.get("cell_progress") or 0.0)
        new_cell = (not cells) or cp < prev_cp - 0.15   # a wrap (progress fell) = entered a new cell
        prev_cp = cp
        if not new_cell:
            continue
        pt = int(f.get("path_type") or 1)
        obst = 0
        if f.get("must_jump"):
            obst = 1 if not (11 <= pt <= 16) else 1     # Hole/TreeJump family -> jump
        elif f.get("must_slide"):
            obst = 5
        nl = f.get("narrow_target_lane")
        cell = {"type": pt if pt >= 0 else 1, "level": int(f.get("path_level") or 2),
                "obstacles": obst, "is_turn": bool(f.get("is_turn")),
                "can_turn_left": bool(f.get("can_turn_left")),
                "can_turn_right": bool(f.get("can_turn_right")),
                "wide_gap": bool(f.get("next_wide_gap")) or (11 <= pt <= 16 and pt != 11 and pt != 14),
                "narrow_lane": None}
        cells.append(cell)
    return cells


def replay_episode(frames, cfg):
    """Feed the reconstructed track + recorded actions into the CPU sim (replay mode) anchored on
    the recorded distance. Return per-frame (sim_alive, sim_vel, dev_vel) and the death comparison."""
    track = reconstruct_track(frames)
    if len(track) < 4:
        return None
    env = TempleRunSimEnv(track_source=track, config=cfg, max_episode_steps=len(frames) + 5,
                          reward_mode="distance")
    f0 = frames[0]
    env.reset()
    env.set_state(f0, {"lane": f0.get("lane"), "distance": f0.get("distance"),
                       "velocity": f0.get("velocity")})
    sim_vel, dev_vel, sim_alive = [], [], []
    sim_died_at = None
    for i in range(1, len(frames)):
        a = int(frames[i].get("action") if frames[i].get("action") is not None else 6)
        ext = float(frames[i].get("distance") or 0.0)
        o, r, term, trunc, info = env.step(a, ext_distance=ext)
        st = info["raw_state"]
        sim_vel.append(float(st.get("velocity") or 0))
        dev_vel.append(float(frames[i].get("velocity") or 0))
        sim_alive.append(not term)
        if term and sim_died_at is None:
            sim_died_at = i
            break
    dev_died = frames[-1].get("is_dead")
    return {"n": len(frames), "track_cells": len(track),
            "sim_died_at": sim_died_at, "dev_died": bool(dev_died),
            "dev_death_pt": int(frames[-1].get("path_type") or -1),
            "sim_vel": sim_vel, "dev_vel": dev_vel}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture")
    p.add_argument("--ep", type=int, default=None, help="only this episode")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cfg = {"jump_exec_lat_frames": 0, "jump_exec_lat_jitter": 0, "hook_turn_buffer": True,
           "strafe_exec_lat_frames": 0, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0,
           "stepup_clip_band": 0.25, "domain_rand": False, "turn_approach_buffer": True,
           "turn_linger_frames": 3}
    eps = load_eps(args.capture)
    print(f"{Path(args.capture).name}: {len(eps)} episodes\n")
    print(f"{'ep':>3} {'frames':>6} {'devDeath':>10} {'simDeath':>10} {'verdict':>22} {'velMAE':>7}")
    agree = disagree_survive = 0
    death_fam = Counter()
    for ep in sorted(eps):
        if args.ep is not None and ep != args.ep:
            continue
        res = replay_episode(eps[ep], cfg)
        if res is None:
            continue
        fam = pt_family(res["dev_death_pt"]) if res["dev_died"] else "-"
        n = min(len(res["sim_vel"]), len(res["dev_vel"]))
        vmae = float(np.mean(np.abs(np.array(res["sim_vel"][:n]) - np.array(res["dev_vel"][:n])))) if n else 0.0
        # verdict: did the sim reproduce the device death near the end?
        if res["dev_died"]:
            if res["sim_died_at"] is not None and res["sim_died_at"] >= res["n"] - 12:
                verdict = "MATCH (both die)"
                agree += 1
            elif res["sim_died_at"] is not None:
                verdict = f"sim dies EARLY @{res['sim_died_at']}"
            else:
                verdict = "GAP sim SURVIVES death"
                disagree_survive += 1
                death_fam[fam] += 1
        else:
            verdict = "dev alive"
        dd = f"{fam}@{res['n']}" if res["dev_died"] else "-"
        sd = str(res["sim_died_at"]) if res["sim_died_at"] is not None else "-"
        print(f"{ep:>3} {res['n']:>6} {dd:>10} {sd:>10} {verdict:>22} {vmae:>7.1f}")

    print(f"\nmatched deaths: {agree}  |  GAPS (device died, sim survives): {disagree_survive}")
    if death_fam:
        print("  sim-survives-death by device piece family:",
              dict(death_fam.most_common()))
        print("  -> these are the mechanics the sim is too LENIENT on (device-fatal, sim-safe).")


if __name__ == "__main__":
    main()
