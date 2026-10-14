#!/usr/bin/env python3
"""WAVE 15: what lies in the first cells AFTER a junction — device vs sim generation.

Measured (experiments/wasted_jump.py): 61-70% of every policy's WASTED jumps happen within 3 cells of
a junction, and so do the human's — the post-turn hedge-jump is universal (obs is degraded through
junctions: the hook's neighbour walk breaks, next/next2_path_level stream -1). The difference is the
CONSEQUENCE: the human's post-turn hedge is 0/65 lethal, the policies' is ~7% lethal.

Hypothesis: on DEVICE an obstacle can sit 2-4 cells past a junction — inside the hedge-jump's landing
zone — while in the SIM `min_clear_after_turn`(2) + `min_space_between_obstacles`(6) keep that zone
empty, so the hedge-jump is structurally FREE in-sim and the policy never learns to hold it.
`turnobstacle_prob` only ever fills cell +1, which a 2.5-3.3-cell arc flies straight over.

Prints P(must-jump obstacle) at each offset 1..6 after a junction, device vs sim generation.

Usage: python experiments/postturn_obstacles.py [capture globs...]
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import glob, sys
from collections import defaultdict
from hole_landing_forensic import load_eps, cell_seq

MUST_JUMP = {1, 3, 4, 7}
MAXOFF = 6


def device(paths):
    hit, tot = defaultdict(int), defaultdict(int)
    for path in paths:
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            obst, turn = {}, {}
            for i, fr in enumerate(ep):
                obst.setdefault(cells[i], int(fr.get("obstacle") or 0))
                if fr.get("is_turn"):
                    turn[cells[i]] = True
            last = max(obst) if obst else 0
            for j in sorted(turn):
                for off in range(1, MAXOFF + 1):
                    if j + off > last:
                        break
                    tot[off] += 1
                    hit[off] += int(obst.get(j + off, 0) in MUST_JUMP)
    return hit, tot


def sim(cfg, n=60000):
    sys.path.insert(0, "src")
    from templerun_rl.sim_env import TempleRunSimEnv
    env = TempleRunSimEnv(track_source="procedural", config=cfg, seed=5)
    env.reset(seed=5)
    env._distance_real = 4000.0
    cells = [env._next_cell_procedural() for _ in range(n)]
    hit, tot = defaultdict(int), defaultdict(int)
    for i, c in enumerate(cells):
        if not c.get("is_turn"):
            continue
        for off in range(1, MAXOFF + 1):
            if i + off >= len(cells):
                break
            tot[off] += 1
            hit[off] += int(int(cells[i + off].get("obstacles", 0)) in MUST_JUMP)
    return hit, tot


def show(tag, hit, tot):
    print(f"  {tag:32} " + "  ".join(
        f"+{o}:{100*hit[o]/max(1,tot[o]):5.1f}%" for o in range(1, MAXOFF + 1)))


if __name__ == "__main__":
    gl = sys.argv[1:] or ["captures/w14_v78_holes.jsonl", "captures/ab_v78_r1.jsonl",
                          "captures/ab_v74_r*.jsonl", "captures/ab_v70_r*.jsonl",
                          "captures/w14_v79_device.jsonl"]
    paths = [p for g in gl for p in sorted(glob.glob(g))]
    print("P(must-jump obstacle) at cell offset AFTER a junction:")
    show("DEVICE (all policy captures)", *device(paths))
    show("SIM v78 recipe (turnobst 0.08)", *sim({"turnobstacle_prob": 0.08, "zone_after_distance": 0.0}))
    show("SIM turnobst 0 (bare)", *sim({"turnobstacle_prob": 0.0, "zone_after_distance": 0.0}))
    show("SIM + tight_pair 0.55/2", *sim({"turnobstacle_prob": 0.08, "zone_after_distance": 0.0,
                                          "tight_pair_prob": 0.55, "tight_pair_min_cells": 2}))
