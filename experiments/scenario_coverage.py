#!/usr/bin/env python3
"""WAVE 15 (USER-directed): SCENARIO COVERAGE AUDIT — device vs sim generation.

USER: "The policy died when faced a succession of obstacles. The obstacles were made of one that
forced the character to jump followed by a crevasse. In that case, the policy should have jumped
earlier ... Here it jumped as usual over the first obstacle and landed directly into the crevasse."

Rather than chase one mechanic at a time, enumerate the COMPOUND situations that actually occur on
device and ask, for each: can the sim generate it AT ALL? A situation the sim never produces cannot
be trained, no matter how good the death model for it is (this is the lesson of the wasted jump: 0
occurrences in 74 km of sim vs 6.5-7.3% of takeoffs on device).

Pairs are keyed by the FIRST cell of each structure, so a 2-cell crevasse counts once. Spacing is in
cells; the jump arc spans ~2.5-3.3 cells, so a gap of 1-3 means the second hazard lands INSIDE the
arc of a jump taken for the first -- the situation the user describes.

Usage: python experiments/scenario_coverage.py [capture globs...]
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import glob, sys
from collections import defaultdict
from hole_landing_forensic import load_eps, cell_seq

HOLE, STUMBLE, TREEJUMP, CREV = 1, 3, 4, 7
NAME = {HOLE: "Hole", STUMBLE: "TreeStumble", TREEJUMP: "TreeJump", CREV: "Crevasse"}
MUST_JUMP = {HOLE, STUMBLE, TREEJUMP, CREV}
ARC = 3          # cells: a hazard this close sits inside the jump arc


def structures(seq):
    """Collapse runs of the same obstacle into one structure: [(first_cell, obst), ...]."""
    out, prev_c, prev_o = [], None, None
    for c in sorted(seq):
        o = seq[c]
        if o not in MUST_JUMP:
            prev_c, prev_o = c, o
            continue
        if prev_o == o and prev_c is not None and c - prev_c == 1:
            prev_c, prev_o = c, o
            continue                      # continuation of the same structure
        out.append((c, o))
        prev_c, prev_o = c, o
    return out


def pairs_from(seq):
    st = structures(seq)
    return [(a_o, b_o, b_c - a_c) for (a_c, a_o), (b_c, b_o) in zip(st, st[1:])]


def device(paths):
    ps = []
    for path in paths:
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            seq = {}
            for i, fr in enumerate(ep):
                seq.setdefault(cells[i], int(fr.get("obstacle") or 0))
            ps += pairs_from(seq)
    return ps


def sim(cfg, n=60000):
    sys.path.insert(0, "src")
    from templerun_rl.sim_env import TempleRunSimEnv
    env = TempleRunSimEnv(track_source="procedural", config=cfg, seed=7)
    env.reset(seed=7)
    env._distance_real = 4000.0
    cells = [env._next_cell_procedural() for _ in range(n)]
    seq = {i: int(c.get("obstacles", 0)) for i, c in enumerate(cells)}
    return pairs_from(seq)


def report(tag, ps):
    tot = len(ps)
    inside = [p for p in ps if p[2] <= ARC]
    print(f"\n=== {tag} ===  {tot} consecutive must-jump pairs; "
          f"{len(inside)} ({100*len(inside)/max(1,tot):.1f}%) sit INSIDE the jump arc (<={ARC} cells)")
    by = defaultdict(int)
    for a, b, g in inside:
        by[(NAME.get(a, a), NAME.get(b, b))] += 1
    if not by:
        print("    (none — the sim cannot produce ANY of these compounds)")
    for k, n in sorted(by.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {k[0]:12} -> {k[1]:12}  n={n}")
    # the user's exact case: anything -> Crevasse inside the arc
    tocrev = [p for p in inside if p[1] == CREV]
    print(f"    ** anything -> CREVASSE inside the arc: {len(tocrev)} "
          f"({100*len(tocrev)/max(1,tot):.2f}% of all pairs) **")


if __name__ == "__main__":
    gl = sys.argv[1:] or ["captures/w14_v78_holes.jsonl", "captures/ab_v78_r1.jsonl",
                          "captures/ab_v74_r*.jsonl", "captures/ab_v70_r*.jsonl",
                          "captures/w15_v83_device.jsonl", "captures/human_202607*.jsonl"]
    report("DEVICE (all captures)", device([p for g in gl for p in sorted(glob.glob(g))]))
    report("SIM v78/v87 training generation",
           sim({"turnobstacle_prob": 0.08, "jumpslide_prob": 0.12, "jumpnarrow_prob": 0.12,
                "jumpgap_prob": 0.15, "jumpgap_chain_max": 3, "slidegap_prob": 0.12,
                "max_back_to_back_obstacles": 6, "zone_after_distance": 0.0}))
    report("SIM + tight_pair 0.10/2 (the dropped knob)",
           sim({"turnobstacle_prob": 0.08, "jumpslide_prob": 0.12, "jumpnarrow_prob": 0.12,
                "jumpgap_prob": 0.15, "jumpgap_chain_max": 3, "slidegap_prob": 0.12,
                "max_back_to_back_obstacles": 6, "zone_after_distance": 0.0,
                "tight_pair_prob": 0.10, "tight_pair_min_cells": 2}))
