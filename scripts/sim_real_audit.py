#!/usr/bin/env python3
"""SIM-TO-REAL AUDIT — the standing gate for generation fidelity (WAVE 15, user-directed).

WHY THIS EXISTS. For ~20 models the loop was: pick a mechanic, improve its DEATH MODEL, retrain,
watch the phone disagree with the offline number. The 2026-07-26 finding is that the binding gap is
usually not the death model at all -- it is that the SITUATION DOES NOT EXIST IN THE SIM. Two checks,
two zeros: the "wasted jump" (0 occurrences in 74 km of sim rollout vs 6.5-7.3% of device takeoffs)
and obstacle->crevasse compounds (0 in the sim vs 35 on device). A perfect death model for a
situation that is never generated is worth nothing.

The gap has THREE parts and this script measures all three from device captures vs sim generation:

  1. COVERAGE   — for every hazard pair (A -> B) at every spacing, does the sim produce it at all?
                  Any row where device > 0 and sim == 0 is a hole in the trained world.
  2. DENSITY    — hazard mix and spacing distribution (the sim's hard min_space floor of 6 removes
                  19% of the device's real spacings).
  3. DIFFICULTY — reported separately by eval: the champion's SIM median must land on its DEVICE
                  median. At the time of writing: 6786 vs 4425 = the sim is ~1.5x too easy.

Usage:
    python scripts/sim_real_audit.py                        # default captures vs the training cfg
    python scripts/sim_real_audit.py --captures "captures/w15_*.jsonl" --sim-config '{"k":v}'
"""
from __future__ import annotations
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scratchpad"))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

NAME = {1: "Hole", 2: "Narrow", 3: "TreeStumble", 4: "TreeJump", 5: "TreeSlide",
        6: "FlameTower", 7: "Crevasse"}
HAZARDS = set(NAME)
ARC = 3                      # a jump spans ~2.5-3.3 cells
TRAIN_CFG = {"turnobstacle_prob": 0.08, "jumpslide_prob": 0.12, "jumpnarrow_prob": 0.12,
             "jumpgap_prob": 0.15, "jumpgap_chain_max": 3, "slidegap_prob": 0.12,
             "max_back_to_back_obstacles": 6, "zone_after_distance": 0.0}


def structures(seq):
    """Collapse runs of one obstacle type into a single structure -> [(first_cell, type), ...]."""
    out, pc, po = [], None, None
    for c in sorted(seq):
        o = seq[c]
        if o in HAZARDS and not (po == o and pc is not None and c - pc == 1):
            out.append((c, o))
        pc, po = c, o
    return out


def pairs(seq):
    st = structures(seq)
    return [(a[1], b[1], b[0] - a[0]) for a, b in zip(st, st[1:])]


def from_device(globs):
    ps, cells_tot = [], 0
    for g in globs:
        for path in sorted(glob.glob(g)):
            for _, ep in load_eps(path):
                cl = cell_seq(ep)
                seq = {}
                for i, fr in enumerate(ep):
                    seq.setdefault(cl[i], int(fr.get("obstacle") or 0))
                ps += pairs(seq)
                cells_tot += len(seq)
    return ps, cells_tot


def from_sim(cfg, n=80000):
    from templerun_rl.sim_env import TempleRunSimEnv
    env = TempleRunSimEnv(track_source="procedural", config=cfg, seed=11)
    env.reset(seed=11)
    # ADVANCE DISTANCE AS CELLS ARE GENERATED. Pinning it (the old behaviour) made the whole sample
    # come from ONE ramp state, which is not comparable to device captures pooled over a full run --
    # and made the ramped sim look far too tight. Episodes also restart, so cycle the distance the way
    # real runs do: many short ones, fewer deep ones.
    cell_len = float(env.cfg.get("cell_length", 60.0))
    cells, d, cap = [], 0.0, 0.0
    import random as _r
    rng = _r.Random(11)
    for i in range(n):
        if d > cap:                      # start a new "episode" with a realistic depth
            cap = rng.choice([600, 1200, 2000, 3200, 4500, 7000, 11000])
            d = 0.0
        env._distance_real = d
        cells.append(env._next_cell_procedural())
        d += cell_len
    seq = {i: int(c.get("obstacles", 0)) for i, c in enumerate(cells)}
    return pairs(seq), n


def pct(ps, pred):
    return 100.0 * sum(1 for p in ps if pred(p)) / max(1, len(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="*", default=[
        "captures/w14_v78_holes.jsonl", "captures/ab_v78_r1.jsonl", "captures/ab_v74_r*.jsonl",
        "captures/ab_v70_r*.jsonl", "captures/w15_v8*_device.jsonl"])
    ap.add_argument("--sim-config", default="")
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()
    cfg = dict(TRAIN_CFG)
    if a.sim_config:
        cfg.update(json.loads(a.sim_config))

    dev, dcells = from_device(a.captures)
    sim, scells = from_sim(cfg)
    print(f"device: {len(dev)} hazard pairs over {dcells} cells | "
          f"sim: {len(sim)} pairs over {scells} cells\n")

    print("2. SPACING DISTRIBUTION (share of consecutive hazard pairs at each gap, in cells)")
    print("   gap:      " + "".join(f"{g:>7}" for g in range(1, 9)))
    for tag, ps in (("device", dev), ("sim   ", sim)):
        print(f"   {tag}:   " + "".join(f"{pct(ps, lambda p, g=g: p[2] == g):6.1f}%" for g in range(1, 9)))
    print(f"   INSIDE the jump arc (<={ARC}): device {pct(dev, lambda p: p[2] <= ARC):.1f}%  "
          f"sim {pct(sim, lambda p: p[2] <= ARC):.1f}%")

    print("\n1. COVERAGE — hazard pairs INSIDE the jump arc. 'MISSING' = the sim cannot produce it.")
    dcount, scount = defaultdict(int), defaultdict(int)
    for A, B, g in dev:
        if g <= ARC:
            dcount[(A, B)] += 1
    for A, B, g in sim:
        if g <= ARC:
            scount[(A, B)] += 1
    rows = sorted(set(dcount) | set(scount), key=lambda k: -dcount[k])
    print(f"   {'pair':30} {'device':>8} {'sim':>8}   verdict")
    missing = 0
    for k in rows[:a.top]:
        d, s = dcount[k], scount[k]
        verdict = "** MISSING IN SIM **" if d > 0 and s == 0 else ("sim-only" if d == 0 else "covered")
        missing += 1 if (d > 0 and s == 0) else 0
        print(f"   {NAME.get(k[0],k[0]):13}-> {NAME.get(k[1],k[1]):13} {d:8} {s:8}   {verdict}")
    tot_missing = sum(1 for k in set(dcount) if dcount[k] > 0 and scount[k] == 0)
    dev_missing = sum(dcount[k] for k in set(dcount) if scount[k] == 0)
    print(f"\n   {tot_missing} device pair-types have ZERO sim coverage, "
          f"accounting for {dev_missing} device occurrences "
          f"({100*dev_missing/max(1,len(dev)):.2f}% of all pairs)")
    print("\n3. DIFFICULTY — run eval_ckpt_torch on the champion and compare its sim median to its\n"
          "   measured DEVICE median (v78: sim 6786 vs device 4425 = sim ~1.5x too easy).")


if __name__ == "__main__":
    main()
