#!/usr/bin/env python3
"""WAVE 15 (USER-directed 2026-07-25): THE RANDOM JUMP.

USER, watching live: "the policy tends to jump randomly after a turn. No reason involved... I saw a
useless jump after a crevasse on high ground... a random jump is often the reason of death. If those
erratic moments are mastered, the reliable 10,000m is in the pocket."

A SPURIOUS jump = a takeoff with NO must-jump obstacle anywhere inside the jump arc (~2.5-3.3 cells,
i.e. must_jump / next_must_jump / next2_must_jump / next3_must_jump all false at takeoff). It costs
~0.77s of airtime during which NO other action can be taken and the landing frame is action-locked.

Measures, per capture set: spurious-jump RATE (per 1000m and as a share of all takeoffs), split by
CONTEXT (<=3 cells after a junction / after a crevasse / on high ground / plain), and their
LETHALITY (died within 3s of the spurious takeoff). Human captures are the reference: the human
should be far cleaner.

Usage: python scratchpad/spurious_jump.py <capture globs...>
"""
from __future__ import annotations
import glob, sys
from collections import defaultdict
from hole_landing_forensic import load_eps, cell_seq

CREVASSE = 7


def analyse(paths, tag, hz=10.0):
    tot = defaultdict(int)
    deaths = defaultdict(int)
    dist_m = 0.0
    for path in paths:
        _hz = 30.0 if "human" in path else hz
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            dist_m += max(float(f.get("distance") or 0) for f in ep)
            since_turn = 99
            since_crev = 99
            for i, fr in enumerate(ep):
                if i and cells[i] != cells[i - 1]:
                    since_turn = 0 if ep[i - 1].get("is_turn") else since_turn + 1
                    since_crev = 0 if int(ep[i - 1].get("obstacle") or 0) == CREVASSE else since_crev + 1
                jmp = bool(fr.get("is_jumping"))
                pj = bool(ep[i - 1].get("is_jumping")) if i else False
                if not (jmp and not pj):
                    continue                       # liftoff frames only
                src = ep[i - 1] if i else fr
                need = any(bool(src.get(k)) for k in
                           ("must_jump", "next_must_jump", "next2_must_jump", "next3_must_jump"))
                tot["takeoffs"] += 1
                if need:
                    continue
                tot["spurious"] += 1
                ctx = ("post_turn" if since_turn <= 3 else
                       "post_crevasse" if since_crev <= 3 else
                       "high_ground" if int(src.get("path_level") or 2) >= 3 else "plain")
                tot["sp_" + ctx] += 1
                # lethality: died within 3s of this spurious takeoff
                for j in range(i, min(len(ep), i + int(3 * _hz))):
                    if ep[j].get("is_dead"):
                        deaths[ctx] += 1
                        deaths["any"] += 1
                        break
    km = dist_m / 1000.0
    print(f"\n=== {tag} ===")
    print(f"  {tot['takeoffs']} takeoffs over {km:.1f} km | SPURIOUS (no must-jump within the arc): "
          f"{tot['spurious']} = {100*tot['spurious']/max(1,tot['takeoffs']):.1f}% of takeoffs, "
          f"{tot['spurious']/max(1e-9,km):.1f} per km")
    for ctx in ("post_turn", "post_crevasse", "high_ground", "plain"):
        n = tot["sp_" + ctx]
        if not n:
            continue
        print(f"    {ctx:14} n={n:4} ({100*n/max(1,tot['spurious']):4.1f}% of spurious)  "
              f"died<=3s: {deaths[ctx]:3} ({100*deaths[ctx]/n:4.1f}%)")
    print(f"    ANY spurious followed by death <=3s: {deaths['any']}"
          f" ({100*deaths['any']/max(1,tot['spurious']):.1f}%)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        analyse([p for g in args for p in sorted(glob.glob(g))], " ".join(args))
    else:
        for tag, gl in (("v79 device", ["captures/w14_v79_device.jsonl"]),
                        ("v78 device", ["captures/w14_v78_holes.jsonl", "captures/ab_v78_r1.jsonl"]),
                        ("v74 device", ["captures/ab_v74_r1.jsonl", "captures/ab_v74_r2.jsonl"]),
                        ("HUMAN", ["captures/human_202607*.jsonl"])):
            analyse([p for g in gl for p in sorted(glob.glob(g))], tag)
