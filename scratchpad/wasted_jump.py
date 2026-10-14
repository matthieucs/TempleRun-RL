#!/usr/bin/env python3
"""WAVE 15: THE WASTED JUMP (the user's 'useless jump' / 'jumps randomly after a turn').

spurious_jump.py asked "was there ANY must-jump obstacle within ~3 cells at takeoff?" — too narrow.
The behaviour the user describes is a jump taken when an obstacle IS visible but still too FAR: the
arc lands short of it, the jump accomplishes nothing, and a SECOND jump is now required. Worse, the
runner spends ~0.77s airborne (unable to act) and the landing frame is action-locked.

WASTED JUMP = a takeoff whose landing (takeoff_pos + span(v)) falls BEFORE the next must-jump cell,
i.e. it did not clear anything. Split by context (<=3 cells after a junction / after a crevasse /
on high ground / plain) and scored for lethality (death within 3s).

Reference = the human: they should waste far fewer, and the ones they waste should be harmless.

Usage: python scratchpad/wasted_jump.py [capture globs...]
"""
from __future__ import annotations
import glob, sys
from collections import defaultdict
from hole_landing_forensic import load_eps, cell_seq

MUST_JUMP = {1, 3, 4, 7}
CREVASSE = 7
SPAN_A, SPAN_B = 0.0204, -0.94       # device jump-span fit (cells)


def analyse(paths, tag):
    tot = defaultdict(int)
    died = defaultdict(int)
    km = 0.0
    for path in paths:
        hz = 30.0 if "human" in path else 10.0
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            km += max(float(f.get("distance") or 0) for f in ep) / 1000.0
            obst = {}
            # is_turn is only presented across the EXECUTABLE window (turn_exec_early..late), so it is
            # already False on the LAST frame of a junction cell — reading it there found 3 junctions
            # in 4590 cells (~0.1%) instead of ~4%. Mark a cell a junction if ANY frame on it says so.
            turn_cell = {}
            for i, fr in enumerate(ep):
                obst.setdefault(cells[i], int(fr.get("obstacle") or 0))
                if fr.get("is_turn"):
                    turn_cell[cells[i]] = True
            mj = sorted(c for c, o in obst.items() if o in MUST_JUMP)
            since_turn = since_crev = 99
            for i, fr in enumerate(ep):
                if i and cells[i] != cells[i - 1]:
                    since_turn = 0 if turn_cell.get(cells[i - 1]) else since_turn + 1
                    since_crev = 0 if int(ep[i - 1].get("obstacle") or 0) == CREVASSE else since_crev + 1
                if not (bool(fr.get("is_jumping")) and not (i and bool(ep[i - 1].get("is_jumping")))):
                    continue
                src = ep[i - 1] if i else fr
                tk = cells[max(0, i - 1)] + float(src.get("cell_progress") or 0.0)
                v = float(src.get("velocity") or 0.0)
                span = SPAN_A * v + SPAN_B
                nxt = next((c for c in mj if c >= cells[i]), None)
                tot["takeoffs"] += 1
                if nxt is None:
                    continue
                # wasted = the arc lands strictly before the obstacle cell -> cleared nothing
                if tk + span >= nxt:
                    continue
                tot["wasted"] += 1
                ctx = ("post_turn" if since_turn <= 3 else
                       "post_crevasse" if since_crev <= 3 else
                       "high_ground" if int(src.get("path_level") or 2) >= 3 else "plain")
                tot["w_" + ctx] += 1
                for j in range(i, min(len(ep), i + int(3 * hz))):
                    if ep[j].get("is_dead"):
                        died[ctx] += 1
                        died["any"] += 1
                        break
    print(f"\n=== {tag} ===")
    print(f"  {tot['takeoffs']} takeoffs / {km:.1f} km | WASTED (arc lands short of the next must-jump): "
          f"{tot['wasted']} = {100*tot['wasted']/max(1,tot['takeoffs']):.1f}% of takeoffs, "
          f"{tot['wasted']/max(1e-9,km):.1f} per km")
    for ctx in ("post_turn", "post_crevasse", "high_ground", "plain"):
        n = tot["w_" + ctx]
        if n:
            print(f"    {ctx:14} n={n:4} ({100*n/max(1,tot['wasted']):4.1f}%)  "
                  f"died<=3s: {died[ctx]:3} ({100*died[ctx]/n:4.1f}%)")
    print(f"    lethality of wasted jumps overall: {died['any']}/{tot['wasted']} = "
          f"{100*died['any']/max(1,tot['wasted']):.1f}%")


if __name__ == "__main__":
    args = sys.argv[1:]
    sets = ([(" ".join(args), args)] if args else
            [("v79 device", ["captures/w14_v79_device.jsonl"]),
             ("v78 device", ["captures/w14_v78_holes.jsonl", "captures/ab_v78_r1.jsonl"]),
             ("v74 device", ["captures/ab_v74_r*.jsonl"]),
             ("v70 device", ["captures/ab_v70_r*.jsonl"]),
             ("HUMAN", ["captures/human_202607*.jsonl"])])
    for tag, gl in sets:
        analyse([p for g in gl for p in sorted(glob.glob(g))], tag)
