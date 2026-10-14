#!/usr/bin/env python3
"""Reconcile the TWO device post-junction measurements that disagree.

    wave-15 postturn_obstacles.py :  +2 = 0.7%   +3 = 21.7%
    2026-07-27 chain-era pass     :  +2 = 9.5%   +3 = 36.5%

Same device, same question, 13x apart at +2. One of them drove a real config change
(min_clear_after_turn 2 -> 3), so the disagreement has to be resolved BEFORE either is trusted --
the user's v107 death was "turn, then a crevasse, then another crevasse", which is precisely the
region in dispute.

THREE methodological differences were found by reading both implementations:

  (1) PER-CELL OBSTACLE. wave-15 uses `obst.setdefault(cell, ...)` = the FIRST frame on the cell
      wins. Device flags are presented across a WINDOW (the documented `is_turn` trap: it is already
      False on the last frame of a junction cell), so a cell whose obstacle field is 0 on its first
      sampled frame is recorded as EMPTY. Under-counts.
  (2) CREVASSES. wave-15 tests `obstacle in {1,3,4,7}` only. A crevasse is a PATH TYPE (33-36), and
      on device its `obstacle` field is not reliably 7 -- so wave-15 may count NONE of them.
  (3) SLIDES. wave-15 counts MUST_JUMP only; the chain-era pass counts MUST_JUMP + MUST_SLIDE.

This runs all four definitions over the SAME captures, adding one difference at a time, so each
one's contribution is isolated. Then it repeats on BOTH capture eras to separate METHOD from DATA.

    python scratchpad/postturn_reconcile.py
"""
from __future__ import annotations
import glob, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

MUST_JUMP = {1, 3, 4, 7}
MUST_SLIDE = {5, 6}
CREV_PT = {33, 34, 35, 36}
MAXOFF = 6


def measure(paths, *, first_frame_wins, crev_from_pathtype, include_slides):
    hit, tot = defaultdict(int), defaultdict(int)
    for path in paths:
        for _, ep in load_eps(path):
            cells = cell_seq(ep)
            obst, turn = {}, {}
            for i, fr in enumerate(ep):
                c = cells[i]
                o = int(fr.get("obstacle") or 0)
                if crev_from_pathtype and fr.get("path_type") in CREV_PT:
                    o = 7
                if first_frame_wins:
                    obst.setdefault(c, o)
                else:
                    if c not in obst or (not obst[c] and o):
                        obst[c] = o
                if fr.get("is_turn"):
                    turn[c] = True
            last = max(obst) if obst else 0
            want = MUST_JUMP | (MUST_SLIDE if include_slides else set())
            for j in sorted(turn):
                for off in range(1, MAXOFF + 1):
                    if j + off > last:
                        break
                    tot[off] += 1
                    hit[off] += int(obst.get(j + off, 0) in want)
    return hit, tot


def show(tag, hit, tot):
    print(f"  {tag:44} " + "  ".join(
        f"+{o}:{100*hit[o]/max(1,tot[o]):5.1f}%" for o in range(1, MAXOFF + 1))
        + f"   n_junc={tot[1]}")


def run(label, paths):
    if not paths:
        print(f"\n=== {label}: no captures found ===")
        return
    print(f"\n=== {label} ===  ({len(paths)} files)")
    show("A  wave-15 exact (first-frame, no crev, jump)",
         *measure(paths, first_frame_wins=True, crev_from_pathtype=False, include_slides=False))
    show("B  A + crevasse from path_type",
         *measure(paths, first_frame_wins=True, crev_from_pathtype=True, include_slides=False))
    show("C  B + include slides",
         *measure(paths, first_frame_wins=True, crev_from_pathtype=True, include_slides=True))
    show("D  C + any-frame obstacle  (= chain-era)",
         *measure(paths, first_frame_wins=False, crev_from_pathtype=True, include_slides=True))


if __name__ == "__main__":
    old = [p for g in ["captures/w14_v78_holes.jsonl", "captures/ab_v78_r1.jsonl",
                       "captures/ab_v74_r*.jsonl", "captures/ab_v70_r*.jsonl",
                       "captures/w14_v79_device.jsonl"] for p in sorted(glob.glob(g))]
    new = [p for g in ["captures/human_20260708_*.jsonl", "captures/w17_v104_device.jsonl",
                       "captures/w17_v107_device.jsonl"] for p in sorted(glob.glob(g))]
    print("P(hazard) at cell offset AFTER a junction -- one methodological change at a time")
    run("WAVE-15 CAPTURE ERA (what postturn_obstacles.py read)", old)
    run("CHAIN ERA (what the 2026-07-27 pass read)", new)
