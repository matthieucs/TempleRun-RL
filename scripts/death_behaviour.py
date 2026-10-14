#!/usr/bin/env python3
"""BEHAVIOURAL DEATH BUDGET — not "what killed the run" but "what did the POLICY DO wrong".

USER (2026-07-26): "your table does not contain combinations of them which is a cause and can be an
explanation. When you say hole, is it after a first obstacle or just a hole? Why did the policy die
in the hole? You're missing a behavioural study. I can tell you that one of the main causes of this
death comes from a random jump taken by the policy."

Correct. A per-hazard table ("Hole = 44% of deaths") says WHERE a run ended, not WHY, so it cannot
be acted on. Every death here is instead classified by the RUNNER'S OWN LAST DECISION:

  RANDOM-JUMP TRAP  the runner was AIRBORNE into the fatal cell, and the jump that put it there was
                    taken when NOTHING required it (no must-jump obstacle anywhere in the arc at
                    takeoff). The user's reported cause: a jump for no reason that removes the
                    ability to act, and drops the runner into the next hazard.
  COMPOUND TRAP     airborne into the fatal cell, but the jump WAS justified -- it cleared a real
                    obstacle, and the fatal hazard sat inside the arc. Nothing could be done after
                    takeoff; the error (if any) was the takeoff TIMING.
  LATE / NO ACTION  grounded at the fatal cell with a must-jump/slide hazard: the action was never
                    taken, or came too late.
  LATERAL           fell off the side (narrow strip / track edge).
  OTHER             everything else (stumble chains, flame timing, unclassifiable).

Also reports, per class, whether a PRECEDING hazard sat within the jump arc -- the "combination"
question -- so a compound cause is visible rather than hidden inside the hazard label.

    python scripts/death_behaviour.py                      # default capture set
    python scripts/death_behaviour.py captures/w15_*.jsonl
"""
from __future__ import annotations
import glob, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scratchpad"))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

NAME = {0: "none", 1: "Hole", 2: "Narrow", 3: "TreeStumble", 4: "TreeJump", 5: "TreeSlide",
        6: "FlameTower", 7: "Crevasse"}
MUST_JUMP = {1, 3, 4, 7}
MUST_SLIDE = {5, 6}
SPAN_A, SPAN_B = 0.0204, -0.94


def classify(ep, cells):
    """Return (class, hazard, detail) for this episode's death, or None if it did not die."""
    if not ep[-1].get("is_dead"):
        return None
    fatal_cell = cells[-1]
    obst, lvl = {}, {}
    for i, fr in enumerate(ep):
        obst.setdefault(cells[i], int(fr.get("obstacle") or 0))
        lvl.setdefault(cells[i], int(fr.get("path_level") or 2))
    haz = obst.get(fatal_cell, 0)
    # a hazard the fatal one sits close behind = the "combination" case
    prev_haz, prev_gap = None, None
    for back in range(1, 6):
        o = obst.get(fatal_cell - back)
        if o in MUST_JUMP | MUST_SLIDE:
            prev_haz, prev_gap = o, back
            break
    # the runner's last takeoff before dying, and whether anything required it
    last_takeoff = None
    for i in range(len(ep) - 1, 0, -1):
        if ep[i].get("is_jumping") and not ep[i - 1].get("is_jumping"):
            last_takeoff = i
            break
    airborne = bool(ep[-1].get("is_jumping"))
    justified, d = None, None
    if last_takeoff is not None:
        src = ep[last_takeoff - 1]
        justified = any(bool(src.get(k)) for k in
                        ("must_jump", "next_must_jump", "next2_must_jump", "next3_must_jump"))
        tk = cells[max(0, last_takeoff - 1)] + float(src.get("cell_progress") or 0.0)
        d = fatal_cell - tk
    lateral = (haz == 0 and not airborne) or haz == 2
    if airborne and justified is False:
        cls = "RANDOM-JUMP TRAP"
    elif airborne and justified:
        cls = "COMPOUND TRAP"
    elif haz in MUST_JUMP | MUST_SLIDE and not airborne:
        cls = "LATE / NO ACTION"
    elif lateral:
        cls = "LATERAL"
    else:
        cls = "OTHER"
    return cls, haz, {"prev": prev_haz, "gap": prev_gap, "d": d, "lvl": lvl.get(fatal_cell, 2)}


def main():
    globs = sys.argv[1:] or ["captures/w14_v78_holes.jsonl", "captures/ab_v78_r1.jsonl",
                             "captures/w15_v88_device.jsonl", "captures/w15_v83_device.jsonl",
                             "captures/w14_v79_device.jsonl", "captures/w15_v87_device.jsonl"]
    rows = []
    for g in globs:
        for path in sorted(glob.glob(g)):
            for epk, ep in load_eps(path):
                cl = cell_seq(ep)
                r = classify(ep, cl)
                if r:
                    rows.append((path.replace("\\", "/").split("/")[-1], epk) + r)
    n = len(rows)
    print(f"\nBEHAVIOURAL DEATH BUDGET — {n} deaths across {len(globs)} capture sets\n")
    byc = Counter(r[2] for r in rows)
    print(f"  {'cause (what the POLICY did)':26} {'deaths':>7} {'share':>7}   hazards it ended on")
    for c, k in byc.most_common():
        hz = Counter(NAME.get(r[3], r[3]) for r in rows if r[2] == c)
        print(f"  {c:26} {k:7} {100*k/n:6.0f}%   " + ", ".join(f"{a}x{b}" for a, b in hz.most_common(4)))
    comp = [r for r in rows if r[4]["prev"] is not None]
    print(f"\n  COMBINATIONS: {len(comp)}/{n} ({100*len(comp)/n:.0f}%) of deaths had ANOTHER hazard "
          f"within 5 cells behind the fatal one")
    gaps = Counter(r[4]["gap"] for r in comp)
    print("     gap to the preceding hazard: " + "  ".join(f"{g}:{gaps[g]}" for g in sorted(gaps)))
    pairs = Counter(f"{NAME.get(r[4]['prev'])}->{NAME.get(r[3])}" for r in comp)
    print("     most common fatal pairs: " + ", ".join(f"{a} ({b})" for a, b in pairs.most_common(6)))
    rj = [r for r in rows if r[2] == "RANDOM-JUMP TRAP"]
    if rj:
        print(f"\n  RANDOM-JUMP TRAP detail (the user's reported cause) — {len(rj)} deaths:")
        for r in rj:
            det = r[4]
            print(f"    {r[0][:24]:24} ep{r[1]} died on {NAME.get(r[3], r[3]):11} lvl{det['lvl']}"
                  f"  takeoff {det['d']:.1f} cells back"
                  + (f", after {NAME.get(det['prev'])} {det['gap']} cells earlier" if det["prev"] else
                     ", no preceding hazard"))


if __name__ == "__main__":
    main()
