#!/usr/bin/env python3
"""NARROW-STRIP + STRAFE FORENSIC.

USER (2026-07-27, watching v92b): "it has come with the cost of a policy that is not as great on the
narrow and strafes a bit randomly sometimes. I think that both are linked. The training recipe may
have overfitted on the fix. We need extra training for the narrows."

Two claims to separate, because they imply different fixes:
  * NARROW SKILL   — can it hold the strip? Measured as death rate per narrow encounter and the LANE
                     ERROR (|lane - narrow_target_lane|) while on one. Device strips sit at |lane|
                     ~4.5, never centre, so a policy that drifts toward centre falls off the inner
                     edge. If lane error is up, the strip skill itself eroded -> needs narrow exposure.
  * STRAFE DITHER  — pointless lateral input where no narrow is involved. Measured as strafe-direction
                     REVERSALS per km while OFF a narrow and with no narrow within 2 cells. A policy
                     that flips left/right repeatedly on open track is dithering, which is a different
                     failure from being bad at strips (and would point at the action-cost/penalty
                     balance rather than at exposure).

Comparing two captures tells you whether a recipe change traded one for the other.

    python scripts/analysis/narrow_forensic.py captures/w16_v90_device.jsonl captures/w17_v92b_device.jsonl
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, glob, statistics as S, sys
from pathlib import Path

from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

NARROW_OBST = 2
STRAFE_L, STRAFE_R = 0, 1


def analyse(paths, label):
    km = 0.0
    enc, deaths = 0, 0
    lane_err, off_narrow_frames, reversals = [], 0, 0
    narrow_frames = 0
    for p in paths:
        for _, ep in load_eps(p):
            cells = cell_seq(ep)
            km += max(float(f.get("distance") or 0) for f in ep) / 1000.0
            obst = {}
            for i, fr in enumerate(ep):
                obst.setdefault(cells[i], int(fr.get("obstacle") or 0))
            last_dir, prev_narrow = None, False
            for i, fr in enumerate(ep):
                on_narrow = int(fr.get("obstacle") or 0) == NARROW_OBST
                tgt = fr.get("narrow_target_lane")
                if on_narrow and not prev_narrow:
                    enc += 1
                    # did the run end on this strip?
                    if any(ep[j].get("is_dead") for j in range(i, min(len(ep), i + 40))):
                        deaths += 1
                prev_narrow = on_narrow
                if on_narrow and tgt is not None and fr.get("lane") is not None:
                    lane_err.append(abs(float(fr["lane"]) - float(tgt)))
                    narrow_frames += 1
                # dither: strafe reversals with NO narrow on this or the next 2 cells
                near_narrow = any(obst.get(cells[i] + d, 0) == NARROW_OBST for d in (0, 1, 2))
                if not near_narrow:
                    off_narrow_frames += 1
                    a = fr.get("action")
                    if a in (STRAFE_L, STRAFE_R):
                        if last_dir is not None and a != last_dir:
                            reversals += 1
                        last_dir = a
    print(f"\n=== {label} ===  {km:.1f} km")
    print(f"  narrow encounters      {enc}  ({enc/max(1e-9,km):.1f} per km)")
    print(f"  died on a narrow       {deaths}  = {100*deaths/max(1,enc):.2f}% per encounter")
    if lane_err:
        print(f"  LANE ERROR on strip    median {S.median(lane_err):.2f}  "
              f"mean {S.mean(lane_err):.2f}  (n={len(lane_err)} frames)")
        bad = sum(1 for e in lane_err if e > 2.0)
        print(f"     frames >2.0 off the target lane: {100*bad/len(lane_err):.1f}%")
    print(f"  STRAFE REVERSALS off-narrow  {reversals}  = {reversals/max(1e-9,km):.1f} per km "
          f"({100*reversals/max(1,off_narrow_frames):.2f}% of off-narrow frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    a = ap.parse_args()
    for g in a.captures:
        paths = sorted(glob.glob(g))
        if paths:
            analyse(paths, g)


if __name__ == "__main__":
    main()
