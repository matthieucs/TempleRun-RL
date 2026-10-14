#!/usr/bin/env python3
"""NAMED-PATTERN DETECTOR — find the SPECIFIC mistakes the user reports, by name, with instances.

USER (2026-07-26, after reporting it ~50 times): "It still has the problem on the high-ground where
it jumps for no reason just after having jumped over a crevasse! ... For such classic mistakes, how
do you analyze the logs to detect them. Do you read them? Can't you build a script to detect that?"

Yes -- and this should have existed long ago. Every analysis so far has been AGGREGATE (per-hazard
death rates, behavioural budgets). Aggregates average away a specific recurring mistake: a pattern
that fires 6 times in a run and kills once is invisible at 1.5%/encounter, but it is exactly what a
human watching the screen sees every single time.

This scans any capture for NAMED patterns, reports rate per km, how often each is UNJUSTIFIED (no
hazard required the action) and how often it KILLS, and prints individual instances with distance so
they can be checked against the footage.

  POST_CREVASSE_REJUMP   the user's #1 complaint: land from a crevasse jump, then jump again within
                         ~1s with nothing requiring it. Split by ground level (their report is that
                         it happens on HIGH ground).
  BLIND_LANDING_JUMP     a jump whose arc comes down on a must-jump hazard (the landing_hazard case).
  HELD_JUMP_SPAM         3+ consecutive JUMP presses while airborne (the documented held-jump reflex).
  SLIDE_INTO_GAP         a slide committed within 2 cells of a hole (the ~0.7s commit eats the jump).

    python scripts/detect_patterns.py captures/w16_v90_device.jsonl
    python scripts/detect_patterns.py "captures/*.jsonl" --pattern POST_CREVASSE_REJUMP
"""
from __future__ import annotations
import argparse, glob, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scratchpad"))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from templerun_rl import obs as O  # noqa: E402

CREV, HOLE = 7, 1
MUST_JUMP = {1, 3, 4, 7}
JUMP = 2


def justified(fr):
    return any(bool(fr.get(k)) for k in
               ("must_jump", "next_must_jump", "next2_must_jump", "next3_must_jump"))


def scan(ep, cells, hz):
    """Yield (pattern, dist, level, unjustified, killed) for each detected instance."""
    n = len(ep)
    died_at = n - 1 if ep[-1].get("is_dead") else None

    def killed_after(i, secs=3.0):
        return died_at is not None and 0 <= died_at - i <= int(secs * hz)

    obst = {}
    for i, fr in enumerate(ep):
        obst.setdefault(cells[i], int(fr.get("obstacle") or 0))

    for i in range(2, n):
        fr, prev = ep[i], ep[i - 1]
        took_off = fr.get("is_jumping") and not prev.get("is_jumping")
        landed = prev.get("is_jumping") and not fr.get("is_jumping")

        # --- POST_CREVASSE_REJUMP -------------------------------------------------------------
        if took_off:
            # was the PREVIOUS airborne phase a crevasse crossing, and did it end recently?
            land_i = None
            for j in range(i - 1, max(0, i - int(1.2 * hz)) - 1, -1):
                if ep[j - 1].get("is_jumping") and not ep[j].get("is_jumping"):
                    land_i = j
                    break
            if land_i is not None:
                crossed_crev = any(int(ep[j].get("obstacle") or 0) == CREV
                                   for j in range(max(0, land_i - int(1.5 * hz)), land_i + 1))
                if crossed_crev:
                    yield ("POST_CREVASSE_REJUMP", float(prev.get("distance") or 0),
                           int(prev.get("path_level") or 2), not justified(prev), killed_after(i))
            # --- BLIND_LANDING_JUMP: does the arc land on a hazard? ----------------------------
            # ⚠️ WAS `0.0204*v - 0.94` -- the RETIRED wave-8 fit, which over-states the span by
            # 13-17% at running speed and therefore flagged the WRONG landing cell for this, the
            # deadliest pattern in the project. Use the shipped constants (and the same height
            # scaling the obs uses) so the detector and the policy's own cue agree on where the
            # arc comes down.
            v = float(prev.get("velocity") or 0)
            span = max(0.0, O.LAND_SPAN_A * v + O.LAND_SPAN_B)
            if O.LAND_SPAN_LEVEL_SCALED:
                l0 = int(prev.get("path_level") or 2)
                la = int(prev.get("next2_path_level") or l0)
                if la <= 0:
                    la = int(prev.get("next_path_level") or l0)
                if la > 0 and l0 > 0:
                    if la < l0:
                        span *= O.LAND_SPAN_STEPDOWN
                    elif la > l0:
                        span *= O.LAND_SPAN_STEPUP
            off = int(float(prev.get("cell_progress") or 0) + span)
            if 0 < off <= 3 and obst.get(cells[i - 1] + off, 0) in MUST_JUMP:
                yield ("BLIND_LANDING_JUMP", float(prev.get("distance") or 0),
                       int(prev.get("path_level") or 2), not justified(prev), killed_after(i))

        # --- HELD_JUMP_SPAM ---------------------------------------------------------------------
        if i + 2 < n and all(ep[k].get("action") == JUMP and ep[k].get("is_jumping")
                             for k in (i, i + 1, i + 2)):
            if not (ep[i - 1].get("action") == JUMP and ep[i - 1].get("is_jumping")):
                yield ("HELD_JUMP_SPAM", float(fr.get("distance") or 0),
                       int(fr.get("path_level") or 2), True, killed_after(i))

        # --- SLIDE_INTO_GAP ---------------------------------------------------------------------
        if fr.get("is_sliding") and not prev.get("is_sliding"):
            if any(obst.get(cells[i] + d, 0) == HOLE for d in (1, 2)):
                yield ("SLIDE_INTO_GAP", float(fr.get("distance") or 0),
                       int(fr.get("path_level") or 2), False, killed_after(i))

        # --- AIRBORNE_STRAFE --------------------------------------------------------------------
        # USER 2026-07-27, watching v98: "It strafes randomly to the left and it died on a narrow."
        # MECHANISM: `airborne_action_penalty` makes JUMP-while-airborne costly, and while airborne a
        # JUMP does nothing anyway — so the policy dumps its action mass into the cheapest remaining
        # action. A pointless mid-air tilt is nearly free in the sim; on DEVICE it moves the runner
        # laterally and walks it off a narrow strip. Degenerate and one-sided when it appears:
        # airborne strafes were 1.0% (L/R 20/0) for v92b but 8.5% (502/18) for v98 in sim, and 0.9%
        # vs 9.1% (38/0) on device — i.e. THE SIM PREDICTS IT, so gate on this BEFORE spending phone
        # time. Only counted when no narrow is within reach, so the legitimate jump->narrow mid-air
        # tilt (the jumpnarrow scenario) is not flagged.
        if fr.get("is_jumping") and fr.get("action") in (0, 1):
            if not any(obst.get(cells[i] + d, 0) == 2 for d in (0, 1, 2, 3)):
                yield ("AIRBORNE_STRAFE", float(fr.get("distance") or 0),
                       int(fr.get("path_level") or 2), True, killed_after(i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--pattern", default="")
    ap.add_argument("--list", action="store_true", help="print every instance")
    a = ap.parse_args()
    hits = defaultdict(list)
    km = 0.0
    for g in a.captures:
        for path in sorted(glob.glob(g)):
            hz = 30.0 if "human" in path else 10.0
            for epk, ep in load_eps(path):
                cells = cell_seq(ep)
                km += max(float(f.get("distance") or 0) for f in ep) / 1000.0
                for pat, dist, lvl, unj, kill in scan(ep, cells, hz):
                    if a.pattern and pat != a.pattern:
                        continue
                    hits[pat].append((path.replace("\\", "/").split("/")[-1], epk, dist, lvl, unj, kill))
    print(f"\nNAMED-PATTERN SCAN — {km:.1f} km\n")
    print(f"  {'pattern':24} {'count':>6} {'per km':>7} {'unjustified':>12} {'killed<=3s':>11}")
    for pat, xs in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        u = sum(1 for x in xs if x[4])
        k = sum(1 for x in xs if x[5])
        print(f"  {pat:24} {len(xs):6} {len(xs)/max(1e-9,km):7.2f} "
              f"{u:6} ({100*u/len(xs):3.0f}%) {k:5} ({100*k/len(xs):3.0f}%)")
    for pat, xs in hits.items():
        hi = [x for x in xs if x[3] >= 3]
        if hi:
            u = sum(1 for x in hi if x[4])
            print(f"\n  {pat} on HIGH GROUND (level>=3): {len(hi)} of {len(xs)}, "
                  f"{u} unjustified, {sum(1 for x in hi if x[5])} lethal")
    if a.list or a.pattern:
        for pat, xs in hits.items():
            print(f"\n  --- {pat} instances ---")
            for src, epk, dist, lvl, unj, kill in xs[:40]:
                print(f"    {src[:26]:26} ep{epk} @{dist:6.0f}m lvl{lvl}"
                      f"{'  UNJUSTIFIED' if unj else ''}{'  <<< KILLED' if kill else ''}")


if __name__ == "__main__":
    main()
