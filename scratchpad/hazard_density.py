#!/usr/bin/env python3
"""HAZARD DENSITY — how many hazards per km does each world actually contain?

WHY THIS EXISTS (2026-07-27). device_compare says the sim is, per ENCOUNTER, no harder than the
phone (death/hazard 1.19% sim vs 1.88% device, p=0.33) -- yet the same policy's sim median sits far
below its device median. Both cannot be true unless the sim serves MORE ENCOUNTERS PER KM. Per-km
risk = (hazards/km) x (death/hazard), so density is the missing factor, and it is policy-independent
and high-power: thousands of cells per capture, no death events required.

This is the leg the wave-15 audit missed. That audit matched the SPACING SHAPE (share of consecutive
hazard pairs falling inside the jump arc: tuned to 11.1% vs the device's 10.7%) -- but that statistic
is CONDITIONAL on a pair existing. It says how close neighbours sit, never how many there are. A
world can match it exactly while emitting twice as many obstacles.

Hazards are STRUCTURE-COLLAPSED (a 2-cell HoleLong or a multi-cell narrow counts once), so the count
is "things the runner must solve", not cells touched. Reported per distance band because the game's
difficulty ramp is distance-keyed: comparing a sim rollout that dies at 1800 m against device runs
reaching 5100 m would otherwise compare different ramp states.

    python scratchpad/hazard_density.py scratchpad/_sim_holes_v90simhold.jsonl captures/w16_v90_device.jsonl
"""
from __future__ import annotations
import glob, sys
from collections import defaultdict
from hole_landing_forensic import load_eps, cell_seq

HAZARDS = {1, 2, 3, 4, 5, 6, 7}
NAME = {1: "Hole", 2: "Narrow", 3: "TreeStumble", 4: "TreeJump", 5: "TreeSlide",
        6: "FlameTower", 7: "Crevasse"}
BANDS = [(0, 1500), (1500, 3000), (3000, 4500), (4500, 100000)]


def measure(paths):
    """-> {band: [hazard_structures, distance_covered]}, {type: count}, cells, distance"""
    per = {b: [0, 0.0] for b in BANDS}
    kinds = defaultdict(int)
    cells_tot, dist_tot = 0, 0.0
    for path in paths:
        for _, ep in load_eps(path):
            cl = cell_seq(ep)
            # cell -> (obstacle, distance at first sight)
            cell_ob, cell_d = {}, {}
            for i, fr in enumerate(ep):
                if cl[i] not in cell_ob:
                    cell_ob[cl[i]] = int(fr.get("obstacle") or 0)
                    cell_d[cl[i]] = float(fr.get("distance") or 0.0)
            ks = sorted(cell_ob)
            if len(ks) < 2:
                continue
            # ⚠️ A REAL sim narrow strip is path_type 17..30 with obstacles LEFT AT 0, while on DEVICE a
            # narrow sets obstacle == 2. Counting the obstacle field alone therefore counts only the
            # sim's FAKE (laneless, harmless) narrows and misses every real strip — which on 2026-07-27
            # made the sim look 2.2x SHORT of narrows when it actually emits ~3x TOO MANY, and sent a
            # whole training arm the wrong way. Normalise both sides here.
            for i2, fr2 in enumerate(ep):
                if cell_ob.get(cl[i2]) == 0 and 17 <= int(fr2.get("path_type") or 0) <= 30:
                    cell_ob[cl[i2]] = 2
            span = cell_d[ks[-1]] - cell_d[ks[0]]
            if span <= 0:
                continue
            cells_tot += len(ks)
            dist_tot += span
            cell_len = span / max(1, len(ks) - 1)
            prev_ob = None
            for k in ks:
                ob = cell_ob[k]
                # collapse a run of the same hazard type into ONE structure
                new = ob in HAZARDS and ob != prev_ob
                prev_ob = ob
                d = cell_d[k]
                for b in BANDS:
                    if b[0] <= d < b[1]:
                        per[b][1] += cell_len
                        if new:
                            per[b][0] += 1
                            kinds[ob] += 1
                        break
    return per, kinds, cells_tot, dist_tot


def main():
    for g in sys.argv[1:]:
        paths = sorted(glob.glob(g))
        if not paths:
            continue
        per, kinds, cells, dist = measure(paths)
        tot_h = sum(v[0] for v in per.values())
        print(f"\n=== {g} ===  {cells} cells over {dist/1000:.1f} km "
              f"(cell length ~{dist/max(1,cells):.1f})")
        print(f"  OVERALL: {tot_h} hazard structures = {1000*tot_h/max(1e-9,dist):.1f} per km")
        for b in BANDS:
            h, d = per[b]
            if d > 100:
                print(f"    {b[0]:5}-{b[1] if b[1]<99999 else 0:5} m: "
                      f"{1000*h/d:5.1f} /km   ({h} over {d/1000:.1f} km)")
        s = sum(kinds.values()) or 1
        print("  mix: " + ", ".join(f"{NAME.get(k,k)} {100*v/s:.0f}%"
                                    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
