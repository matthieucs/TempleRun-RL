#!/usr/bin/env python3
"""HIGH-POWER DEVICE COMPARISON — per-ENCOUNTER rates, not 8 episode distances.

WHY (USER, 2026-07-26): "the record of v78 is based on randomness... any of the models deployed after
v78 had the potential to do as great." Measured and CORRECT. An 8-episode median has a 95% CI of
roughly [3235, 8092] on this game, v78-vs-v88 comes out at p=0.50, and the POWER to detect a genuine
20% median improvement at n=8 is **11%** -- i.e. a real 20% gain is missed 9 times out of 10.
Reaching 80% power needs ~48 episodes per arm (~2.5 h of phone time). Every device "verdict" issued
on n=8 medians this week was a coin flip dressed as a conclusion.

THE FIX: stop ranking on 8 numbers. One 8-episode batch already contains HUNDREDS of hazard
encounters, so per-encounter rates carry 10-100x more information for the SAME phone time -- and they
say WHICH mechanic changed instead of just "better/worse". Distance stays as a headline, but it is
the least informative column here.

    python scripts/device_compare.py A.jsonl B.jsonl [--labels v78 v88]
"""
from __future__ import annotations
import argparse, glob, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scratchpad"))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

HOLE, CREV = 1, 7
MUST_JUMP = {1, 3, 4, 7}


def wilson(k, n, z=1.96):
    """Wilson score interval — correct for small-n rates, unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))


def two_prop_p(k1, n1, k2, n2):
    """Two-proportion z-test (normal approx; fine at the n these rates reach)."""
    if min(n1, n2) == 0:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(k1 / n1 - k2 / n2) / se
    return math.erfc(z / math.sqrt(2))


def collect(globs):
    """Per-encounter counters. Each is (events_that_killed, total_events)."""
    st = {k: [0, 0] for k in ("hole", "crevasse_lvl3_graze", "junction", "any_hazard")}
    dists = []
    for g in globs:
        for path in sorted(glob.glob(g)):
            for _, ep in load_eps(path):
                cells = cell_seq(ep)
                dists.append(max(float(f.get("distance") or 0) for f in ep))
                obst, turn, lvl = {}, {}, {}
                for i, fr in enumerate(ep):
                    obst.setdefault(cells[i], int(fr.get("obstacle") or 0))
                    lvl.setdefault(cells[i], int(fr.get("path_level") or 2))
                    if fr.get("is_turn"):
                        turn[cells[i]] = True
                died_cell = cells[-1] if ep[-1].get("is_dead") else None
                grazed = set()
                for i, fr in enumerate(ep):
                    if fr.get("is_stumbling") and not ep[max(0, i - 1)].get("is_stumbling"):
                        grazed.add(cells[i])
                for c, o in obst.items():
                    if o in MUST_JUMP:
                        st["any_hazard"][1] += 1
                        if died_cell is not None and 0 <= died_cell - c <= 1:
                            st["any_hazard"][0] += 1
                    if o == HOLE:
                        st["hole"][1] += 1
                        if died_cell is not None and 0 <= died_cell - c <= 1:
                            st["hole"][0] += 1
                    if o == CREV and lvl.get(c, 2) >= 3:
                        st["crevasse_lvl3_graze"][1] += 1
                        if any(c <= gc <= c + 2 for gc in grazed):
                            st["crevasse_lvl3_graze"][0] += 1
                for c in turn:
                    st["junction"][1] += 1
                    if died_cell is not None and 0 <= died_cell - c <= 3:
                        st["junction"][0] += 1
    return st, dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    x = ap.parse_args()
    A, da = collect([x.a])
    B, db = collect([x.b])
    la, lb = x.labels
    import statistics as S
    print(f"\n  {'metric':26} {la:>22} {lb:>22}   p")
    print(f"  {'-'*26} {'-'*22} {'-'*22}   ---")
    print(f"  {'episodes':26} {len(da):>22} {len(db):>22}")
    print(f"  {'median distance (LOW POWER)':26} {S.median(da):>22.0f} {S.median(db):>22.0f}   ~0.5 at n=8")
    for key, nice in (("any_hazard", "death / hazard passed"),
                      ("hole", "death / hole crossing"),
                      ("crevasse_lvl3_graze", "wall-clip / lvl3 crevasse"),
                      ("junction", "death within 3 cells of turn")):
        ka, na = A[key]
        kb, nb = B[key]
        ra = wilson(ka, na)
        rb = wilson(kb, nb)
        p = two_prop_p(ka, na, kb, nb)
        star = " <<<" if p < 0.05 else ""
        print(f"  {nice:26} {ra[0]:8.2f}% [{ra[1]:4.1f},{ra[2]:4.1f}] n={na:<4} "
              f"{rb[0]:8.2f}% [{rb[1]:4.1f},{rb[2]:4.1f}] n={nb:<4} {p:5.3f}{star}")
    print("\n  n = ENCOUNTERS, not episodes — this is where the statistical power is.")


if __name__ == "__main__":
    main()
