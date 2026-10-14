#!/usr/bin/env python3
"""THREE-OBSTACLE CHAIN — the user's new blocker, measured.

USER (2026-07-27, watching v102 on the phone): "It does not strafe randomly anymore. It does not
jump dumbly onto the high ground. ... it hasn't been able to go further because it struggles to
handle the edge case where you have 3 obstacles one after the other. When this is fixed, the policy
should be able to reach the 10,000m. Maybe a new observation variable?"

Per the project's own standing rule (coverage BEFORE death model), the first question is NOT "is the
death model right" but "does this situation EXIST in the sim, at the device rate?". So this script
measures the same thing on both sides:

  * CHAIN = three hazard cells h1 < h2 < h3 with each gap <= --gap cells. A "hazard cell" is one
    that REQUIRES an action (must_jump or must_slide) -- narrows and plain cells do not count.
  * per km, per encounter, and the DEATH rate inside the chain, split by WHICH link kills.
  * VISIBILITY at commit: the obs carries must_jump/slide at +1,+2,+3 only. At the moment the runner
    must commit to h1 (the approach cell h1-1), h3 sits at (h3-h1+1) cells ahead -- so if that is
    > 3 the third obstacle is STRUCTURALLY INVISIBLE when the rhythm has to be chosen. That number
    is the evidence for or against a new observation, and it is what this reports.

    python scripts/analysis/chain3_forensic.py captures/w17_v102_device.jsonl
    python scripts/analysis/chain3_forensic.py "captures/human_20260708_*.jsonl" --label HUMAN
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, glob, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

# obstacle enum on the stream: 1 Hole, 2 Narrow, 3 TreeStumble, 4 TreeJump, 5 TreeSlide,
# 6 FlameTower, 7 LevelChange(crevasse)
MUST_JUMP = {1, 3, 4, 7}
MUST_SLIDE = {5, 6}
NAME = {1: "Hole", 2: "Narrow", 3: "Stumble", 4: "TreeJump", 5: "TreeSlide",
        6: "Flame", 7: "Crevasse"}


def cell_map(ep, cells):
    """cell index -> (kind, obstacle_type, level). kind in {'J','S',''}."""
    out = {}
    for i, fr in enumerate(ep):
        c = cells[i]
        o = int(fr.get("obstacle") or 0)
        kind = "J" if (o in MUST_JUMP or fr.get("must_jump")) else \
               "S" if (o in MUST_SLIDE or fr.get("must_slide")) else ""
        prev = out.get(c)
        # a cell keeps the strongest claim seen on it (flags are only presented in a window)
        if prev is None or (not prev[0] and kind):
            out[c] = (kind, o, int(fr.get("path_level") or 2))
    return out


def structures(cmap):
    """Collapse ADJACENT hazard cells of the SAME obstacle type into one structure.

    This matters: a 2-cell crevasse (pt 33-36) and a 2-cell HoleLong (pt 12/13/15/16) each occupy
    TWO hazard cells, so a naive cell-level scan reads one obstacle as the pair "Crevasse->Crevasse"
    at gap 1 and reports a tight chain that the player never sees. What the user means by three
    obstacles in a row is three STRUCTURES, so that is the unit here.

    Returns [(start_cell, end_cell, obstacle_type, kind, level)] in track order.
    """
    hz = sorted(c for c, v in cmap.items() if v[0])
    out = []
    for c in hz:
        kind, o, lvl = cmap[c]
        if out and c == out[-1][1] + 1 and o == out[-1][2]:
            s, _, oo, kk, ll = out[-1]
            out[-1] = (s, c, oo, kk, ll)
        else:
            out.append((c, c, o, kind, lvl))
    return out


def chains(cmap, gap):
    """Yield triples of STRUCTURES whose inter-structure clear gaps are <= gap cells."""
    st = structures(cmap)
    for i in range(len(st) - 2):
        a, b, c = st[i], st[i + 1], st[i + 2]
        if b[0] - a[1] <= gap and c[0] - b[1] <= gap:
            yield (a, b, c)


def scan(path, hz_rate, gap):
    rows, km = [], 0.0
    for _, ep in load_eps(path):
        cells = cell_seq(ep)
        km += max(float(f.get("distance") or 0) for f in ep) / 1000.0
        cmap = cell_map(ep, cells)
        death_cell = cells[-1] if ep[-1].get("is_dead") else None
        # distance/velocity at the first frame of each cell, for reporting
        first = {}
        for i, fr in enumerate(ep):
            first.setdefault(cells[i], fr)
        seen = set()
        for (a, b, c) in chains(cmap, gap):
            if a[0] in seen:         # count each chain START once
                continue
            seen.add(a[0])
            fr = first.get(a[0], ep[0])
            killed_at = None
            if death_cell is not None and a[0] <= death_cell <= c[1] + 2:
                killed_at = ("h1" if death_cell <= a[1] else
                             "h2" if death_cell <= b[1] else
                             "h3" if death_cell <= c[1] else "after")
            rows.append(dict(
                gaps=(b[0] - a[1], c[0] - b[1]), span=c[1] - a[0],
                kinds="".join(x[3] for x in (a, b, c)),
                types=tuple(x[2] for x in (a, b, c)),
                level=max(x[4] for x in (a, b, c)),
                v=float(fr.get("velocity") or 0.0),
                dist=float(fr.get("distance") or 0.0),
                killed=killed_at))
    return rows, km


def report(rows, km, label):
    n = len(rows)
    print(f"\n=== {label} ===  {n} three-obstacle chains over {km:.1f} km")
    if not n:
        return
    dead = [r for r in rows if r["killed"]]
    print(f"  rate {n/max(1e-9,km):5.2f} chains/km   "
          f"DEATHS inside a chain {len(dead)} = {100*len(dead)/n:4.1f}% per encounter")
    if dead:
        w = Counter(r["killed"] for r in dead)
        print(f"    which link kills: " + "  ".join(f"{k} {v}" for k, v in sorted(w.items())))
    # VISIBILITY: at commit to h1 (the approach cell), h3 sits span+1 cells ahead; obs reaches +3.
    vis = sum(1 for r in rows if r["span"] + 1 <= 3)
    print(f"  VISIBLE at commit (h3 within the +3 obs horizon): {vis}/{n} = {100*vis/n:.0f}%"
          f"   -> {100*(n-vis)/n:.0f}% of chains have an INVISIBLE third obstacle")
    g = Counter(r["gaps"] for r in rows)
    print("  gap pattern (h1->h2, h2->h3), top 8:")
    for k, v in g.most_common(8):
        sub = [r for r in rows if r["gaps"] == k]
        d = sum(1 for r in sub if r["killed"])
        print(f"     {k[0]}+{k[1]:<2} n={v:4}  deaths {d:3} ({100*d/v:4.1f}%)")
    print("  composition (top 6):")
    for k, v in Counter(r["types"] for r in rows).most_common(6):
        print(f"     {'->'.join(NAME.get(t, str(t)) for t in k):42} n={v}")
    hi = [r for r in rows if r["level"] >= 3]
    if hi:
        d = sum(1 for r in hi if r["killed"])
        print(f"  on HIGH ground (level>=3): {len(hi)} chains, {d} deaths ({100*d/len(hi):.1f}%)")
    fast = [r for r in rows if r["v"] >= 180]
    if fast:
        d = sum(1 for r in fast if r["killed"])
        print(f"  at speed (v>=180):         {len(fast)} chains, {d} deaths ({100*d/len(fast):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--gap", type=int, default=4, help="max cells between consecutive hazards")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    for g in a.captures:
        paths = sorted(glob.glob(g))
        if not paths:
            continue
        rows, km = [], 0.0
        for p in paths:
            r, k = scan(p, 30.0 if "human" in p else 10.0, a.gap)
            rows += r
            km += k
        report(rows, km, a.label or g)


if __name__ == "__main__":
    main()
