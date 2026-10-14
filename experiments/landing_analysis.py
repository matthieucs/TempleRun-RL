#!/usr/bin/env python3
"""Landing-position addendum to human_vs_policy.py: for each gap-crossing jump, where does
the runner LAND relative to the gap? Landing on a hole cell (or cp<0.25 just past it) = lip graze
risk — the death mode. Compare human vs policy distributions, and takeoff-cp -> outcome coupling."""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict

HOLE_PT = set(range(11, 17))
CREVASSE_PT = set(range(33, 37))
GAP_PT = HOLE_PT | CREVASSE_PT


def median(xs):
    xs = sorted(xs)
    return xs[len(xs)//2] if xs else float("nan")


def load_eps(path):
    eps = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            eps[d.get("ep", 0)].append(d)
    out = []
    for k in sorted(eps):
        fs = eps[k]
        i = 0
        while i < len(fs) and fs[i].get("is_dead"):
            i += 1
        j = i
        while j < len(fs) and not fs[j].get("is_dead"):
            j += 1
        seg = fs[i:min(j+1, len(fs))]
        if len(seg) > 30:
            out.append(seg)
    return out


def analyze(paths, kind):
    lands = []   # dicts: takeoff cp, landing pt-is-gap, landing cp, vel, died_soon
    for path in paths:
        for ep in load_eps(path):
            hz = 30.0 if kind == "human" else 10.0
            prev_j = False
            tk = None
            death_i = len(ep) - 1 if ep[-1].get("is_dead") else None
            for i, fr in enumerate(ep):
                j = bool(fr.get("is_jumping"))
                if j and not prev_j:
                    prev = ep[i-1] if i else fr
                    # gap-crossing jump: takeoff on approach cell w/ next=gap, or on the gap cell
                    npt = prev.get("next_path_type")
                    on_gap = prev.get("path_type") in GAP_PT
                    before_gap = (npt in GAP_PT) if npt is not None else bool(prev.get("next_must_jump"))
                    if on_gap or before_gap:
                        tk = {"cp": prev.get("cell_progress"), "vel": float(fr.get("velocity") or 0),
                              "on_gap": on_gap, "i": i}
                    else:
                        tk = None
                if (not j) and prev_j and tk is not None:
                    land_pt = fr.get("path_type")
                    land_cp = fr.get("cell_progress")
                    died_soon = death_i is not None and (death_i - i) <= 1.5 * hz
                    lands.append({**tk, "land_gap": land_pt in GAP_PT, "land_cp": land_cp,
                                  "died_soon": died_soon})
                    tk = None
                prev_j = j
    return lands


def report(name, lands):
    n = len(lands)
    if not n:
        print(f"{name}: no gap jumps")
        return
    lg = [l for l in lands if l["land_gap"]]
    tight = [l for l in lands if not l["land_gap"] and l["land_cp"] is not None and l["land_cp"] < 0.25]
    died = [l for l in lands if l["died_soon"]]
    print(f"\n{name}: {n} gap-crossing jumps")
    print(f"  landed ON a gap cell (inside/lip): {len(lg)} ({100*len(lg)/n:.1f}%)")
    print(f"  landed just past (cp<0.25 next cell): {len(tight)} ({100*len(tight)/n:.1f}%)")
    print(f"  death within 1.5s of landing: {len(died)} ({100*len(died)/n:.1f}%)")
    # takeoff cp vs outcome
    early = [l for l in lands if not l["on_gap"] and l["cp"] is not None and l["cp"] < 0.4]
    late = [l for l in lands if not l["on_gap"] and l["cp"] is not None and l["cp"] >= 0.4]
    for lbl, xs in [("takeoff cp<0.4 (early)", early), ("takeoff cp>=0.4 (late)", late)]:
        if xs:
            d = sum(1 for l in xs if l["died_soon"])
            g = sum(1 for l in xs if l["land_gap"])
            print(f"  {lbl:24}: n={len(xs):3} land-on-gap {100*g/len(xs):5.1f}%  died-soon {100*d/len(xs):5.1f}%")
    # median landing cp when landing clean
    clean_cp = [l["land_cp"] for l in lands if not l["land_gap"] and l["land_cp"] is not None]
    print(f"  median landing cp (clean landings): {median(clean_cp):.2f}")


hpaths = sorted(glob.glob("captures/human_202607*.jsonl"))
ppaths = [p for g in ["captures/ab_v66_fresh17a.jsonl", "captures/ab_v66_bench.jsonl",
                      "captures/ab_v66_r1.jsonl", "captures/ab_v66_r2.jsonl"] for p in glob.glob(g)]
report("HUMAN", analyze(hpaths, "human"))
report("POLICY", analyze(ppaths, "policy"))
