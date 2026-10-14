#!/usr/bin/env python3
"""WAVE 14: measure the JUMP SPAN (cells travelled between liftoff and touchdown) with ONE
estimator applied to BOTH device captures and sim rollouts.

Why: the device dies on holes it enters ALREADY AIRBORNE (d = cells from takeoff to the hole > 1);
the sim never even produces d > 1. If the sim's arc is systematically SHORTER, its runner lands
sooner, is free to re-act, and the airborne-trap can never be experienced -> unlearnable.

Usage:
  python scratchpad/jump_span_compare.py device captures/ab_v78*.jsonl captures/ab_v74*.jsonl
  python scratchpad/jump_span_compare.py sim scratchpad/_sim_holes.jsonl
"""
from __future__ import annotations
import glob, sys
from collections import defaultdict
from hole_landing_forensic import load_eps, cell_seq


def spans(paths):
    out = []
    for path in paths:
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            takeoff = None
            for i, fr in enumerate(ep):
                jmp = bool(fr.get("is_jumping"))
                pj = bool(ep[i - 1].get("is_jumping")) if i else False
                if jmp and not pj:
                    src = ep[i - 1] if i else fr
                    takeoff = (cells[max(0, i - 1)] + float(src.get("cell_progress") or 0.0),
                               float(src.get("velocity") or 0.0))
                elif pj and not jmp and takeoff is not None:
                    if fr.get("is_dead"):
                        takeoff = None
                        continue
                    land = cells[i] + float(fr.get("cell_progress") or 0.0)
                    span = land - takeoff[0]
                    if 0.2 < span < 8.0:
                        out.append((takeoff[1], span, takeoff[0] % 1.0))
                    takeoff = None
    return out


def report(tag, xs):
    print(f"\n=== {tag}: n={len(xs)} jumps ===")
    buckets = defaultdict(list)
    tk = defaultdict(list)
    for v, s, c in xs:
        buckets[int(v // 20) * 20].append(s)
        tk[int(v // 20) * 20].append(c)
    print("   v-bucket    n   span median   mean   |  TAKEOFF cp median  mean")
    for k in sorted(buckets):
        ss = sorted(buckets[k]); cc = sorted(tk[k])
        if len(ss) < 3:
            continue
        print(f"   {k:3}-{k+19:3}  {len(ss):4}   {ss[len(ss)//2]:6.2f}      {sum(ss)/len(ss):6.2f}   |"
              f"        {cc[len(cc)//2]:5.2f}      {sum(cc)/len(cc):5.2f}")
    if len(xs) > 10:
        import numpy as np
        v = np.array([a for a, _, _ in xs]); s = np.array([b for _, b, _ in xs])
        a, b = np.polyfit(v, s, 1)
        allc = sorted(c for _, _, c in xs)
        print(f"   linear fit: span = {a:.4f}*v + {b:.2f}   (device wave-8 fit: 0.0204*v - 0.94)")
        print(f"   takeoff cp overall: median {allc[len(allc)//2]:.2f}  mean {sum(allc)/len(allc):.2f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    paths = [p for g in sys.argv[2:] for p in sorted(glob.glob(g))]
    report(mode.upper() + " " + ",".join(sys.argv[2:]), spans(paths))
