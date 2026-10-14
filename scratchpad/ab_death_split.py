#!/usr/bin/env python3
"""Death-type + hole-band split for the v74 vs v70 A/B (WAVE 12 crown)."""
import json, glob
from collections import Counter

def deaths(paths):
    c = Counter()
    band = [0, 0]  # [median-band 1500-4500 deaths, of which hole/flame pt]
    for path in paths:
        eps = {}
        for line in open(path, encoding="utf-8", errors="replace"):
            try: d = json.loads(line)
            except ValueError: continue
            eps.setdefault(d.get("ep", 0), []).append(d)
        for fs in eps.values():
            last = fs[-1]
            if not last.get("is_dead"): continue
            pt = last.get("path_type")
            c[pt] += 1
            dist = float(last.get("distance") or 0)
            if 1500 <= dist <= 4500:
                band[0] += 1
                if pt in (11,12,13,14,15,16,31,32): band[1] += 1
    return c, band

for name, glb in (("v74", ["captures/ab_v74_r1.jsonl", "captures/ab_v74_r2.jsonl"]),
                  ("v70", ["captures/ab_v70_crown2_r1.jsonl", "captures/ab_v70_crown2_r2.jsonl"])):
    paths = [p for g in glb for p in glob.glob(g)]
    c, band = deaths(paths)
    print(f"{name}: death pt {dict(c)} | median-band(1500-4500) deaths {band[0]}, of which hole/flame {band[1]}")
