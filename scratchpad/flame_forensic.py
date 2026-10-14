#!/usr/bin/env python3
"""FLAME SLIDE-UNDER forensic (WAVE 12): the mechanic that ended all three record runs
(10545/11560/11731) and kills in the median band (1731/2872 today). For every flame-cell
(pt31/32) encounter across captures: slide onset/end timing vs cell entry, velocity,
outcome (clean / stumble / Burnt dt5). Died-vs-survived timing separation = the sim knob spec.
Usage: python scratchpad/flame_forensic.py captures\\ab_v70*.jsonl captures\\best_run*.jsonl ..."""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict

FLAME_PT = {31, 32}


def load_eps(path):
    eps = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
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
        if j - i > 30:
            out.append((k, fs[i:min(j + 1, len(fs))]))
    return out


events = []
paths = [p for g in (sys.argv[1:] or ["captures/ab_v70*.jsonl", "captures/best_run_11*.jsonl",
                                      "captures/ab_v66_wave9*.jsonl", "captures/human_202607*.jsonl"])
         for p in sorted(glob.glob(g))]
for path in paths:
    hz = 30.0 if "human" in path else 10.0
    for epk, ep in load_eps(path):
        for i, fr in enumerate(ep):
            pt = fr.get("path_type")
            if pt in FLAME_PT and (i == 0 or ep[i - 1].get("path_type") not in FLAME_PT):
                # flame-cell entry at frame i; scan the encounter
                entry_v = float(fr.get("velocity") or 0)
                sliding_at_entry = bool(fr.get("is_sliding"))
                jumping_at_entry = bool(fr.get("is_jumping"))
                # slide onset relative to entry (scan back 2s and fwd 1s)
                onset_rel = None
                for j in range(max(0, i - int(2 * hz)), min(len(ep), i + int(1 * hz))):
                    if ep[j].get("is_sliding") and (j == 0 or not ep[j - 1].get("is_sliding")):
                        onset_rel = (j - i) / hz
                        if j <= i:
                            break  # keep the latest onset at/before entry, else first after
                died = False
                stumbled = False
                for j in range(i, min(len(ep), i + int(1.5 * hz))):
                    if ep[j].get("is_dead"):
                        died = int(ep[j].get("death_type") or 0) == 5 or \
                               ep[j].get("path_type") in FLAME_PT
                        break
                    if ep[j].get("is_stumbling") and not ep[max(0, j - 1)].get("is_stumbling"):
                        stumbled = True
                events.append({"src": path.split("\\")[-1].split("/")[-1], "ep": epk,
                               "dist": float(fr.get("distance") or 0), "v": entry_v,
                               "slide_entry": sliding_at_entry, "jump_entry": jumping_at_entry,
                               "onset_rel": onset_rel, "died": died, "stumbled": stumbled})

died = [e for e in events if e["died"]]
ok = [e for e in events if not e["died"]]
print(f"{len(events)} flame encounters | {len(died)} deaths | "
      f"{sum(1 for e in events if e['stumbled'])} stumbles")


def show(tag, xs):
    print(f"\n--- {tag} (n={len(xs)}) ---")
    for e in xs:
        print(f"  {e['src'][:28]:28} ep{e['ep']:2} @{e['dist']:6.0f} v={e['v']:5.0f} "
              f"slideEntry={str(e['slide_entry']):5} jumpEntry={str(e['jump_entry']):5} "
              f"onsetRel={e['onset_rel'] if e['onset_rel'] is not None else 'NONE'}")


show("DIED", died)
# survived: summarize by velocity band
import statistics as st
for lo, hi in ((0, 170), (170, 200), (200, 999)):
    band = [e for e in ok if lo <= e["v"] < hi and e["onset_rel"] is not None]
    if band:
        med = st.median(e["onset_rel"] for e in band)
        se = sum(1 for e in band if e["slide_entry"]) / len(band)
        print(f"\nSURVIVED v[{lo},{hi}): n={len(band)} onsetRel med {med:+.2f}s "
              f"slideAtEntry {100*se:.0f}%")
