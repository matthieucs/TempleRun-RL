#!/usr/bin/env python3
"""How does the human handle SLIDE->HOLE stacks (a slide obstacle 1-2 cells before a hole)?
Killed the human twice (ep2/ep3: sliding when the hole arrived, jump swipe refused mid-fall).
Find all slide onsets followed by a hole cell within 1.5s: survived vs died, and the timing
(slide onset cp, gap between slide end and hole entry, jump takeoff timing)."""
from __future__ import annotations
import glob, json
from collections import defaultdict

HOLE_PT = set(range(11, 17))

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
        seg = fs[i:min(j+1, len(fs))]
        if len(seg) > 30:
            out.append((k, seg))
    return out

events = []
for path in sorted(glob.glob("captures/human_202607*.jsonl")):
    for epk, ep in load_eps(path):
        prev_slide = False
        for i, fr in enumerate(ep):
            sl = bool(fr.get("is_sliding"))
            if sl and not prev_slide:
                # slide onset; scan forward 1.5s (45 fr) for a hole cell entry
                onset = fr
                hole_i = None
                for j in range(i, min(i+46, len(ep))):
                    if ep[j].get("path_type") in HOLE_PT and (j == 0 or ep[j-1].get("path_type") not in HOLE_PT):
                        hole_i = j
                        break
                if hole_i is not None:
                    # what happened: did a jump takeoff occur before falling? outcome?
                    died = False; jumped_at = None; slide_end = None
                    for j in range(i, min(hole_i+40, len(ep))):
                        if slide_end is None and not ep[j].get("is_sliding"):
                            slide_end = (j - i) / 30.0
                        if jumped_at is None and j > 0 and ep[j].get("is_jumping") and not ep[j-1].get("is_jumping"):
                            jumped_at = (j - hole_i) / 30.0  # takeoff rel to hole entry (s)
                        if ep[j].get("is_dead"):
                            died = True
                            break
                    events.append({
                        "ep": epk, "dist": float(onset.get("distance") or 0),
                        "vel": float(onset.get("velocity") or 0),
                        "slide_to_hole_s": (hole_i - i) / 30.0,
                        "slide_dur_s": slide_end, "jump_rel_hole_s": jumped_at,
                        "died": died,
                    })
            prev_slide = sl

print(f"{len(events)} slide->hole(within 1.5s) events")
for e in sorted(events, key=lambda x: x["slide_to_hole_s"]):
    print(f"  ep{e['ep']:2} @{e['dist']:6.0f} v={e['vel']:5.0f} slide->hole {e['slide_to_hole_s']:.2f}s "
          f"slideDur {e['slide_dur_s'] if e['slide_dur_s'] is not None else -1:.2f}s "
          f"jumpRelHole {e['jump_rel_hole_s'] if e['jump_rel_hole_s'] is not None else float('nan'):+.2f}s "
          f"{'DIED' if e['died'] else 'ok'}")
