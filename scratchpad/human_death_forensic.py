#!/usr/bin/env python3
"""Forensic on the HUMAN's deaths: last 3s before each death — swipe intents (gi_should_jump,
including REFUSED swipes that never became a jump), jump/slide state, path context, speed.
Answers: did the human die swiping-but-refused (mid-air / past-lip), or not swiping at all?"""
from __future__ import annotations
import glob, json
from collections import defaultdict

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
            out.append((path, k, seg))
    return out

for path in sorted(glob.glob("captures/human_202607*.jsonl")):
    for p, epk, ep in load_eps(path):
        last = ep[-1]
        if not last.get("is_dead"):
            continue
        dist = float(last.get("distance") or 0)
        dt, pt = last.get("death_type"), last.get("path_type")
        print(f"\n=== {p.split(chr(92))[-1]} ep{epk}: DEATH @{dist:.0f}m pt={pt} dt={dt} "
              f"vel={last.get('death_run_velocity')}")
        # walk back 3s (90 frames), print salient transitions
        tail = ep[max(0, len(ep)-91):]
        prev = None
        for fr in tail:
            ev = []
            if prev is not None:
                if fr.get("is_jumping") and not prev.get("is_jumping"): ev.append("JUMP-TAKEOFF")
                if prev.get("is_jumping") and not fr.get("is_jumping"): ev.append("land")
                if fr.get("is_sliding") and not prev.get("is_sliding"): ev.append("SLIDE")
                if fr.get("gi_should_jump") and not prev.get("gi_should_jump"): ev.append("swipe-UP")
                if fr.get("is_falling") and not prev.get("is_falling"): ev.append("FALLING")
                if fr.get("is_stumbling") and not prev.get("is_stumbling"): ev.append("STUMBLE")
                if fr.get("path_type") != prev.get("path_type"):
                    ev.append(f"cell:pt{prev.get('path_type')}->pt{fr.get('path_type')}")
                if fr.get("is_dead") and not prev.get("is_dead"): ev.append("**DEAD**")
            if ev:
                print(f"  t-{(len(tail)-1-tail.index(fr))/30:5.2f}s d={float(fr.get('distance') or 0):7.1f} "
                      f"cp={fr.get('cell_progress') if fr.get('cell_progress') is not None else -1:5.2f} "
                      f"v={float(fr.get('velocity') or 0):5.1f} lvl={fr.get('path_level')} "
                      f"wy={fr.get('world_y'):6.1f} | {', '.join(ev)}")
            prev = fr
