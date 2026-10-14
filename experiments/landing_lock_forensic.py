#!/usr/bin/env python3
"""LANDING-LOCK probe from existing captures (WAVE 12): does the game EAT actions in the frames
right after a jump lands? For every landing (is_jumping falling edge) in policy captures (10Hz,
explicit action stream), track the next grounded JUMP/SLIDE action: delay-since-landing (frames)
vs whether it FIRED (is_jumping/is_sliding rose within 2 frames). A fire-rate dip at delay 0-1
vs 3+ = the lock window the sim must model (the post-jump action chaos root, CLAUDE.md wave 12).
NOTE capture alignment: action(i) = the response to state(i-1); exec latency ~1 frame @10Hz."""
from __future__ import annotations
import glob, json, sys
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
    return [fs for _, fs in sorted(eps.items()) if len(fs) > 30]

jump_stats = defaultdict(lambda: [0, 0])   # delay-frames -> [fired, attempts]
slide_stats = defaultdict(lambda: [0, 0])
paths = [p for g in (sys.argv[1:] or ["captures/ab_v70*.jsonl", "captures/ab_v66_wave9*.jsonl",
                                      "captures/ab_v66_fresh17a.jsonl", "captures/best_run_11*.jsonl"])
         for p in sorted(glob.glob(g))]
for path in paths:
    for ep in load_eps(path):
        last_land = None
        for i in range(1, len(ep) - 2):
            prev, fr = ep[i - 1], ep[i]
            if prev.get("is_jumping") and not fr.get("is_jumping") and not fr.get("is_dead"):
                last_land = i
            if last_land is None:
                continue
            delay = i - last_land
            if delay > 6:
                continue
            act = fr.get("action")
            grounded = not fr.get("is_jumping") and not fr.get("is_sliding")
            if act == 2 and grounded:
                fired = bool(ep[i + 1].get("is_jumping") or ep[i + 2].get("is_jumping"))
                # context split: on/near a gap cell (where the lip rule refuses) vs solid ground
                on_gap = fr.get("path_type") in (11, 12, 13, 14, 15, 16, 33, 34, 35, 36) \
                    or fr.get("next_wide_gap")
                key = (delay, "gap" if on_gap else "solid")
                jump_stats[key][1] += 1
                jump_stats[key][0] += 1 if fired else 0
            elif act == 3 and not fr.get("is_sliding") and not fr.get("is_jumping"):
                fired = bool(ep[i + 1].get("is_sliding") or ep[i + 2].get("is_sliding"))
                slide_stats[delay][1] += 1
                slide_stats[delay][0] += 1 if fired else 0

print("JUMP-after-landing fire rate by delay (frames @10Hz since touchdown):")
for d in sorted(jump_stats):
    f, n = jump_stats[d]
    print(f"  delay {d}: fired {f}/{n} = {100*f/max(n,1):.0f}%")
print("SLIDE-after-landing fire rate by delay:")
for d in sorted(slide_stats):
    f, n = slide_stats[d]
    print(f"  delay {d}: fired {f}/{n} = {100*f/max(n,1):.0f}%")
