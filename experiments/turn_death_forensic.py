#!/usr/bin/env python3
"""Decode a policy capture's turn (pt-1) death: walk the last N seconds — cell sequence, can_turn
flags, every action byte, turn_write verdicts, cp/velocity — to name the exact failure mode:
(a) policy never decided a turn, (b) decided too late (dead-letter past the consume window),
(c) write refused (debounce/invalid), (d) turned the wrong way."""
from __future__ import annotations
import json, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "captures/ab_v70_r2b.jsonl"
target_ep = int(sys.argv[2]) if len(sys.argv) > 2 else None

eps = defaultdict(list)
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        eps[d.get("ep", 0)].append(d)

A = {0: "STRAFE_L", 1: "STRAFE_R", 2: "JUMP", 3: "SLIDE", 4: "TURN_L", 5: "TURN_R", 6: "-"}

for k in sorted(eps):
    fs = eps[k]
    last = fs[-1]
    if not last.get("is_dead"):
        continue
    if target_ep is not None and k != target_ep:
        continue
    print(f"\n=== ep{k}: DEATH @{float(last.get('distance') or 0):.0f} pt={last.get('path_type')} "
          f"dt={last.get('death_type')}")
    tail = fs[max(0, len(fs)-45):]
    prev = None
    for i, fr in enumerate(tail):
        ev = []
        act = fr.get("action")
        if act in (0, 1, 2, 3, 4, 5):
            ev.append(f"act={A[act]}")
        if fr.get("turn_write"):
            ev.append(f"turn_write={fr['turn_write']}")
        if prev is not None:
            if fr.get("path_type") != prev.get("path_type"):
                ev.append(f"cell pt{prev.get('path_type')}->pt{fr.get('path_type')}")
            for fkey, tag in (("can_turn_left", "canL"), ("can_turn_right", "canR"),
                              ("is_turn", "IS_TURN"), ("over_turn_area", "overTurn"),
                              ("should_turn_left", "giL"), ("should_turn_right", "giR")):
                if bool(fr.get(fkey)) != bool(prev.get(fkey)):
                    ev.append(f"{tag}={'ON' if fr.get(fkey) else 'off'}")
            if bool(fr.get("is_dead")) and not prev.get("is_dead"):
                ev.append("**DEAD**")
        if ev:
            print(f"  t-{(len(tail)-1-i)/10.0:4.1f}s d={float(fr.get('distance') or 0):7.1f} "
                  f"cp={fr.get('cell_progress') if fr.get('cell_progress') is not None else -1:5.2f} "
                  f"v={float(fr.get('velocity') or 0):5.1f} "
                  f"pt={fr.get('path_type')} | {', '.join(ev)}")
        prev = fr
