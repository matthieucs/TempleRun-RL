#!/usr/bin/env python3
"""Decode the HIGH-GROUND redundant-jump death (USER-observed on v59, 2026-07-08): on high-ground the
policy jumps AGAIN right after jumping an obstacle -> jumps into its death. Only on high-ground.
For each death in a capture, print the last ~25 frames (action byte, is_jumping, ground_height,
path_type, cell_progress, world_y, vel_y) so we can see the DOUBLE-JUMP + the raised-ledge geometry."""
import json, sys
from pathlib import Path
from collections import defaultdict

path = Path(sys.argv[1] if len(sys.argv) > 1 else "captures/ab_v59_r1.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
eps = defaultdict(list)
for r in rows:
    eps[r.get("ep")].append(r)

ACT = {0: "SL", 1: "SR", 2: "JMP", 3: "SLD", 4: "TL", 5: "TR", 6: "-"}


def num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


for ep, frames in sorted(eps.items()):
    frames.sort(key=lambda r: r.get("step", 0))
    last = frames[-1]
    dt = last.get("death_type"); pt = int(num(last.get("path_type"), -1))
    gh_last = num(last.get("ground_height"))
    hi = gh_last > 0.5 or any(num(f.get("ground_height")) > 0.5 for f in frames[-10:])
    tag = "  <<< HIGH-GROUND" if hi else ""
    print(f"\n===== ep{ep}: death dist={num(last.get('distance')):.0f} death_type={dt} pt={pt} "
          f"ground_height={gh_last:.0f}{tag} =====")
    print(f"  {'step':>5} {'act':>4} {'jmp':>3} {'gh':>4} {'pt':>4} {'cp':>5} {'wy':>7} {'vy':>7} {'stmbl':>5}")
    for f in frames[-25:]:
        j = "J" if f.get("is_jumping") else "."
        s = "S" if f.get("is_stumbling") else " "
        print(f"  {int(num(f.get('step'))):>5} {ACT.get(int(num(f.get('action'),6)),'?'):>4} {j:>3} "
              f"{num(f.get('ground_height')):>4.0f} {int(num(f.get('path_type'),-1)):>4} "
              f"{num(f.get('cell_progress')):>5.2f} {num(f.get('world_y')):>7.1f} {num(f.get('vel_y')):>7.1f} {s:>5}")
