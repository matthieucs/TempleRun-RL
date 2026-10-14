#!/usr/bin/env python3
"""Analyze captures/jump_timing_probe.jsonl: characterize single vs burst airtime, the queueing
mechanic, liftoff latency, and the world_y arc. Prints example is_jumping run-strings so the exact
sim jump-queue model (one re-jump vs a chain) can be read off the device behaviour."""
import json, sys, statistics as S
from pathlib import Path
from collections import Counter

path = Path(sys.argv[1] if len(sys.argv) > 1 else "captures/jump_timing_probe.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
singles = [r for r in rows if r["kind"] == "single" and r.get("liftoff") is not None and r["airframes"] > 0]
bursts = [r for r in rows if r["kind"] == "burst" and r.get("liftoff") is not None and r["airframes"] > 0]


def runstring(trace):
    """Compact is_jumping timeline: ^=liftoff-onset frames until first True, then J for airborne, .=ground."""
    return "".join("J" if f["jumping"] else "." for f in trace)


def report(rows, label):
    if not rows:
        print(f"\n{label}: none"); return
    lift = [r["liftoff"] for r in rows]
    air = [r["airframes"] for r in rows]
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"  liftoff latency frames: median={S.median(lift)} dist={dict(sorted(Counter(lift).items()))}")
    print(f"  airborne frames:        median={S.median(air)} mean={S.mean(air):.2f} "
          f"dist={dict(sorted(Counter(air).items()))}")
    print(f"  airborne seconds @10Hz: median={S.median(air)*0.1:.2f}s   (sim jump_duration=0.8s=8fr)")
    pk = [r["peak_h"] for r in rows if r.get("peak_h") is not None]
    if pk:
        print(f"  peak height (world_y-ground): median={S.median(pk):.2f}")
    print(f"  --- example timelines (is_jumping; J=airborne .=ground) ---")
    for r in rows[:6]:
        print(f"    v={r['start_v']:6.1f} lat={r['liftoff']} air={r['airframes']:2d}  {runstring(r['trace'])}")


report(singles, "SINGLE (1 press)")
report(bursts, f"BURST ({bursts[0]['n_presses'] if bursts else '?'} presses)")

if singles and bursts:
    rs, rb = S.median([r["airframes"] for r in singles]), S.median([r["airframes"] for r in bursts])
    print(f"\n*** QUEUEING RATIO: burst/single airborne = {rb}/{rs} = {rb/rs:.2f}x")
    if rb > rs * 1.4:
        extra = rb - rs
        print(f"    -> device QUEUES re-jumps: a {bursts[0]['n_presses']}-press burst adds ~{extra} airborne "
              f"frames (~{extra/max(rs,1):.1f} extra jump(s)). Model: airborne JUMP buffers a re-jump on landing.")
    else:
        print("    -> NO extension: device does NOT queue; airborne presses are dropped (like the sim). "
              "Spam is free -> the wall is elsewhere (obs staleness / is_jumping latency).")
