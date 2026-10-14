#!/usr/bin/env python3
"""Jump-TIMING probe (device): the calibration the jump-spam wall needs.

Documented next step (CLAUDE.md WAVE 3): the airborne_action_penalty fixed jump-spam OFFLINE
(80%->27%) but NOT on device (97%/97%). Hypothesis: a sim-to-real JUMP-TIMING gap -- the device
jump lasts longer than the sim's 0.8s AND/OR pressing JUMP while airborne QUEUES a re-jump on
landing (perpetual airborne), which the sim IGNORES (`_apply_action` `_jump_t<=0` guard makes
airborne presses free). We can't calibrate the sim until we measure the real device behaviour.

This probe, on SAFE straight cells, runs two kinds of trials and records the full is_jumping /
world_y / vel_y trace of each:
  * SINGLE : one JUMP byte, then NoOp -> measures liftoff latency + airtime of ONE jump.
  * BURST  : JUMP every frame for --burst-n frames, then NoOp -> if total airborne time exceeds
             the single-jump airtime, the device is QUEUEING re-jumps (the root cause). The ratio
             calibrates how the sim should model an airborne press.

Outputs per-trial metrics + raw traces to JSONL and prints distributions.

    python scripts/probes/probe_jump_timing.py --episodes 6 --burst-n 8 --out captures/jump_timing_probe.jsonl
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6


def safe_to_jump(st):
    """A plain straight cell with nothing that would make a jump fatal/ambiguous (from probe_jump_exec)."""
    if st.get("is_turn") or st.get("can_turn_left") or st.get("can_turn_right"):
        return False
    if st.get("must_jump") or st.get("must_slide") or st.get("next_must_jump") or st.get("next_must_slide"):
        return False
    if st.get("next_wide_gap") or st.get("next_ground_change_higher"):
        return False
    pt = int(st.get("path_type") or -1); npt = int(st.get("next_path_type") or -1)
    if pt not in (1, 2) or npt in range(11, 37):
        return False
    return True


def frame(st):
    """Snapshot the arc-relevant fields of one state."""
    return {
        "jumping": bool(st.get("is_jumping")),
        "world_y": float(st.get("world_y") or 0.0),
        "ground_height": float(st.get("ground_height") or 0.0),
        "vel_y": float(st.get("vel_y") or 0.0),
        "velocity": float(st.get("velocity") or 0.0),
        "cp": float(st.get("cell_progress") or 0.0),
    }


def analyze(trace):
    """trace = frames from the press onward. Return liftoff latency + airborne-frame count + peak height."""
    liftoff = None
    for i, fr in enumerate(trace):
        if fr["jumping"]:
            liftoff = i
            break
    if liftoff is None:
        return {"liftoff": None, "airframes": 0, "peak_h": None}
    airframes = 0
    for fr in trace[liftoff:]:
        if fr["jumping"]:
            airframes += 1
        else:
            break
    heights = [fr["world_y"] - fr["ground_height"] for fr in trace[liftoff:liftoff + airframes]]
    peak_h = max(heights) if heights else None
    return {"liftoff": liftoff, "airframes": airframes, "peak_h": peak_h}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--burst-n", type=int, default=8, help="JUMP presses in a burst trial")
    p.add_argument("--record-frames", type=int, default=30, help="max frames to record per trial")
    p.add_argument("--out", default="captures/jump_timing_probe.jsonl")
    a = p.parse_args()
    ports = C.device_ports(a.device_index)
    env = TempleRunEnv(host="127.0.0.1", state_port=ports["state_port"], action_port=ports["action_port"],
                       max_episode_steps=a.max_steps, reward_mode="distance", auto_connect=True,
                       freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); f = out.open("w")
    singles, bursts = [], []
    try:
        for ep in range(a.episodes):
            o, info = env.reset(); st = info["raw_state"]
            mode = "single"          # alternate single / burst
            trial = None; cooldown = 8
            for step in range(a.max_steps):
                act = NOP
                jumping = bool(st.get("is_jumping"))
                if trial is not None:
                    trial["trace"].append(frame(st))
                    # keep pressing during a burst
                    if trial["kind"] == "burst" and trial["burst_left"] > 0:
                        act = JMP; trial["burst_left"] -= 1
                    # completion: saw liftoff then landed, or recording budget spent
                    tr = trial["trace"]
                    seen_air = any(fr["jumping"] for fr in tr)
                    landed = seen_air and not tr[-1]["jumping"]
                    done = (trial["burst_left"] == 0 and landed) or len(tr) >= a.record_frames
                    if done:
                        m = analyze(tr)
                        rec = {"ep": ep, "kind": trial["kind"], "n_presses": trial["n_presses"],
                               "start_v": trial["start_v"], **m, "trace": tr}
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        (singles if trial["kind"] == "single" else bursts).append(rec)
                        trial = None; cooldown = 12; mode = "burst" if mode == "single" else "single"
                elif cooldown > 0:
                    cooldown -= 1
                elif not jumping and safe_to_jump(st):
                    n = 1 if mode == "single" else a.burst_n
                    act = JMP
                    trial = {"kind": mode, "burst_left": n - 1, "n_presses": n,
                             "start_v": float(st.get("velocity") or 0.0), "trace": []}
                o, r, t, trc, info = env.step(act); st = info["raw_state"]
                if info.get("degraded"):
                    print(f"ep {ep}: DEGRADED at step {step}", flush=True); break
                if t or trc:
                    break
            print(f"ep {ep}: dist={st.get('distance')} singles={len(singles)} bursts={len(bursts)}", flush=True)
    finally:
        f.close(); env.close()

    import statistics as S
    def summ(rows, label):
        rows = [r for r in rows if r["liftoff"] is not None and r["airframes"] > 0]
        if not rows:
            print(f"\n{label}: no valid trials"); return
        lift = [r["liftoff"] for r in rows]
        air = [r["airframes"] for r in rows]
        pk = [r["peak_h"] for r in rows if r["peak_h"] is not None]
        print(f"\n{label} (n={len(rows)}):")
        print(f"  liftoff latency (frames@10Hz): median={S.median(lift)} mean={S.mean(lift):.2f} "
              f"min={min(lift)} max={max(lift)}")
        print(f"  AIRBORNE duration (frames@10Hz): median={S.median(air)} mean={S.mean(air):.2f} "
              f"min={min(air)} max={max(air)}  (~{S.median(air)*0.1:.2f}s;  sim=0.8s=8fr)")
        if pk:
            print(f"  peak height (world_y-ground): median={S.median(pk):.2f} max={max(pk):.2f}")
    summ(singles, "SINGLE jump")
    summ(bursts, f"BURST x{a.burst_n}")
    sv = [r["airframes"] for r in singles if r["liftoff"] is not None and r["airframes"] > 0]
    bv = [r["airframes"] for r in bursts if r["liftoff"] is not None and r["airframes"] > 0]
    if sv and bv:
        rs, rb = S.median(sv), S.median(bv)
        print(f"\n*** QUEUEING TEST: burst airborne / single airborne = {rb}/{rs} = {rb/rs:.2f}x")
        print(f"    ({'QUEUES re-jumps (perpetual airborne) -> confirms root cause' if rb > rs*1.4 else 'no extension -> device does NOT queue; spam is free like the sim'})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
