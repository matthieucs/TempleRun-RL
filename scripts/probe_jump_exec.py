#!/usr/bin/env python3
"""Jump-EXECUTION latency probe: on SAFE straight cells (no turn/obstacle/crevasse), send ONE JUMP byte and
count how many state-frames until is_jumping flips true. That k = the device jump execution latency (in
10Hz policy-frames) that the sim's jump_exec_lat_frames should match. Report the DISTRIBUTION (median = the
mean latency, spread = the jitter). The v25 device eval showed 2±1 was too big (jumps too early) -> measure
the real number instead of guessing.

    python scripts/probe_jump_exec.py --episodes 6 --out captures/jump_exec_probe.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6


def safe_to_jump(st):
    """A plain straight cell with nothing that would make a jump fatal/ambiguous."""
    if st.get("is_turn") or st.get("can_turn_left") or st.get("can_turn_right"):
        return False
    if st.get("must_jump") or st.get("must_slide") or st.get("next_must_jump") or st.get("next_must_slide"):
        return False
    if st.get("next_wide_gap") or st.get("next_ground_change_higher"):
        return False
    pt = int(st.get("path_type") or -1); npt = int(st.get("next_path_type") or -1)
    if pt not in (1, 2) or npt in range(11, 37):   # only pure straights, no hole/narrow/crevasse ahead
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--out", default="captures/jump_exec_probe.jsonl")
    a = p.parse_args()
    ports = C.device_ports(a.device_index)
    env = TempleRunEnv(host="127.0.0.1", state_port=ports["state_port"], action_port=ports["action_port"],
                       max_episode_steps=a.max_steps, reward_mode="distance", auto_connect=True,
                       freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); f = out.open("w")
    ks = []
    try:
        for ep in range(a.episodes):
            o, info = env.reset(); st = info["raw_state"]
            sent_step = None; sent_cp = None; sent_v = None; cooldown = 0
            for step in range(a.max_steps):
                act = NOP
                jumping = bool(st.get("is_jumping"))
                if sent_step is not None:
                    # waiting for the flip
                    if jumping:
                        k = step - sent_step
                        ks.append(k)
                        f.write(json.dumps({"ep": ep, "k_frames": k, "sent_cp": sent_cp,
                                            "velocity": sent_v, "step": step}) + "\n")
                        sent_step = None; cooldown = 12   # wait for landing before next probe
                    elif step - sent_step > 6:
                        sent_step = None; cooldown = 6    # never flipped (missed) -> reset
                elif cooldown > 0:
                    cooldown -= 1
                elif not jumping and safe_to_jump(st):
                    act = JMP; sent_step = step; sent_cp = float(st.get("cell_progress") or 0.0)
                    sent_v = float(st.get("velocity") or 0.0)
                o, r, t, tr, info = env.step(act); st = info["raw_state"]
                if info.get("degraded"):
                    print(f"ep {ep}: degraded", flush=True); break
                if t or tr:
                    break
            f.flush(); print(f"ep {ep}: dist={st.get('distance')} jumps_measured={len(ks)}", flush=True)
    finally:
        f.close(); env.close()
    if ks:
        import statistics as S
        ks_sorted = sorted(ks)
        print(f"\nJUMP EXEC LATENCY (frames@10Hz): n={len(ks)} min={min(ks)} median={S.median(ks)} "
              f"mean={S.mean(ks):.2f} max={max(ks)} p10={ks_sorted[len(ks)//10]} p90={ks_sorted[9*len(ks)//10]}")
        from collections import Counter
        print("distribution:", dict(sorted(Counter(ks).items())))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
