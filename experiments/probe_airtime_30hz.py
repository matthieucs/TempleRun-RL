#!/usr/bin/env python3
"""Definitive device airtime at 30Hz stream resolution (the 10Hz probe's 7 frames had a +/-1
start-edge ambiguity => airtime in [0.70,0.80]s; this disambiguates: 21 frames=0.70s vs 24=0.80s).

Reuses TempleRunEnv for the (flaky) reset+nav, but taps env._state (the StateClient background
reader, which increments _seq per 30Hz frame) directly during the airborne window instead of the
10Hz step() cadence. Sends ONE jump on a safe straight, then samples every new _seq frame until the
runner lands; records the is_jumping run-length + vel_y arc at 30Hz."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

JUMP, NOP = 2, 6


def safe_to_jump(st):
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


def sample_airborne(env, budget=1.6):
    """After a jump byte was sent, capture every new 30Hz frame until we've seen airborne then landed."""
    frames = []           # (is_jumping, vel_y, world_y, ground_height)
    _, seq = env._state.latest()
    t_end = time.monotonic() + budget
    seen_air = False
    while time.monotonic() < t_end:
        st, sq = env._state.latest()
        if sq != seq:
            seq = sq
            j = bool(st.get("is_jumping"))
            frames.append((j, float(st.get("vel_y") or 0.0),
                           float(st.get("world_y") or 0.0), float(st.get("ground_height") or 0.0)))
            if j:
                seen_air = True
            elif seen_air:
                break          # landed
        else:
            time.sleep(0.002)
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--out", default="captures/airtime_30hz.jsonl")
    a = p.parse_args()
    ports = C.device_ports(0)
    env = TempleRunEnv(host="127.0.0.1", state_port=ports["state_port"], action_port=ports["action_port"],
                       max_episode_steps=1500, reward_mode="distance", auto_connect=True,
                       freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); f = out.open("w")
    runlens = []
    try:
        for ep in range(a.episodes):
            env.reset()
            cooldown = 0.0; got = 0
            t_die = time.monotonic() + 25.0     # bail if this run drags (NoOp usually dies ~60m fast)
            last_seq = -1
            while time.monotonic() < t_die and got < 2:
                st, seq = env._state.latest()
                if st is None or seq == last_seq:
                    time.sleep(0.003); continue
                last_seq = seq
                if bool(st.get("is_dead")):
                    break
                if time.monotonic() < cooldown or bool(st.get("is_jumping")):
                    continue
                if safe_to_jump(st):
                    v = float(st.get("velocity") or 0.0)
                    env._action.send(JUMP)
                    fr = sample_airborne(env)
                    # 30Hz run-length of is_jumping True
                    rl = 0
                    for j, *_ in fr:
                        if j: rl += 1
                        elif rl: break
                    if rl > 0:
                        runlens.append(rl); got += 1
                        f.write(json.dumps({"ep": ep, "runlen30": rl, "start_v": v,
                                            "sec": round(rl / 30.0, 3), "arc": fr}) + "\n"); f.flush()
                    cooldown = time.monotonic() + 1.2
            print(f"ep {ep}: trials={got} total={len(runlens)}", flush=True)
    finally:
        f.close(); env.close()
    if runlens:
        import statistics as S
        from collections import Counter
        print(f"\nAIRTIME @30Hz: n={len(runlens)} median={S.median(runlens)} mean={S.mean(runlens):.2f} "
              f"min={min(runlens)} max={max(runlens)}  dist={dict(sorted(Counter(runlens).items()))}")
        print(f"  => median airtime = {S.median(runlens)/30.0:.3f}s   (sim jump_duration=0.80s;  0.70s=21fr, 0.80s=24fr)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
