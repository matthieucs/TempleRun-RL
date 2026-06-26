#!/usr/bin/env python3
"""Phase-0 sim-calibration capture: drive a policy on the phone and log every frame's raw state.

Feeds the two sim-fidelity targets:
  * crevasse / level transition (path_type 33-36, obstacle==7): measure the real gap SPAN (cells) and
    the jump cell_progress that clears vs falls in (Horz 33/34 wider than Vert 35/36 — verify).
  * narrows / one-line traps (path_type 17-30): record subtype+facing+lane held + outcome to rebuild
    the safe-lane map (the hook's NARROW_SIDE guess is unverified/contradictory).

Each JSONL line = one control step: {ep, step, action, narrow_cmd, <curated raw_state fields>}. The
last frame of an episode is the death frame (is_dead/terminated). Analyze with analyze_calib.py.

The optional --narrow-override forces a commanded lane at each narrow (independent of the hook's
narrow_target_lane guess), so the sweep measures which lane ACTUALLY survives per subtype+facing.

    python scripts/capture_calib.py runs/ppo_nn_best.zip --episodes 30 --out captures/calib.jsonl
    # clean narrow sweep (try L/C/R across encounters):
    python scripts/capture_calib.py runs/ppo_nn_best.zip --episodes 40 --narrow-override cycle \
        --out captures/narrow_sweep.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import templerun_rl.constants as C  # noqa: E402
import templerun_rl.obs as O  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

# curated raw_state fields kept per frame (crevasse + narrow calibration).
LOG_FIELDS = (
    "distance", "path_type", "next_path_type", "obstacle", "path_level",
    "lane", "cell_progress", "facing", "velocity",
    "is_jumping", "is_sliding", "is_strafing", "is_dead", "is_game_over",
    "is_turn", "next_is_turn", "must_jump", "must_slide",
    "next_must_jump", "next_must_slide", "next_wide_gap", "narrow_target_lane",
)

NARROW_LO, NARROW_HI = 17, 30
# action bytes (mirror C.Action / exp_phone ANM)
SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6


def load_policy(model_path: str):
    m = PPO.load(model_path)
    side = Path(str(model_path).replace(".zip", "") + ".json")
    if side.exists():
        d = json.loads(side.read_text())
        return m, int(d.get("frame_stack", 1)), int(d.get("action_history_len", 0))
    # fall back: infer K from the obs width (7 one-hot slots per remembered action).
    K = max(0, (int(m.observation_space.shape[0]) - O.OBS_LEN) // 7)
    return m, 1, K


def lane_bucket(lane) -> str:
    """L / C / R from the signed lateral offset (same thresholds as obs.lane_to_obs)."""
    if lane is None:
        return "?"
    if float(lane) < O.LANE_LEFT_MAX:
        return "L"
    if float(lane) > O.LANE_RIGHT_MIN:
        return "R"
    return "C"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--turn-cooldown", type=int, default=0)
    p.add_argument("--out", default="captures/calib.jsonl")
    p.add_argument("--narrow-override", choices=["off", "L", "C", "R", "cycle"], default="off",
                   help="force a commanded lane at each narrow (isolates the sweep from the hook guess)")
    a = p.parse_args()

    m, fs, K = load_policy(a.model)
    ports = C.device_ports(a.device_index)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # obs adapter: ppo_nn_best was trained BEFORE narrow_target_lane was appended (commit 4c018d7),
    # so its obs is 1 narrower than the current env. The action-history block is appended AFTER the
    # OBS_LEN base, so dropping the last base index (narrow_target_lane) reconstructs the trained obs.
    model_w = int(m.observation_space.shape[0])

    def adapt(o):
        if o.shape[-1] == model_w + 1:
            return np.delete(o, O.OBS_LEN - 1, axis=-1)
        return o

    env = TempleRunEnv(
        host="127.0.0.1", state_port=ports["state_port"], action_port=ports["action_port"],
        max_episode_steps=a.max_steps, reward_mode="distance", auto_connect=True,
        frame_stack=fs, action_history_len=K, turn_cooldown=a.turn_cooldown,
        freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0,
    )

    cyc = [SL, NOP, SR]   # cycle order maps to L / C / R
    cyc_lbl = {SL: "L", NOP: "C", SR: "R"}
    n_written = 0
    narrow_count = 0   # global counter for --narrow-override cycle

    f = out.open("w")
    try:
        for ep in range(a.episodes):
            o, info = env.reset()
            st = info["raw_state"]
            # per-episode narrow-override state
            narrow_active = False
            narrow_strafed = False
            narrow_cmd = ""

            for step in range(a.max_steps):
                act, _ = m.predict(adapt(o), deterministic=True)
                act = int(act)
                cmd_log = ""

                if a.narrow_override != "off":
                    npt = int(st.get("next_path_type") or -1)
                    pt = int(st.get("path_type") or -1)
                    approaching = NARROW_LO <= npt <= NARROW_HI
                    on_narrow = NARROW_LO <= pt <= NARROW_HI
                    if approaching or on_narrow:
                        if not narrow_active:
                            narrow_active = True
                            narrow_strafed = False
                            if a.narrow_override == "cycle":
                                strafe = cyc[narrow_count % 3]
                                narrow_cmd = cyc_lbl[strafe]
                            else:
                                narrow_cmd = a.narrow_override
                            narrow_count += 1
                        # steer toward the commanded lane ONCE on approach, then hold.
                        if not narrow_strafed and approaching:
                            act = {"L": SL, "R": SR, "C": NOP}[narrow_cmd]
                            narrow_strafed = True
                        else:
                            act = NOP
                        cmd_log = narrow_cmd
                    elif narrow_active:
                        # cleared the narrow -> recenter once (opposite strafe), reset state.
                        act = {"L": SR, "R": SL, "C": NOP}[narrow_cmd]
                        narrow_active = False
                        cmd_log = "recenter"

                o, r, t, tr, info = env.step(act)
                st = info["raw_state"]
                if info.get("degraded"):
                    print(f"ep {ep}: device degraded, skipping rest", flush=True)
                    break

                rec = {"ep": ep, "step": step, "action": act, "narrow_cmd": cmd_log}
                for k in LOG_FIELDS:
                    rec[k] = st.get(k)
                rec["lane_bucket"] = lane_bucket(st.get("lane"))
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                if t or tr:
                    break

            f.flush()
            print(f"ep {ep}: dist={st.get('distance')} dead={st.get('is_dead')} "
                  f"pt={st.get('path_type')} total_frames={n_written}", flush=True)
    finally:
        f.close()
        env.close()
    print(f"wrote {n_written} frames over {a.episodes} episodes -> {out}")


if __name__ == "__main__":
    main()
