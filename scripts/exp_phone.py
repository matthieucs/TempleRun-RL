#!/usr/bin/env python3
"""Phone experiment: run a model at several turn-cooldown values, log distance + CLASSIFY each death.

Death classes (from the frames just before distance freezes):
  turn     - is_turn was True / a TL|TR was issued near the end (junction death)
  obstacle - must_jump/must_slide / obstacle!=0 near the end
  zone     - path_level changed in the last few frames (altitude transition crevasse)
  unknown  - none of the above

    python scripts/exp_phone.py runs/ppo_hist4.zip --cooldowns 2,3,4 --eps 6
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import templerun_rl.constants as C, templerun_rl.obs as O  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

ANM = {0: "SL", 1: "SR", 2: "JMP", 3: "SLD", 4: "TL", 5: "TR", 6: "NOP"}


def classify(hist):
    """hist: list of (dist, action, is_turn, must_jump, must_slide, obstacle, path_level). Look at the
    last frames where distance still advanced (the real death region), not the frozen tail."""
    if not hist:
        return "unknown"
    dfin = hist[-1][0]
    region = [h for h in hist if h[0] >= dfin - 12][-10:]   # last ~12 distance of real motion
    acts = {ANM[h[1]] for h in region}
    if any(h[2] for h in region) or "TL" in acts or "TR" in acts:
        return "turn"
    if any(h[3] or h[4] or (h[5] not in (0, None)) for h in region):
        return "obstacle"
    lvls = [h[6] for h in region if h[6] is not None]
    if len(set(lvls)) > 1:
        return "zone"
    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--cooldowns", default="2,3,4")
    p.add_argument("--eps", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=2000)
    a = p.parse_args()
    m = PPO.load(a.model)
    side = Path(str(a.model).replace(".zip", "") + ".json")
    if side.exists():
        d = json.loads(side.read_text()); fs, K = int(d.get("frame_stack", 1)), int(d.get("action_history_len", 0))
    else:
        fs, K = 1, max(0, (int(m.observation_space.shape[0]) - O.OBS_LEN) // 7)
    ports = C.device_ports(0)
    for cd in [int(x) for x in a.cooldowns.split(",")]:
        env = TempleRunEnv(host="127.0.0.1", state_port=ports["state_port"], action_port=ports["action_port"],
                           max_episode_steps=a.max_steps, reward_mode="distance", auto_connect=True,
                           frame_stack=fs, action_history_len=K, turn_cooldown=cd,
                           freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0)
        dists, deaths = [], Counter()
        for ep in range(a.eps):
            o, i = env.reset(); st = i["raw_state"]; d0 = float(st.get("distance", 0) or 0); hist = []
            degraded = False
            while True:
                act, _ = m.predict(o, deterministic=True); act = int(act)
                o, r, t, tr, info = env.step(act); st = info["raw_state"]
                if info.get("degraded"): degraded = True; break
                hist.append((round(float(st.get("distance", 0) or 0)), act, int(st.get("is_turn", 0)),
                             int(st.get("must_jump", 0)), int(st.get("must_slide", 0)),
                             st.get("obstacle"), st.get("path_level")))
                if t or tr: break
            prog = (float(st.get("distance", 0) or 0) - d0)
            if degraded or prog < 5: continue
            dists.append(prog); deaths[classify(hist)] += 1
        if dists:
            print(f"cooldown={cd}: runs={len(dists)} median={np.median(dists):.0f} best={max(dists):.0f} "
                  f"mean={np.mean(dists):.0f}  deaths_by_cause={dict(deaths)}", flush=True)
        else:
            print(f"cooldown={cd}: NO valid runs (device frozen?)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
