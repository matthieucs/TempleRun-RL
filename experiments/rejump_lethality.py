#!/usr/bin/env python3
"""THE diagnostic for the high-ground double-jump (2026-07-08): detect the exact lethal event — a
LANDING-FRAME RE-JUMP (a JUMP action fired within <=2 frames of landing from a prior jump, with NO
must_jump demanding it) — and measure its DEATH RATE within the next 15 frames, split by path_level
(high-ground lvl>=3 vs mid lvl2). Run on the SIM (v59, device-true cfg) and compare with the same
stat mined from DEVICE captures. Hypothesis: device kills these on high-ground, the sim does NOT
(no learning signal) -> the fix is generation-side (model the scenario), not reward-shaping.

  sim:    python experiments/rejump_lethality.py sim runs/ppo_v59_crevasse_1503232_steps.zip
  device: python experiments/rejump_lethality.py dev captures/ab_v59_r1.jsonl [more.jsonl ...]
"""
import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "src")

JUMP = 2
WIN_AFTER_LAND = 2     # re-jump = JUMP fired within this many frames of landing
DEATH_WIN = 15         # death within this many frames after the re-jump counts as caused-by-it


def new_tally():
    return defaultdict(lambda: {"events": 0, "deaths": 0})


def feed(tally, stream):
    """stream = iterable of (action:int, is_jumping:bool, must_jump:bool, level:int, dead:bool).
    Detects landing-frame re-jumps and attributes deaths within DEATH_WIN frames."""
    since_land = 99
    was_jumping = False
    pending = []          # [frames_left, level] for open re-jump windows
    for act, jmp, mj, lvl, dead in stream:
        if was_jumping and not jmp:
            since_land = 0
        elif not jmp:
            since_land = min(99, since_land + 1)
        was_jumping = jmp
        # close/advance windows
        for p in pending:
            p[0] -= 1
        if dead:
            for p in pending:
                if p[0] > 0:
                    tally[p[1]]["deaths"] += 1
                    p[0] = 0
            pending = [p for p in pending if p[0] > 0]
        else:
            pending = [p for p in pending if p[0] > 0]
        # the event: JUMP pressed just after landing — split by whether must_jump was flagged at the
        # press (the "legit-looking retry" vs the purely spurious one). v61's gate excluded mj=True;
        # if THOSE are the lethal ones, the gate excluded exactly the killer.
        if act == JUMP and since_land <= WIN_AFTER_LAND:
            key = (3 if lvl >= 3 else 2, "mj" if mj else "spur")
            tally[key]["events"] += 1
            pending.append([DEATH_WIN, key])
        if dead:
            since_land = 99
            was_jumping = False


def report(tag, tally):
    print(f"\n=== {tag}: landing-frame re-jump lethality ===")
    print(f"{'lvl':>4} {'kind':>5} {'events':>7} {'deaths<=15f':>11} {'lethal%':>8}")
    for key in sorted(tally, key=str):
        t = tally[key]
        pct = 100 * t["deaths"] / t["events"] if t["events"] else 0
        lvl, kind = key if isinstance(key, tuple) else (key, "?")
        print(f"{lvl:>4} {kind:>5} {t['events']:>7} {t['deaths']:>11} {pct:>7.1f}%")


def run_sim(model_path, n_eps=60, seed0=1000, cfg_over=None):
    import numpy as np
    from stable_baselines3 import PPO
    from templerun_rl.sim_env import TempleRunSimEnv
    import templerun_rl.obs as O
    CFG = {"jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True,
           "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1,
           "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25,
           "wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
           "crevasse_graze_worldy": True, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2,
           "turn_approach_buffer": True, "turn_linger_frames": 3, "curriculum_max_start_dist": 0.0}
    if cfg_over:
        CFG.update(cfg_over)
    m = PPO.load(model_path, device="cpu")
    K = max(0, (int(m.observation_space.shape[0]) - O.OBS_LEN) // 7)
    env = TempleRunSimEnv(config=CFG, max_episode_steps=4000, reward_mode="distance",
                          action_history_len=K, frame_stack=1)
    tally = new_tally()
    for ep in range(n_eps):
        o, i = env.reset(seed=seed0 + ep); hist = []

        def obsv():
            base = O.state_to_obs(env._snapshot())
            if K:
                h = [int(x) for x in hist[-K:]]; h = [6] * (K - len(h)) + h
                oh = np.zeros(K * 7, dtype=np.float32)
                for j, x in enumerate(h):
                    oh[j * 7 + x] = 1.0
                return np.concatenate([base, oh])
            return base
        obs = obsv(); done = False; steps = 0
        stream = []
        st = env._snapshot()
        while not done and steps < 4000:
            a, _ = m.predict(obs, deterministic=True); a = int(a)
            prev = st
            o, r, t, tr, info = env.step(a); hist.append(a); obs = obsv()
            st = env._snapshot()
            stream.append((a, bool(prev.get("is_jumping")), bool(prev.get("must_jump")),
                           int(prev.get("path_level") or 2), bool(st.get("is_dead"))))
            done = t or tr; steps += 1
        feed(tally, stream)
    env.close()
    report(f"SIM {Path(model_path).name} ({n_eps} eps)", tally)


def run_dev(paths):
    tally = new_tally()
    for path in paths:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        eps = defaultdict(list)
        for r in rows:
            eps[r.get("ep")].append(r)
        for ep, frames in eps.items():
            frames.sort(key=lambda r: r.get("step", 0))
            stream = []
            for f in frames:
                def num(x, d=0):
                    try:
                        return float(x)
                    except (TypeError, ValueError):
                        return d
                stream.append((int(num(f.get("action"), 6)), bool(f.get("is_jumping")),
                               bool(f.get("must_jump")), int(num(f.get("path_level"), 2)),
                               bool(f.get("is_dead"))))
            feed(tally, stream)
    report(f"DEVICE {[Path(p).name for p in paths]}", tally)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sim":
        over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
        run_sim(sys.argv[2], cfg_over=over)
    else:
        run_dev(sys.argv[2:])
