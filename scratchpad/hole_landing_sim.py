#!/usr/bin/env python3
"""WAVE 14: SIM-side mirror of hole_landing_forensic.py — roll a policy in the CPU sim
(device-true cfg, matched capture semantics) and run the SAME `d` (cells takeoff->hole) bucket
analysis. Device says: single hole d<=1.0 -> 0/858 deaths, d>1.0 -> 6/20 (30%); long hole
d>=2.5 -> 6/6 (100%). If the sim shows ~0% in the d>1 buckets, the airborne-trap is a FALSE
AFFORDANCE in-sim (the policy is never punished for arriving airborne) -> that is the gap.

Usage: python scratchpad/hole_landing_sim.py <model.zip> [--episodes 40] [--config '{"k":v}']
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scratchpad"))
import gymnasium as gym, templerun_rl  # noqa: F401,E402
from stable_baselines3 import PPO      # noqa: E402
import templerun_rl.obs as O           # noqa: E402
import templerun_rl.constants as C     # noqa: E402
from hole_landing_forensic import analyse  # noqa: E402  (same-dir import)

DEVICE_TRUE_CFG = {
    "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True,
    "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1,
    "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "domain_rand": False,
    "curriculum_max_start_dist": 0.0, "stepup_clip_band": 0.25,
    "wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
    "crevasse_apex_all": True, "turn_send_lat_frames": 0.4,
    "post_land_gap_rejump_prob": 0.46,
    # MUST be inside the config dict -- passing these as gym.make kwargs silently leaves the cfg
    # defaults (turn_approach_buffer False), which invalidates v78's pre-swipe -> dies at the first
    # junction (~100m). Mirrors eval_ckpt_torch.DEVICE_TRUE_CFG.
    "turn_approach_buffer": True, "turn_linger_frames": 3,
}


def infer_k_fs(obs_dim):
    for fs in (1, 2, 3, 4):
        rem = obs_dim - fs * O.OBS_LEN
        if rem >= 0 and rem % C.N_ACTIONS == 0 and rem // C.N_ACTIONS <= 8:
            return rem // C.N_ACTIONS, fs
    raise ValueError(f"cannot infer K/frame_stack from obs dim {obs_dim}")


def rollout(model_path, episodes, cfg, seed0=7000, max_steps=8000):
    model = PPO.load(model_path, device="cpu")
    K, FS = infer_k_fs(int(model.observation_space.shape[0]))
    # ⚠️ CAP [fixed 2026-07-27 night]: this harness used the env default max_episode_steps AND a
    # hard `step < 4000`, so a good policy in an easy world SATURATED and every median came back as
    # the same censored number (2607 m). That silently made "v104 in the v103 world = 2607" look like
    # a real distance and turned a difficulty comparison into a comparison of two ceilings. Matches
    # gate_battery's 8000 default.
    env = gym.make("TempleRunSim-v0", config=cfg, action_history_len=K, frame_stack=FS,
                   turn_approach_buffer=True, turn_linger_frames=3).unwrapped
    # NOTE: passing max_episode_steps through gym.make() is SILENTLY IGNORED here (the registered
    # env keeps its own 1000-step default => every distance censored at ~2607 m). Set it on the
    # unwrapped env, and assert, so this can never regress into a fake ceiling again.
    env.max_episode_steps = int(max_steps)
    assert env.max_episode_steps == int(max_steps)
    rows = []
    for epi in range(episodes):
        obs, info = env.reset(seed=seed0 + epi)
        done, step = False, 0
        while not done and step < max_steps:
            act = int(model.predict(obs, deterministic=True)[0])
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            s = info["raw_state"]
            _cell = env._track[env._cell_index] if env._cell_index < len(env._track) else {}
            rows.append({"ep": epi, "step": step, "action": act,
                         "obstacle": int(_cell.get("obstacles", 0)),
                         "wide_gap": bool(_cell.get("wide_gap")),
                         "path_type": int(s.get("path_type") or 0),
                         "cell_progress": float(s.get("cell_progress") or 0.0),
                         "is_jumping": bool(s.get("is_jumping")),
                         "velocity": float(s.get("velocity") or 0.0),
                         "distance": float(s.get("distance") or 0.0),
                         "next_wide_gap": bool(s.get("next_wide_gap")),
                         "is_sliding": bool(s.get("is_sliding")),
                         # the SLIDE ladder, so a takeoff can be tested against slide hazards the
                         # same way landing_hazard tests it against jump hazards. Absent before
                         # 2026-07-28, which silently made every slide-arc probe read zero.
                         "must_slide": bool(s.get("must_slide")),
                         "next_must_slide": bool(s.get("next_must_slide")),
                         "next2_must_slide": bool(s.get("next2_must_slide")),
                         "next3_must_slide": bool(s.get("next3_must_slide")),
                         "death_type": s.get("death_type"),
                         # fields wasted_jump.py needs (post-turn context + the arc lookahead)
                         "is_turn": bool(s.get("is_turn")),
                         "is_stumbling": bool(s.get("is_stumbling")),
                         "path_level": int(s.get("path_level") or 2),
                         "must_jump": bool(s.get("must_jump")),
                         "next_must_jump": bool(s.get("next_must_jump")),
                         "next2_must_jump": bool(s.get("next2_must_jump")),
                         "next3_must_jump": bool(s.get("next3_must_jump")),
                         "is_dead": bool(s.get("is_dead")),
                         "death_type": s.get("death_type") or 0,
                         "world_y": 0.0})
            step += 1
    return rows


class _Fake:
    """Adapter so hole_landing_forensic.analyse() can read in-memory sim rows as if they were a
    capture file: it groups by 'ep' and expects the same field names."""
    def __init__(self, rows):
        self.rows = rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--config", default="")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    cfg = dict(DEVICE_TRUE_CFG)
    if a.config:
        cfg.update(json.loads(a.config))
    rows = rollout(a.model, a.episodes, cfg)
    # write a temp capture so analyse() (file-based) can consume it unchanged
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (a.tag or "run"))
    out = str(Path(__file__).resolve().parent / f"_sim_holes_{safe}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    analyse([out], f"SIM {a.tag or a.model.split('/')[-1]} ({a.episodes} eps) cfg+{a.config or '{}'}")


if __name__ == "__main__":
    main()
