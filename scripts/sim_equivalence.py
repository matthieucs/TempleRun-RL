#!/usr/bin/env python3
"""CPU-sim vs torch-sim equivalence harness — the gate for ``sim_env_torch.py``.

The CPU sim (``sim_env.py``) is the device-calibrated reference; the torch port must do
THE SAME THING. Two tiers:

1. DETERMINISTIC (exact): both sims get the SAME fixed track (every mechanic scenario) and
   the SAME action stream (scripted policies incl. the train_sim oracle), with every
   stochastic knob zeroed. Every step must match field-for-field: raw state, obs vector,
   reward, termination, death_type. Any mismatch = FAIL with the first divergence printed.
   Run over a matrix of config variants (hook buffer, approach buffer, linger, exec
   latencies, shaping, K/frame-stack).

2. STATISTICAL (procedural): same full-training config, oracle policy, N episodes in each
   sim (their RNG streams differ by construction) — median distance, death mix, and
   per-frame event rates must agree within bands.

    python scripts/sim_equivalence.py             # both tiers
    python scripts/sim_equivalence.py --det-only
    python scripts/sim_equivalence.py --stats-eps 150
Exit code 0 = all pass.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

import templerun_rl.constants as C  # noqa: E402
from templerun_rl.sim_env import TempleRunSimEnv, _element  # noqa: E402
from templerun_rl.sim_env_torch import TorchSimGymAdapter  # noqa: E402
import train_sim  # noqa: E402  (the oracle — shared driver for both sims)

FLOAT_KEYS = {"velocity", "distance", "cell_progress", "lane", "stumble_kill_timer", "score"}
ATOL = {"velocity": 1e-6, "distance": 1e-6, "cell_progress": 1e-9, "lane": 1e-9,
        "stumble_kill_timer": 1e-9, "score": 0.5}   # score is int-truncated from a float64 sum


# --------------------------------------------------------------------- tracks
def cell(**kw) -> dict:
    e = _element(type=kw.pop("type", 1), level=kw.pop("level", 2),
                 obstacles=kw.pop("obstacles", 0), is_turn=kw.pop("is_turn", False),
                 can_turn_left=kw.pop("can_turn_left", False),
                 can_turn_right=kw.pop("can_turn_right", False),
                 is_coin=kw.pop("is_coin", False))
    e["wide_gap"] = kw.pop("wide_gap", False)
    e["stepup"] = kw.pop("stepup", False)
    e["narrow_lane"] = kw.pop("narrow_lane", None)
    assert not kw, kw
    return e


def S(n=1):
    return [cell() for _ in range(n)]


def scenario_tracks() -> dict:
    t = {}
    t["holes"] = S(4) + [cell(type=11, obstacles=1)] + S(5) + [cell(type=11, obstacles=1)] + S(5) \
                 + [cell(type=11, obstacles=1)] + S(4)
    t["treejump_stumble"] = S(4) + [cell(obstacles=4)] + S(5) + [cell(obstacles=3)] + S(5) \
                            + [cell(obstacles=4)] + S(4)
    t["slides"] = S(4) + [cell(obstacles=5)] + S(5) + [cell(obstacles=6)] + S(5) \
                  + [cell(obstacles=5)] + S(4)
    t["wide_hole"] = S(5) + [cell(type=12, obstacles=1, wide_gap=True),
                             cell(type=12, obstacles=1, wide_gap=True)] + S(6) \
                     + [cell(type=12, obstacles=1, wide_gap=True),
                        cell(type=12, obstacles=1, wide_gap=True)] + S(5)
    t["crevasse_up"] = S(6) + [cell(type=35, obstacles=7, wide_gap=True, level=3),
                               cell(type=35, obstacles=7, wide_gap=True, stepup=True, level=3)] + S(8)
    t["crevasse_down"] = S(6) + [cell(type=35, obstacles=7, wide_gap=True, level=1),
                                 cell(type=35, obstacles=7, wide_gap=True, level=1)] + S(8)
    t["narrow_r"] = S(4) + [cell(type=22, narrow_lane=1)] * 3 + S(4) \
                    + [cell(type=19, narrow_lane=-1)] * 2 + S(4)
    # jump->slide compound (device death @4991/3916): crevasse -> 1 clear cell -> a slide obstacle within
    # the jump airtime. Locks the airborne-at-slide death-resolution parity for the jump->slide layout.
    t["jumpslide"] = S(6) + [cell(type=35, obstacles=7, wide_gap=True, level=3),
                             cell(type=35, obstacles=7, wide_gap=True, stepup=True, level=3)] \
                     + S(1) + [cell(type=1, obstacles=5)] + S(6)
    t["jumpslide_flame"] = S(6) + [cell(type=12, obstacles=1, wide_gap=True),
                                   cell(type=12, obstacles=1, wide_gap=True)] \
                           + S(1) + [cell(type=31, obstacles=6)] + S(6)
    # jump->strafe (device death @5148/3281/2793): long-hole -> 1 clear -> a side-narrow strip; the runner
    # must strafe into the lane during/after the jump. Locks the narrow-fall death-resolution parity.
    t["jumpnarrow"] = S(6) + [cell(type=12, obstacles=1, wide_gap=True),
                              cell(type=12, obstacles=1, wide_gap=True)] \
                      + S(1) + [cell(type=19, narrow_lane=-1)] * 2 + S(6)
    # gap->gap compound (the high-ground double-jump death, 2026-07-08): long-hole -> 2 clear -> a 2nd
    # 2-cell HoleLong. A blind landing-frame re-jump lands INSIDE gap-2 (landing model dt4); a deliberate
    # jump from the lip clears. Locks the gap->gap death-resolution parity.
    t["jumpgap"] = S(6) + [cell(type=12, obstacles=1, wide_gap=True),
                           cell(type=12, obstacles=1, wide_gap=True)] \
                   + S(2) + [cell(type=12, obstacles=1, wide_gap=True),
                             cell(type=12, obstacles=1, wide_gap=True)] + S(6)
    # RHYTHM CHAIN (high-ground multi-hole metronome, ab_v62_r1 ep0 / ab_v60_r2 ep4): three 2-cell holes
    # with 2 clear cells between — land->run->jump repeated. Locks the chain death-resolution parity.
    t["jumpgap_chain"] = S(6) + ([cell(type=12, obstacles=1, wide_gap=True),
                                  cell(type=12, obstacles=1, wide_gap=True)] + S(2)) * 3 + S(4)
    # POST-TURN OBSTACLE (device 2026-07-08): a tree right on the first post-junction cell, and a 1-cell
    # hole after another junction — the turn-exit instant-reaction scenario. Locks the parity of turning
    # into an immediate obstacle (turn fired late + must_jump/must_slide next cell).
    t["turnobstacle"] = S(5) + [cell(type=3, is_turn=True, can_turn_left=True),
                                cell(type=1, obstacles=4)] + S(4) \
                        + [cell(type=3, is_turn=True, can_turn_right=True),
                           cell(type=1, obstacles=1)] + S(6)
    t["junction_l"] = S(5) + [cell(type=3, is_turn=True, can_turn_left=True)] + S(8)
    t["junction_r"] = S(5) + [cell(type=3, is_turn=True, can_turn_right=True)] + S(8)
    t["junction_t"] = S(5) + [cell(type=7, is_turn=True, can_turn_left=True,
                                   can_turn_right=True)] + S(8)
    t["junction_pair"] = S(5) + [cell(type=3, is_turn=True, can_turn_left=True)] + S(3) \
                         + [cell(type=3, is_turn=True, can_turn_right=True)] + S(6)
    t["coins"] = S(3) + [cell(is_coin=True)] + S(3) + [cell(is_coin=True)] + S(4)
    t["compound"] = S(4) + [cell(type=22, narrow_lane=1)] * 2 + [cell(type=2, obstacles=4)] \
                    + S(2) + [cell(type=19, narrow_lane=-1)] * 2 + S(5)
    t["mixed"] = (S(3) + [cell(obstacles=4)] + S(5)
                  + [cell(type=3, is_turn=True, can_turn_left=True)] + S(2)
                  + [cell(obstacles=5)] + S(4) + [cell(type=22, narrow_lane=1)] * 2 + S(3)
                  + [cell(type=12, obstacles=1, wide_gap=True),
                     cell(type=12, obstacles=1, wide_gap=True)] + S(6))
    return t


# -------------------------------------------------------------------- policies
def pol_noop():
    return lambda st, k: 6


def _oracle_isolated(mem: dict, st: dict) -> int:
    """Call train_sim._oracle with an ISOLATED memory (its module-global mem would be
    corrupted by the interleaved cpu/torch calls)."""
    saved = dict(train_sim._oracle_mem)
    train_sim._oracle_mem.clear()
    train_sim._oracle_mem.update(mem)
    a = train_sim._oracle(st)
    mem.clear()
    mem.update(train_sim._oracle_mem)
    train_sim._oracle_mem.clear()
    train_sim._oracle_mem.update(saved)
    return a


def pol_oracle():
    mem = {"cooldown": 0, "prev_dist": 1e18}
    return lambda st, k: _oracle_isolated(mem, st)


def pol_rng(seed):
    rng = np.random.default_rng(seed)
    return lambda st, k: int(rng.integers(0, 7))


def pol_script(script: dict, default=6):
    """Action = script[k] at step k, else default (step-indexed scripted actions)."""
    return lambda st, k: script.get(k, default)


def pol_turn_at_cp(aim: float, jumper=True):
    """Swipe (correct side) the first frame cp >= aim on a junction; oracle-jump otherwise."""
    done = {"n": 0}
    mem = {"cooldown": 0, "prev_dist": 1e18}
    def f(st, k):
        if st["is_turn"] and st.get("cell_progress", 0.0) >= aim and done["n"] == 0:
            done["n"] = 3
            return 4 if st["can_turn_left"] else 5
        if done["n"] > 0:
            done["n"] -= 1
        return _oracle_isolated(mem, st) if jumper else 6
    return f


def pol_double_turn():
    """Swipe twice in a row at a junction (linger / stumble path)."""
    mem = {"fired": -10}
    def f(st, k):
        if st["is_turn"] and st["can_turn_left"] and mem["fired"] < 0:
            mem["fired"] = k
            return 4
        if k == mem["fired"] + 1:
            return 4
        return 6
    return f


def pol_approach_swipe():
    """Swipe on the approach cell (next_is_turn) — approach-buffer test."""
    mem = {"fired": False}
    omem = {"cooldown": 0, "prev_dist": 1e18}
    def f(st, k):
        if not mem["fired"] and st.get("next_is_turn") and not st["is_turn"]:
            mem["fired"] = True
            return 4 if st.get("next_can_turn_left") else 5
        return _oracle_isolated(omem, st)
    return f


# --------------------------------------------------------------- env builders
DET_BASE = {
    "domain_rand": False, "turn_tail_pmax": 0.0, "jump_drop_prob": 0.0,
    "jump_refuse_strafing_prob": 0.0, "compound_prob": 0.0,
    "curriculum_max_start_dist": 0.0,
}


def make_cpu(cells, cfg, K=0, FS=1, shaping=None):
    env = TempleRunSimEnv(track_source="procedural", config=cfg, seed=0,
                          max_episode_steps=100000, reward_mode="distance",
                          action_history_len=K, frame_stack=FS, **(shaping or {}))
    env.reset(seed=0)
    track = [cell()] + [dict(c) for c in cells] + [cell() for _ in range(320)]
    env._track = track
    env._cell_index = 0
    env._dist_into_cell = 0.0
    env._frames = []
    env._act_hist = []
    st = env._snapshot()
    env._prev_state = st
    env._prev_score = int(st["score"])
    env._prev_distance = float(st["distance"])
    obs = env._build_obs(st)
    return env, obs, st


def make_torch(cells, cfg, K=0, FS=1, shaping=None):
    env = TorchSimGymAdapter(track_source=[dict(c) for c in cells], config=cfg, seed=0,
                             device="cpu", max_episode_steps=100000, reward_mode="distance",
                             action_history_len=K, frame_stack=FS, **(shaping or {}))
    obs, info = env.reset(seed=0)
    return env, obs, info["raw_state"]


# ------------------------------------------------------------------ diff core
def diff_state(sc, kc, tag, k):
    keys = set(sc) & set(kc)
    for f in sorted(keys):
        a, b = sc[f], kc[f]
        if f == "narrow_target_lane":
            a = 0 if a is None else int(a)
            b = 0 if b is None else int(b)
            if a != b:
                return f"{tag} step {k}: narrow_target_lane {a} != {b}"
            continue
        if f in FLOAT_KEYS:
            if abs(float(a) - float(b)) > ATOL.get(f, 1e-6):
                return f"{tag} step {k}: {f} cpu={a!r} torch={b!r}"
        else:
            if bool(a) != bool(b) if isinstance(a, bool) else int(a) != int(b):
                return f"{tag} step {k}: {f} cpu={a!r} torch={b!r}"
    return None


def run_pair(name, cells, cfg_over, pol_factory, K=0, FS=1, shaping=None, max_steps=200):
    cfg = dict(DET_BASE)
    cfg.update(cfg_over)
    c_env, c_obs, c_st = make_cpu(cells, cfg, K, FS, shaping)
    t_env, t_obs, t_st = make_torch(cells, cfg, K, FS, shaping)
    tag = name
    err = diff_state(c_st, t_st, tag, -1)
    if err:
        return err
    if not np.allclose(c_obs, t_obs, atol=2e-6):
        i = int(np.argmax(np.abs(np.asarray(c_obs) - np.asarray(t_obs))))
        return f"{tag} reset obs[{i}] cpu={c_obs[i]} torch={t_obs[i]}"
    pol_c, pol_t = pol_factory(), pol_factory()
    st_c, st_t = c_st, t_st
    for k in range(max_steps):
        a_c = pol_c(st_c, k)
        a_t = pol_t(st_t, k)
        if a_c != a_t:
            return f"{tag} step {k}: policies diverged (state mismatch upstream?) {a_c} != {a_t}"
        c_obs, c_r, c_term, c_trunc, c_info = c_env.step(a_c)
        t_obs, t_r, t_term, t_trunc, t_info = t_env.step(a_t)
        st_c, st_t = c_info["raw_state"], t_info["raw_state"]
        err = diff_state(st_c, st_t, tag, k)
        if err:
            return err
        if abs(c_r - t_r) > 1e-6:
            return f"{tag} step {k}: reward cpu={c_r} torch={t_r}"
        if bool(c_term) != bool(t_term) or bool(c_trunc) != bool(t_trunc):
            return f"{tag} step {k}: done cpu=({c_term},{c_trunc}) torch=({t_term},{t_trunc})"
        if not np.allclose(c_obs, t_obs, atol=2e-6):
            i = int(np.argmax(np.abs(np.asarray(c_obs) - np.asarray(t_obs))))
            return f"{tag} step {k}: obs[{i}] cpu={c_obs[i]} torch={t_obs[i]}"
        if c_term:
            dc = c_info.get("death_type")
            dt = t_info.get("death_type")
            if int(dc or 0) != int(dt or 0):
                return f"{tag} step {k}: death_type cpu={dc} torch={dt}"
            break
    return None


def deterministic_suite() -> int:
    tracks = scenario_tracks()
    shaping_on = dict(action_cost=0.02, reaction_bonus=0.1, turn_spam_penalty=0.1,
                      idle_action_penalty=0.05, redundant_action_penalty=0.12,
                      strafe_penalty=0.3, narrow_hold_bonus=0.18, aim_center_weight=1.0,
                      velocity_drop_penalty=0.02, off_center_penalty=0.03,
                      slide_jump_penalty=0.15, airborne_action_penalty=0.1, jump_hold_penalty=0.12)
    variants = {
        "base": ({}, {}),
        "hookbuf": ({"hook_turn_buffer": True}, {}),
        "appbuf": ({"turn_approach_buffer": True}, {}),
        "linger3": ({"turn_linger_frames": 3}, {}),
        "slat1": ({"strafe_exec_lat_frames": 1}, {}),
        "jlat1": ({"jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 0}, {}),
        "train_narrow": ({"narrow_lane_center": 5.0, "narrow_lane_tol": 2.0,
                          "hook_turn_buffer": True, "turn_approach_buffer": True,
                          "turn_linger_frames": 3, "strafe_exec_lat_frames": 1,
                          "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 0}, {}),
        "shaping": ({"hook_turn_buffer": True}, shaping_on),
        "clipband": ({"stepup_clip_band": 0.25}, {}),
        # wide-gap landing model ON: exercises the new arc-lands-past-gap check (spam_jump /
        # gap_jump_late policies take off short of a 2-cell hole -> the far-wall clip fires).
        "widegap_land": ({"wide_gap_landing_model": True, "wide_gap_land_grace": 0.2}, {}),
        # crevasse apex for ALL level-changes ON: step-DOWN crevasse ledge cells now get the apex check.
        "crevasse_apex": ({"crevasse_apex_all": True, "stepup_clip_band": 0.25}, {}),
        # both device-true fidelity fixes together (the v48 training regime).
        "land_and_apex": ({"wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
                           "crevasse_apex_all": True, "stepup_clip_band": 0.25}, {}),
        # high-ground cap graze — WORLD-Y model (2026-07-07): arc height vs level-aware cap top; the
        # crevasse ledge cells now graze (stumble) / deep-clip (kill) by world-Y instead of the mh-proxy.
        "crevasse_graze": ({"crevasse_graze_worldy": True, "wide_gap_landing_model": True,
                            "wide_gap_land_grace": 0.35}, {}),
    }
    policies = {
        "noop": pol_noop,
        "oracle": pol_oracle,
        "rng17": lambda: pol_rng(17),
        "rng42": lambda: pol_rng(42),
        "turn_cp30": lambda: pol_turn_at_cp(0.30),
        "turn_cp58": lambda: pol_turn_at_cp(0.58),
        "double_turn": pol_double_turn,
        "approach_swipe": pol_approach_swipe,
        "spam_jump": lambda: pol_script({}, default=2),
        "spam_slide": lambda: pol_script({}, default=3),
        "turn_on_straight": lambda: pol_script({3: 4}, default=6),
        "strafe_hold_l": lambda: pol_script({}, default=0),
        # late gap-jump: takeoff near the approach-cell end -> apex height lands in the
        # stepup graze band at low speed (exercises stepup_clip_band's stumble path).
        "gap_jump_late": lambda: (lambda st, k: 2 if (
            (st.get("next_wide_gap") and float(st.get("cell_progress") or 0) >= 0.82)
            or st.get("must_jump") or st.get("next_must_jump")) else 6),
    }
    fails, total = [], 0
    for vname, (cfg_over, shaping) in variants.items():
        for tname, cells in tracks.items():
            for pname, pf in policies.items():
                total += 1
                K, FS = (4, 1) if vname == "shaping" else (0, 1)
                err = run_pair(f"[{vname}/{tname}/{pname}]", cells, cfg_over, pf,
                               K=K, FS=FS, shaping=shaping)
                if err:
                    fails.append(err)
                    print("FAIL", err, flush=True)
    print(f"\nDETERMINISTIC: {total - len(fails)}/{total} pass", flush=True)
    return len(fails)


# ----------------------------------------------------------------- statistics
TRAIN_CFG = {
    "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True,
    "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1,
    "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "domain_rand": True,
    "turn_approach_buffer": True, "turn_linger_frames": 3,
    "curriculum_max_start_dist": 0.0,   # stats compare full-run episodes
}


def collect_stats(kind: str, n_eps: int, seed0: int):
    if kind == "cpu":
        env = TempleRunSimEnv(config=TRAIN_CFG, max_episode_steps=4000,
                              reward_mode="distance")
    else:
        env = TorchSimGymAdapter(config=TRAIN_CFG, device="cpu", max_episode_steps=4000,
                                 reward_mode="distance")
    dists, dtypes = [], Counter()
    ev = Counter()
    frames = 0
    for ep in range(n_eps):
        o, i = env.reset(seed=seed0 + ep)
        st = i["raw_state"]
        train_sim._oracle_mem["cooldown"] = 0
        train_sim._oracle_mem["prev_dist"] = 1e18
        while True:
            a = train_sim._oracle(st)
            o, r, term, trunc, info = env.step(a)
            st = info["raw_state"]
            frames += 1
            pt = int(st.get("path_type", 1) or 1)
            ev["turn"] += bool(st.get("is_turn"))
            ev["must_jump"] += bool(st.get("must_jump"))
            ev["must_slide"] += bool(st.get("must_slide"))
            ev["narrow_pt"] += 17 <= pt <= 30
            ev["cap_pt"] += 31 <= pt <= 36
            ev["wide"] += bool(st.get("next_wide_gap"))
            if term:
                dtypes[int(info.get("death_type") or 0)] += 1
                break
            if trunc:
                break
        dists.append(float(info.get("distance", st.get("distance", 0.0) or 0.0)))
    rates = {k: v / max(1, frames) for k, v in ev.items()}
    return {"median": float(np.median(dists)), "mean": float(np.mean(dists)),
            "deaths": dict(dtypes), "rates": rates, "frames": frames, "n": n_eps}


def stats_suite(n_eps: int) -> int:
    print("\nSTATISTICAL (procedural, oracle, full training cfg):", flush=True)
    a = collect_stats("cpu", n_eps, 50000)
    b = collect_stats("torch", n_eps, 90000)
    fails = 0

    def band(name, x, y, lo=0.75, hi=1.33):
        nonlocal fails
        ratio = (y + 1e-9) / (x + 1e-9)
        ok = lo <= ratio <= hi
        if not ok:
            fails += 1
        print(f"  {name:<12} cpu={x:.4g}  torch={y:.4g}  ratio={ratio:.3f}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    band("median_dist", a["median"], b["median"], 0.75, 1.33)
    band("mean_dist", a["mean"], b["mean"], 0.75, 1.33)
    for k in sorted(set(a["rates"]) | set(b["rates"])):
        band(f"rate.{k}", a["rates"].get(k, 0.0), b["rates"].get(k, 0.0), 0.7, 1.43)
    print(f"  deaths cpu={a['deaths']}  torch={b['deaths']}", flush=True)
    return fails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det-only", action="store_true")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--stats-eps", type=int, default=120)
    args = p.parse_args()
    fails = 0
    if not args.stats_only:
        fails += deterministic_suite()
    if not args.det_only:
        fails += stats_suite(args.stats_eps)
    print(f"\n{'GREEN: torch sim matches the CPU reference' if fails == 0 else f'RED: {fails} failure(s)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
