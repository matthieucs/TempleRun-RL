"""Prove `type_reject_model: False` reproduces the FROZEN baseline sim exactly.

Loads backups/sim_baseline_2026-07-28_v111/src/sim_env.py as a second module inside the
templerun_rl package (so its `from . import constants/obs/reward` resolve to the same modules the
working copy uses) and compares, seed by seed:

  * the GENERATED TRACK, cell field by cell field (this is what the sampler rewrite could move);
  * a full oracle-driven episode: every raw state field, reward, obs vector, death type, distance;
  * the rng draw count, via the shared-sequence proxy from cpp_sim/test_rng_order.py.

A single differing value anywhere fails. Nothing here is statistical.
"""
import importlib.util
import sys
from pathlib import Path

# backups/ is read-only by policy: loading a module from there would otherwise drop a
# __pycache__/*.pyc next to the frozen source.
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import templerun_rl  # noqa: E402
from templerun_rl.sim_env import TempleRunSimEnv as NewEnv  # noqa: E402
from cpp_sim.test_rng_order import SeqRNG  # noqa: E402
import train_sim  # noqa: E402

_p = ROOT / "backups" / "sim_baseline_2026-07-28_v111" / "src" / "sim_env.py"
_spec = importlib.util.spec_from_file_location("templerun_rl._baseline_sim_env", _p)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["templerun_rl._baseline_sim_env"] = _mod
_mod.__package__ = "templerun_rl"
_spec.loader.exec_module(_mod)
OldEnv = _mod.TempleRunSimEnv

# The v111 champion's training world, i.e. the config the frozen baseline actually produced runs
# in -- not bare defaults. Everything the sampler can touch is ON.
CFG = {
    "device_difficulty_ramp": True, "double_obstacle_percent": 0.15,
    "dist_per_velsec_slope": 8.47e-4,
    "turn_prob_late": 0.2, "min_clear_after_turn": 3,
    "tight_pair_prob": 0.05, "tight_pair_min_cells": 2,
    "crevasse_single_cell_prob": 0.675,
    "turnobstacle_prob": 0.08, "turnobstacle3_prob": 0.30,
    "compound_prob": 0.12, "compound_after_distance": 0.0,
    "jumpslide_prob": 0.20, "jumpnarrow_prob": 0.15, "jumpgap_prob": 0.15,
    "slidegap_prob": 0.15, "jumpgap_chain_max": 3,
    "narrows_after_distance": 0.0, "zone_after_distance": 0.0,
    "pair_same_type_prob": 0.6, "pair_same_type_cells": 4,
    "hook_turn_buffer": True, "turn_approach_buffer": True, "turn_linger_frames": 3,
    "turn_aware_lookahead": True, "stepdown_grounded_ok": True,
    "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 0,
    "held_jump_rejump_solid": 0.10, "held_jump_rejump_gap": 0.28,
    "post_land_gap_rejump_prob": 0.46,
    "wide_gap_landing_model": True, "hole_landing_model": True,
    "crevasse_apex_all": True, "crevasse_graze_worldy": True, "stepup_clip_band": 0.25,
    "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2,
    "jump_dur_stepdown_scale": 1.17,
    "domain_rand": True, "curriculum_max_start_dist": 3000.0,
}
CELL_FIELDS = ("type", "level", "obstacles", "is_turn", "can_turn_left", "can_turn_right",
               "is_coin", "wide_gap", "stepup", "stepdown_low", "narrow_lane")


def track_stream(EnvCls, seed, n_cells, cfg):
    """Generated cells only -- drive _next_cell() directly (spec §5)."""
    env = EnvCls(config=cfg, seed=seed, max_episode_steps=100000, reward_mode="distance")
    env.reset(seed=seed)
    env._distance_real = 3000.0
    return [tuple(env._next_cell().get(f) for f in CELL_FIELDS) for _ in range(n_cells)]


def episode_stream(EnvCls, vals, steps, cfg):
    env = EnvCls(config=cfg, seed=0, max_episode_steps=100000, reward_mode="distance",
                 action_history_len=4, frame_stack=1)
    env._rng = SeqRNG(vals)
    o, i = env.reset()
    st = i["raw_state"]
    out = [(dict(st), 0.0, np.asarray(o, dtype=float).copy(), env._rng.i)]
    train_sim._oracle_mem.clear()
    train_sim._oracle_mem.update({"cooldown": 0, "prev_dist": 1e18})
    for _ in range(steps):
        a = train_sim._oracle(st)
        o, r, term, trunc, inf = env.step(a)
        st = inf["raw_state"]
        out.append((dict(st), float(r), np.asarray(o, dtype=float).copy(), env._rng.i))
        if term or trunc:
            out.append(({"death_type": int(inf.get("death_type") or 0)}, 0.0, np.zeros(1), 0))
            break
    return out


def main():
    fails = 0
    for seed in range(40):
        a = track_stream(OldEnv, seed, 4000, CFG)
        b = track_stream(NewEnv, seed, 4000, CFG)
        if a != b:
            k = next(j for j in range(len(a)) if a[j] != b[j])
            print(f"FAIL track seed={seed} cell {k}: baseline={a[k]} new={b[k]}")
            fails += 1
    print(f"TRACK STREAM      : {40 - fails}/40 seeds byte-identical over 4000 generated cells each")

    ef = 0
    for ep in range(40):
        vals = np.random.default_rng(7000 + ep).random(400000)
        A = episode_stream(OldEnv, vals, 600, CFG)
        B = episode_stream(NewEnv, vals, 600, CFG)
        if len(A) != len(B):
            print(f"FAIL episode {ep}: length {len(A)} vs {len(B)}")
            ef += 1
            continue
        bad = None
        for k, ((sa, ra, oa, da), (sb, rb, ob, db)) in enumerate(zip(A, B)):
            if da != db:
                bad = f"step {k}: draw count {da} vs {db}"
            elif abs(ra - rb) > 0:
                bad = f"step {k}: reward {ra} vs {rb}"
            elif oa.shape != ob.shape or not np.array_equal(oa, ob):
                bad = f"step {k}: obs differs"
            else:
                for f in sorted(set(sa) | set(sb)):
                    x, y = sa.get(f), sb.get(f)
                    if (x is None) != (y is None) or (x is not None and x != y):
                        bad = f"step {k}: {f} {x!r} vs {y!r}"
                        break
            if bad:
                break
        if bad:
            print(f"FAIL episode {ep}: {bad}")
            ef += 1
    print(f"EPISODE STREAM    : {40 - ef}/40 episodes identical "
          f"(raw state + reward + obs + draw count + death type)")
    print("GREEN" if (fails + ef) == 0 else f"RED: {fails + ef} failure(s)")
    sys.exit(1 if (fails + ef) else 0)


main()
