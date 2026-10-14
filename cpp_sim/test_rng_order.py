"""RNG DRAW-ORDER ASSERTION for the C++ port.

WHY THIS EXISTS
    ``scripts/sim_equivalence.py``'s deterministic tier INJECTS fixed tracks, so
    procedural generation and every rng consumer are invisible to it.  The
    statistical tier only compares distributions.  Neither can catch a draw-order
    slip -- and a draw-order slip is exactly the failure mode the codebase's
    "check prob > 0 FIRST, only then draw" rule exists to prevent: it silently
    shifts the track stream and invalidates calibrations without failing a test.

WHAT IT DOES
    Drives the Python reference and the C++ port off ONE SHARED sequence of
    uniform doubles.  A proxy Generator feeds the reference; ``sim_rng_replay_set``
    feeds the C++.  With identical randomness the two must stay in lockstep
    step-for-step INCLUDING the generated track, and the number of values each
    has consumed must be equal after EVERY step.  Any extra, missing or reordered
    draw shows up immediately.

    The config turns ON every optional rng consumer at once (all five compound
    scenarios, the device ramp, tight pairs, double obstacles, one-cell
    crevasses, jump latency + jitter, jump drop, strafe refusal, post-land
    re-jump, held-jump re-jump, the probabilistic hole trap, the turn tail risk,
    crevasse takeoff variance, domain randomisation and the deep-start
    curriculum), so every draw site is exercised.

    python cpp_sim/test_rng_order.py
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parent))

from templerun_rl.sim_env import TempleRunSimEnv  # noqa: E402
from cpp_sim.cpp_vec_env import CppSimGymAdapter, lib  # noqa: E402
import train_sim  # noqa: E402


# --------------------------------------------------------------------- proxy
class SeqRNG:
    """numpy-Generator-shaped proxy that consumes ONE shared uniform sequence,
    with exactly the per-call consumption the C++ side uses:

        random()            1 value
        uniform(lo, hi)     1 value   -> lo + (hi-lo)*u
        integers(lo, hi)    1 value   -> lo + int(u*(hi-lo))     [hi exclusive]
        choice(a)           1 value   -> a[int(u*len(a))]
        choice(a, p=w)      1 value   -> cdf searchsorted-right (numpy's own rule)
        normal(mu, sigma)   2 values  -> Box-Muller with an explicit spare

    The last two differ from real numpy internals (ziggurat / Lemire), which is
    the point: BOTH sides use this definition, so the test measures draw ORDER,
    not numpy's bit layout.
    """

    def __init__(self, vals):
        self.v = vals
        self.i = 0
        self._spare = None

    def _u(self):
        if self.i >= len(self.v):
            raise RuntimeError("SeqRNG exhausted -- give the test more values")
        u = self.v[self.i]
        self.i += 1
        return float(u)

    def random(self, *a, **k):
        return self._u()

    def uniform(self, lo=0.0, hi=1.0, size=None):
        return float(lo) + (float(hi) - float(lo)) * self._u()

    def integers(self, lo, hi=None, size=None, endpoint=False):
        if hi is None:
            lo, hi = 0, lo
        n = int(hi) - int(lo) + (1 if endpoint else 0)
        if n <= 1:
            self._u()
            return int(lo)
        v = int(lo) + int(self._u() * n)
        return min(v, int(lo) + n - 1)

    def choice(self, a, p=None, size=None, replace=True):
        a = list(a)
        u = self._u()
        if p is None:
            return a[min(int(u * len(a)), len(a) - 1)]
        w = np.asarray(p, dtype=float)
        c = np.cumsum(w)
        c = c / c[-1]
        idx = int(np.searchsorted(c, u, side="right"))
        return a[min(idx, len(a) - 1)]

    def normal(self, mu=0.0, sigma=1.0, size=None):
        if self._spare is not None:
            z, self._spare = self._spare, None
            return float(mu) + float(sigma) * z
        while True:
            u1 = self._u()
            if u1 > 1e-12:
                break
        u2 = self._u()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        self._spare = r * math.sin(th)
        return float(mu) + float(sigma) * (r * math.cos(th))


# -------------------------------------------------------------------- config
# every optional rng consumer ON at once
CFG = {
    "domain_rand": True,
    "curriculum_max_start_dist": 3000.0,
    "curriculum_deep_start_frac": 0.5,
    "curriculum_dist_power": 1.5,
    # generation
    "device_difficulty_ramp": True,
    "double_obstacle_percent": 0.15,
    "tight_pair_prob": 0.35, "tight_pair_min_cells": 2,
    "crevasse_single_cell_prob": 0.675,
    "turnobstacle_prob": 0.12,
    "compound_prob": 0.20, "compound_after_distance": 0.0,
    "jumpslide_prob": 0.30, "jumpslide_after_distance": 0.0,
    "jumpnarrow_prob": 0.30, "jumpnarrow_after_distance": 0.0,
    "jumpgap_prob": 0.30, "jumpgap_after_distance": 0.0, "jumpgap_chain_max": 3,
    "slidegap_prob": 0.30, "slidegap_after_distance": 0.0,
    "narrows_after_distance": 0.0,
    "zone_after_distance": 0.0, "zone_change_every_cells": 10,
    # action / death models
    "hook_turn_buffer": True, "turn_approach_buffer": True, "turn_linger_frames": 3,
    "turn_send_lat_frames": 0.7, "turn_offjunction_drop": True,
    "turn_drop_debounce_s": 0.3,
    "turn_tail_pmax": 0.15,
    "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1,
    "jump_drop_prob": 0.08, "jump_refuse_strafing_prob": 0.6,
    "post_land_gap_rejump_prob": 0.46,
    "held_jump_rejump_solid": 0.10, "held_jump_rejump_gap": 0.28,
    "wide_gap_landing_model": True, "hole_landing_model": True,
    "hole_trap_kill_prob": 0.35, "hole_trap_apex_frac": 0.5,
    "crevasse_apex_all": True, "crevasse_graze_worldy": True,
    "stepup_clip_band": 0.25,
    "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2,
}
SHAPING = dict(action_cost=0.02, reaction_bonus=0.1, turn_spam_penalty=0.1,
               idle_action_penalty=0.05, redundant_action_penalty=0.12,
               strafe_penalty=0.3, narrow_hold_bonus=0.18, aim_center_weight=1.0,
               velocity_drop_penalty=0.02, off_center_penalty=0.03,
               slide_jump_penalty=0.15, airborne_action_penalty=0.1,
               jump_hold_penalty=0.12, turn_buffer_bonus=0.5)

FLOAT_KEYS = {"velocity", "distance", "cell_progress", "lane", "stumble_kill_timer", "score"}
ATOL = {"velocity": 1e-6, "distance": 1e-6, "cell_progress": 1e-9, "lane": 1e-9,
        "stumble_kill_timer": 1e-9, "score": 0.5}


def diff(a, b, k):
    for f in sorted(set(a) & set(b)):
        x, y = a[f], b[f]
        if f in ("narrow_target_lane", "next2_narrow_lane"):
            x = 0 if x is None else int(x)
            y = 0 if y is None else int(y)
            if x != y:
                return f"step {k}: {f} py={x} cpp={y}"
            continue
        if f in FLOAT_KEYS:
            if abs(float(x) - float(y)) > ATOL.get(f, 1e-6):
                return f"step {k}: {f} py={x!r} cpp={y!r}"
        elif bool(x) != bool(y) if isinstance(x, bool) else int(x) != int(y):
            return f"step {k}: {f} py={x!r} cpp={y!r}"
    return None


def oracle(mem, st):
    saved = dict(train_sim._oracle_mem)
    train_sim._oracle_mem.clear()
    train_sim._oracle_mem.update(mem)
    a = train_sim._oracle(st)
    mem.clear()
    mem.update(train_sim._oracle_mem)
    train_sim._oracle_mem.clear()
    train_sim._oracle_mem.update(saved)
    return a


def run_episode(vals, steps, K, FS):
    L = lib()
    L.sim_rng_replay_set.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    L.sim_rng_replay_pos.restype = ctypes.c_int
    L.sim_rng_replay_overrun.restype = ctypes.c_int
    # detach any previous episode's (now-freed) buffer before constructing envs --
    # sim_init resets each env, which draws.
    L.sim_rng_replay_set(None, 0)
    py = TempleRunSimEnv(config=CFG, seed=0, max_episode_steps=100000,
                         reward_mode="distance", action_history_len=K,
                         frame_stack=FS, **SHAPING)
    py._rng = SeqRNG(vals)
    cpp = CppSimGymAdapter(config=CFG, seed=0, max_episode_steps=100000,
                           reward_mode="distance", action_history_len=K,
                           frame_stack=FS, **SHAPING)
    arr = (ctypes.c_double * len(vals))(*vals)
    L.sim_rng_replay_set(arr, len(vals))

    o_py, i_py = py.reset()
    o_cp, i_cp = cpp.reset()
    st_py, st_cp = i_py["raw_state"], i_cp["raw_state"]

    if py._rng.i != L.sim_rng_replay_pos():
        return f"reset: draw count py={py._rng.i} cpp={L.sim_rng_replay_pos()}", 0
    e = diff(st_py, st_cp, -1)
    if e:
        return e, 0
    if not np.allclose(o_py, o_cp, atol=2e-6):
        j = int(np.argmax(np.abs(np.asarray(o_py) - np.asarray(o_cp))))
        return f"reset obs[{j}] py={o_py[j]} cpp={o_cp[j]}", 0

    mem_py = {"cooldown": 0, "prev_dist": 1e18}
    mem_cp = {"cooldown": 0, "prev_dist": 1e18}
    for k in range(steps):
        a_py = oracle(mem_py, st_py)
        a_cp = oracle(mem_cp, st_cp)
        if a_py != a_cp:
            return f"step {k}: policies diverged {a_py} != {a_cp}", k
        o_py, r_py, t_py, tr_py, inf_py = py.step(a_py)
        o_cp, r_cp, t_cp, tr_cp, inf_cp = cpp.step(a_cp)
        st_py, st_cp = inf_py["raw_state"], inf_cp["raw_state"]
        if py._rng.i != L.sim_rng_replay_pos():
            return (f"step {k}: DRAW COUNT py={py._rng.i} cpp={L.sim_rng_replay_pos()}"
                    f"  (delta {L.sim_rng_replay_pos() - py._rng.i})"), k
        e = diff(st_py, st_cp, k)
        if e:
            return e, k
        if abs(r_py - r_cp) > 1e-6:
            return f"step {k}: reward py={r_py} cpp={r_cp}", k
        if bool(t_py) != bool(t_cp) or bool(tr_py) != bool(tr_cp):
            return f"step {k}: done py=({t_py},{tr_py}) cpp=({t_cp},{tr_cp})", k
        if not np.allclose(o_py, o_cp, atol=2e-6):
            j = int(np.argmax(np.abs(np.asarray(o_py) - np.asarray(o_cp))))
            return f"step {k}: obs[{j}] py={o_py[j]} cpp={o_cp[j]}", k
        if t_py:
            if int(inf_py.get("death_type") or 0) != int(inf_cp.get("death_type") or 0):
                return (f"step {k}: death_type py={inf_py.get('death_type')} "
                        f"cpp={inf_cp.get('death_type')}"), k
            return None, k + 1
    return None, steps


def main():
    fails = 0
    total_steps = 0
    n_ep = int(os.environ.get("RNG_ORDER_EPS", "60"))
    for ep in range(n_ep):
        rng = np.random.default_rng(1000 + ep)
        vals = rng.random(200000)
        K, FS = (4, 1) if ep % 3 == 0 else ((0, 2) if ep % 3 == 1 else (0, 1))
        err, n = run_episode(vals, 400, K, FS)
        total_steps += n
        if err:
            fails += 1
            print(f"FAIL ep{ep} (K={K},FS={FS}): {err}", flush=True)
    print(f"\nRNG DRAW-ORDER: {n_ep - fails}/{n_ep} episodes lockstep "
          f"({total_steps} shared-stream steps)")
    print("GREEN" if fails == 0 else f"RED: {fails} failure(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
