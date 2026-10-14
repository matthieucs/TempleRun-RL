"""experiments/experiments/cpp_bench/bench.py -- drive experiments/cpp_bench/sim_core.dll via ctypes and measure env-steps/s.

Runs the SAME subset in the Python reference (_apply_action + _advance, i.e. no
_snapshot / no obs build / no shaping reward) so the comparison is apples-to-apples,
plus the full env.step() for context, plus lightweight correctness checks.

Two workloads are measured because they answer different questions:
  FIXED   -- a canned repeating action stream.  Pure API shape (actions supplied
             from outside, as PPO would), but a turn on a straight is fatal so
             episodes last ~5 steps and the loop is RESET-dominated => NOT
             representative of training.
  ORACLE  -- a tiny rule-based driver (mirrored bit-for-bit in C++ and Python)
             that keeps episodes hundreds of steps long, so cell crossings and
             track generation happen at the real rate.  This is the number to
             quote.  The driver's own cost is measured and reported separately.

Run:  G:/python312/python.exe experiments/experiments/cpp_bench/bench.py
"""
import ctypes
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

DLL = os.path.join(HERE, "sim_core.dll")
ACTS = [6, 6, 2, 6, 6, 3, 0, 6, 1, 6, 2, 6, 6, 6, 4, 6, 6, 2, 6, 5]

CFG = {"wide_gap_landing_model": True, "crevasse_apex_all": True,
       "crevasse_graze_worldy": True, "hole_landing_model": True,
       "hole_trap_kill_prob": 0.35, "stepup_clip_band": 0.25}

_MJ = frozenset((1, 3, 4, 7))
_MS = frozenset((5, 6))


def load():
    lib = ctypes.CDLL(DLL)
    lib.sim_state_size.restype = ctypes.c_int
    lib.sim_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
    lib.sim_step_many.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int)]
    lib.sim_step_many.restype = ctypes.c_int
    lib.sim_step_many_oracle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.sim_step_many_oracle.restype = ctypes.c_int
    lib.sim_oracle_only.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.sim_oracle_only.restype = ctypes.c_int
    lib.sim_step_many_oracle_obs.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float)]
    lib.sim_step_many_oracle_obs.restype = ctypes.c_int
    lib.sim_step_one.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.sim_reset_one.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
    for n in ("sim_get_distance", "sim_get_last_ep_distance", "sim_get_velocity",
              "sim_get_lane", "sim_get_score"):
        getattr(lib, n).argtypes = [ctypes.c_void_p, ctypes.c_int]
        getattr(lib, n).restype = ctypes.c_double
    for n in ("sim_get_cell_index", "sim_get_episodes", "sim_get_death_type",
              "sim_get_is_dead", "sim_get_zone_level"):
        getattr(lib, n).argtypes = [ctypes.c_void_p, ctypes.c_int]
        getattr(lib, n).restype = ctypes.c_int
    return lib


def make_states(lib, n_envs, seed=12345):
    buf = ctypes.create_string_buffer(lib.sim_state_size() * n_envs)
    lib.sim_init(buf, n_envs, seed)
    return buf


def make_actions(n_steps, n_envs, phase=0):
    arr = (ctypes.c_int * (n_steps * n_envs))()
    na = len(ACTS)
    for i in range(n_steps * n_envs):
        arr[i] = ACTS[(i + phase) % na]
    return arr


def timed(fn, target=2.0, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < target:
        n += fn()
    return n / (time.perf_counter() - t0)


# --------------------------------------------------------------- Python driver
def py_oracle(env):
    """Bit-for-bit mirror of oracle_action() in sim_core.cpp."""
    trk = env._track
    i = env._cell_index
    c = trk[i]
    n = trk[i + 1] if i + 1 < len(trk) else c
    cp = env._cell_progress()
    if c.get("is_turn") and (c.get("can_turn_left") or c.get("can_turn_right")) \
            and 0.20 <= cp <= 0.50:
        return 4 if c.get("can_turn_left") else 5
    no = n.get("obstacles", 0)
    if no in _MJ and cp > 0.55:
        return 2
    if no in _MS and cp > 0.55:
        return 3
    nl = n.get("narrow_lane")
    if nl is None:
        nl = c.get("narrow_lane")
    if nl is not None:
        tgt = nl * 4.5
        if env._lane < tgt - 0.4:
            return 1
        if env._lane > tgt + 0.4:
            return 0
        return 6
    return 6


def make_py_env(hist=0, seed=0):
    from templerun_rl import sim_env as SE
    env = SE.TempleRunSimEnv(config=dict(CFG), seed=seed, max_episode_steps=10 ** 9,
                             action_history_len=hist)
    env.reset(seed=seed)
    return env


def py_core_oracle(env, n):
    dt = env.step_dt
    for i in range(n):
        if env._is_dead:
            env.reset(seed=i % 10007)
        env._apply_action(py_oracle(env))
        env._advance(dt)
    return n


def py_core_fixed(env, n, state=[0]):
    dt = env.step_dt
    na = len(ACTS)
    k = state[0]
    for _ in range(n):
        if env._is_dead:
            env.reset(seed=k % 10007)
        env._apply_action(ACTS[k % na])
        env._advance(dt)
        k += 1
    state[0] = k
    return n


def py_oracle_only(env, n):
    for _ in range(n):
        py_oracle(env)
    return n


def py_full_oracle(env, n):
    for i in range(n):
        _o, _r, term, _tr, _in = env.step(py_oracle(env))
        if term:
            env.reset(seed=i % 10007)
    return n


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    lib = load()
    print("State struct = %d bytes/env  (%.0f KB for 1024 envs)"
          % (lib.sim_state_size(), lib.sim_state_size() * 1024 / 1024.0))

    # ---------------- correctness -------------------------------------------
    print("\n=== CORRECTNESS (lightweight; full equivalence vs sim_env.py is out of scope) ===")

    def trace(seed):
        buf = make_states(lib, 1, seed=seed)
        out = []
        for i in range(500):
            if lib.sim_get_is_dead(buf, 0):
                lib.sim_reset_one(buf, 0, 0)
            lib.sim_step_one(buf, 0, ACTS[i % len(ACTS)])
            out.append((round(lib.sim_get_distance(buf, 0), 9),
                        round(lib.sim_get_lane(buf, 0), 9),
                        lib.sim_get_cell_index(buf, 0), lib.sim_get_death_type(buf, 0)))
        return out
    a, b, c = trace(777), trace(777), trace(778)
    print("  same seed  -> identical 500-step trace : %s" % (a == b))
    print("  other seed -> different trace          : %s" % (a != c))

    # episode-length / death-mix plausibility, oracle-driven, both sims
    buf = make_states(lib, 512, seed=4242)
    lib.sim_step_many_oracle(buf, 512, 6000)
    cd = sorted(lib.sim_get_last_ep_distance(buf, i) for i in range(512)
                if lib.sim_get_episodes(buf, i) > 0)
    ceps = sum(lib.sim_get_episodes(buf, i) for i in range(512))
    cmix = {}
    for i in range(512):
        d = lib.sim_get_death_type(buf, i)
        if d:
            cmix[d] = cmix.get(d, 0) + 1
    penv = make_py_env()
    pd, pmix, k = [], {}, 0
    dt = penv.step_dt
    for i in range(60000):
        if penv._is_dead:
            pd.append(penv._distance)
            pmix[penv._death_type] = pmix.get(penv._death_type, 0) + 1
            penv.reset(seed=i % 10007)
        penv._apply_action(py_oracle(penv))
        penv._advance(dt)
    pd.sort()
    print("  C++    oracle: %5d episodes / 3.07M steps, median last-ep dist %8.1f  death mix %s"
          % (ceps, cd[len(cd) // 2] if cd else 0, cmix))
    print("  Python oracle: %5d episodes /  60k steps, median dist %8.1f  death mix %s"
          % (len(pd), pd[len(pd) // 2] if pd else 0, pmix))
    print("  (independent RNG streams -> expect SIMILAR distributions, not identical)")
    print("  mean episode length: C++ %.0f steps | Python %.0f steps"
          % (512 * 6000.0 / max(1, ceps), 60000.0 / max(1, len(pd))))

    # ---------------- throughput --------------------------------------------
    print("\n=== THROUGHPUT (single-threaded, 1 CPU core) ===")
    e1 = make_py_env()
    py_o = timed(lambda: py_core_oracle(e1, 4000))
    e2 = make_py_env()
    py_f = timed(lambda: py_core_fixed(e2, 4000))
    e3 = make_py_env()
    py_drv = timed(lambda: py_oracle_only(e3, 20000))
    e4 = make_py_env(hist=4)
    py_full = timed(lambda: py_full_oracle(e4, 2000))
    print("  PYTHON reference")
    print("    hot-inner subset, ORACLE-driven : %10.0f env-steps/s (%8.2f us/step)"
          % (py_o, 1e6 / py_o))
    print("    hot-inner subset, FIXED stream  : %10.0f env-steps/s (%8.2f us/step)"
          % (py_f, 1e6 / py_f))
    print("    FULL env.step(), ORACLE-driven  : %10.0f env-steps/s (%8.2f us/step)"
          % (py_full, 1e6 / py_full))
    print("    (driver alone, no physics       : %10.0f calls/s => %.2f us of the above)"
          % (py_drv, 1e6 / py_drv))

    print("  C++ CORE (same subset as 'hot-inner'); each cell = median of 3 runs")
    results = {}

    def med3(fn):
        v = sorted(timed(fn, target=0.7) for _ in range(3))
        return v[1]

    for n_envs in (1, 64, 256, 1024, 4096):
        bo = make_states(lib, n_envs)
        so = med3(lambda: lib.sim_step_many_oracle(bo, n_envs, 256) * 0 + 256 * n_envs)
        bf = make_states(lib, n_envs)
        af = make_actions(256, n_envs)
        sf = med3(lambda: lib.sim_step_many(bf, n_envs, 256, af) * 0 + 256 * n_envs)
        bd = make_states(lib, n_envs)
        lib.sim_step_many_oracle(bd, n_envs, 200)
        sd = med3(lambda: lib.sim_oracle_only(bd, n_envs, 256) * 0 + 256 * n_envs)
        a1 = make_actions(1, n_envs)
        bp = make_states(lib, n_envs)
        sp = med3(lambda: sum(lib.sim_step_many(bp, n_envs, 1, a1) * 0 + n_envs
                              for _ in range(64)))
        bx = make_states(lib, n_envs)
        obs = (ctypes.c_float * (n_envs * 65))()
        rew = (ctypes.c_float * n_envs)()
        sx = med3(lambda: lib.sim_step_many_oracle_obs(bx, n_envs, 256, obs, rew) * 0
                  + 256 * n_envs)
        results[n_envs] = (so, sf, sd, sp, sx)
        print("    %5d envs | ORACLE %9.0f (%.3f us) | FIXED %9.0f | +OBS+REW %9.0f (%.3f us) "
              "| driver-only %10.0f | 1 call/vec-step %9.0f"
              % (n_envs, so, 1e6 / so, sf, sx, 1e6 / sx, sd, sp))

    # CONSERVATIVE headline: median across env counts, not the peak.  A training run is
    # competing for the CPU on this box, so the per-config spread is dominated by external
    # noise; the median is the honest summary and the min is the safe lower bound.
    def med(vals):
        v = sorted(vals)
        return v[len(v) // 2]
    best_o = med([r[0] for r in results.values()])
    best_x = med([r[4] for r in results.values()])
    print("\n  spread across env counts (external CPU contention): hot-inner %.2f-%.2f M/s, "
          "+obs %.2f-%.2f M/s"
          % (min(r[0] for r in results.values()) / 1e6, max(r[0] for r in results.values()) / 1e6,
             min(r[4] for r in results.values()) / 1e6, max(r[4] for r in results.values()) / 1e6))
    print("\n=== HEADLINE (oracle-driven, the representative workload) ===")
    print("  C++ hot-inner subset       : %10.0f env-steps/s" % best_o)
    print("  C++ + obs build + reward   : %10.0f env-steps/s  (the real per-step budget)" % best_x)
    print("  Python same subset         : %10.0f env-steps/s   -> C++ is %6.0fx" % (py_o, best_o / py_o))
    print("  Python full env.step()     : %10.0f env-steps/s   -> C++(+obs) is %6.0fx"
          % (py_full, best_x / py_full))
    print("  vs the quoted 9,962/s      : %6.0fx (hot-inner) / %6.0fx (with obs)"
          % (best_o / 9962.0, best_x / 9962.0))

    # ---------------- end-to-end training projection -------------------------
    print("\n=== END-TO-END TRAINING PROJECTION ===")
    UPD_MS = 1.27 * 32          # 32 minibatches x 1.27 ms  = PPO update per iteration
    SAMPLES = 4096              # samples (= env-steps) per iteration
    for label, sim_sps in (("torch GPU sim (measured today)", 14600.0),
                           ("Python CPU sim (measured today)", py_full),
                           ("C++ core + obs (this benchmark)", best_x)):
        roll_ms = SAMPLES / sim_sps * 1000.0
        tot = roll_ms + UPD_MS
        print("   %-32s rollout %8.2f ms + update %5.2f ms = %8.2f ms/iter"
              "  => %9.0f steps/s end-to-end" % (label, roll_ms, UPD_MS, tot, SAMPLES / tot * 1000.0))
    base = SAMPLES / (SAMPLES / 14600.0 * 1000.0 + UPD_MS) * 1000.0
    new = SAMPLES / (SAMPLES / best_x * 1000.0 + UPD_MS) * 1000.0
    print("   => projected end-to-end speedup vs the torch GPU sim: %.1fx"
          "   (update-bound: %.0f%% of the new iteration is the PPO update)"
          % (new / base, 100.0 * UPD_MS / (SAMPLES / best_x * 1000.0 + UPD_MS)))

    print("\n   SENSITIVITY -- the projection barely depends on how fast the C++ actually is,")
    print("   because past ~1M steps/s the PPO update is the whole iteration:")
    for sps in (200e3, 500e3, 907e3, 2e6, 5e6, 8e6):
        roll = SAMPLES / sps * 1000.0
        e2e = SAMPLES / (roll + UPD_MS) * 1000.0
        print("     C++ at %7.2f M steps/s -> %8.0f steps/s end-to-end (%4.1fx), "
              "update = %3.0f%% of iter" % (sps / 1e6, e2e, e2e / base, 100 * UPD_MS / (roll + UPD_MS)))
