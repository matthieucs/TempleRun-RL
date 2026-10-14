"""Profile the Python reference sim to split env.step() cost into:
    - the HOT INNER STEP  (_apply_action + _advance + _cross_into_next_cell + _resolve_obstacle_on_enter)
    - track GENERATION    (_next_cell_procedural + _extend_track + helpers)
    - the REST            (_snapshot, state_to_obs, shaping reward, bookkeeping, reset)

Uses the sim's own `_reactive_policy` so episodes are LONG (representative of training),
not the 3-step episodes a uniform-random action stream produces.

Run:  G:/python312/python.exe cpp_bench/profile_python.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from templerun_rl import sim_env as SE  # noqa: E402


# device-true-ish config: keep the branchy death paths ALIVE so the profile is representative
CFG = {
    "wide_gap_landing_model": True,
    "crevasse_apex_all": True,
    "crevasse_graze_worldy": True,
    "hole_landing_model": True,
    "hole_trap_kill_prob": 0.35,
    "stepup_clip_band": 0.25,
}


def make_env(seed=0, hist=4):
    env = SE.TempleRunSimEnv(config=dict(CFG), seed=seed, max_episode_steps=100000,
                             action_history_len=hist)
    env.reset(seed=seed)
    return env


def rollout_full(env, n_steps):
    """Full env.step(): action model + physics + generation + snapshot + obs + reward."""
    a = 6
    st = env._prev_state
    for i in range(n_steps):
        a = SE._reactive_policy(st, a)
        _o, _r, term, trunc, info = env.step(a)
        st = info["raw_state"]
        if term or trunc:
            env.reset(seed=(i % 10007))
            st = env._prev_state
            a = 6


def rollout_core(env, n_steps):
    """HOT INNER STEP ONLY: _apply_action + _advance (+ the generation they trigger).
    No _snapshot, no obs build, no shaping reward, no gym bookkeeping.
    Uses a cheap fixed action stream so no policy cost is included."""
    acts = [6, 6, 2, 6, 6, 3, 0, 6, 1, 6, 2, 6, 6, 6, 4, 6, 6, 2, 6, 5]
    na = len(acts)
    dt = env.step_dt
    for i in range(n_steps):
        if env._is_dead:
            env.reset(seed=(i % 10007))
        env._apply_action(acts[i % na])
        env._advance(dt)


def bench(fn, env, n_steps):
    fn(env, 2000)                       # warm
    t0 = time.perf_counter()
    fn(env, n_steps)
    return n_steps / (time.perf_counter() - t0)


def profile_split(n_steps=20000):
    import cProfile
    import pstats
    env = make_env(0)
    pr = cProfile.Profile()
    pr.enable()
    rollout_full(env, n_steps)
    pr.disable()
    st = pstats.Stats(pr)
    buckets = {
        "hot_inner": {"_apply_action", "_advance", "_cross_into_next_cell",
                      "_resolve_obstacle_on_enter", "_stumble", "_kill", "_current_cell",
                      "_cell_progress", "_long_pos", "_attempt_turn", "_execute_turn",
                      "_turn_window_open", "_nearest_cell"},
        "generation": {"_next_cell_procedural", "_extend_track", "_next_cell", "_element",
                       "_peek_next", "_difficulty", "_ramp", "_sample_obstacle_type",
                       "_queue_compound", "_queue_jumpslide", "_queue_jumpnarrow",
                       "_queue_jumpgap", "_queue_slidegap", "_narrow_ptype"},
        "snapshot": {"_snapshot", "_blocked"},
        "obs_build": {"state_to_obs", "stacked_obs", "_norm_enum", "_build_obs", "<lambda>",
                      "clip", "_clip", "_wrapit", "_wrapfunc", "inner"},
        "reward": {"shaping_reward"},
        "reset_seed": {"reset", "_reset_internal_state", "_build_initial_track", "np_random",
                       "_sample_domain", "seed"},
        "policy": {"_reactive_policy"},
    }
    tot = 0.0
    acc = {k: 0.0 for k in buckets}
    acc["other"] = 0.0
    for (fnm, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
        tot += tt
        for bk, names in buckets.items():
            if name in names:
                acc[bk] += tt
                break
        else:
            acc["other"] += tt
    return tot, acc


if __name__ == "__main__":
    N = 30000
    e1 = make_env(0)
    full = bench(rollout_full, e1, N)
    e2 = make_env(0)
    core = bench(rollout_core, e2, N)
    print("Python reference, FULL env.step()        : %8.0f env-steps/s  (%7.1f us/step)"
          % (full, 1e6 / full))
    print("Python reference, HOT INNER ONLY         : %8.0f env-steps/s  (%7.1f us/step)"
          % (core, 1e6 / core))
    print("   (hot inner = _apply_action + _advance + cell crossing + death resolution")
    print("    + the procedural generation those trigger; NO snapshot/obs/reward/gym)")
    tot, acc = profile_split()
    print("\ncProfile self-time split of FULL step (profiler inflates absolutes; read the RATIOS):")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print("   %-22s %7.3f s   %5.1f%%" % (k, v, 100.0 * v / tot))
    print("   %-22s %7.3f s" % ("TOTAL", tot))
