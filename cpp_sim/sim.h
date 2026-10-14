/* ============================================================================
 * cpp_sim/sim.h -- C ABI of the full C++ port of src/templerun_rl/sim_env.py.
 *
 * Everything is plain C so ctypes can drive it (see cpp_sim/cpp_vec_env.py).
 * The Python CPU sim (sim_env.py) is the SPECIFICATION; this is a transcription
 * of it, not a re-derivation.  scripts/validate/sim_equivalence.py --backend cpp is the
 * acceptance gate.
 *
 * CONFIG MODEL
 *   There is ONE global config (a flat array of doubles addressed by name) plus
 *   a small per-env override block for the knobs that domain-randomisation and
 *   episode_profiles touch.  Set the global config, then sim_init(), then step.
 *
 * DETERMINISM
 *   Each env owns an explicit PCG32 stream.  Every rng consumer mirrors the
 *   Python "check prob > 0 FIRST, only then draw" discipline, so a disabled
 *   feature consumes no random numbers and the track stream does not shift.
 *   sim_get_draws() exposes the per-env draw counter for the draw-order test.
 * ==========================================================================*/
#ifndef TEMPLERUN_CPP_SIM_H
#define TEMPLERUN_CPP_SIM_H

#ifdef _WIN32
#define TRS_EXPORT extern "C" __declspec(dllexport)
#else
#define TRS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

/* ---- sizes ------------------------------------------------------------- */
TRS_EXPORT int sim_state_size(void);      /* bytes of ONE env's State        */
TRS_EXPORT int sim_obs_len(void);         /* FS*OBS_BASE + K*7               */
TRS_EXPORT int sim_obs_base(void);        /* 37                              */
TRS_EXPORT int sim_raw_len(void);         /* 37 doubles, see RAW_* order     */

/* ---- config ------------------------------------------------------------ */
TRS_EXPORT void sim_cfg_reset(void);                       /* DEFAULT_SIM_CONFIG */
TRS_EXPORT int  sim_cfg_set(const char* name, double v);   /* 1 ok / 0 unknown   */
TRS_EXPORT double sim_cfg_get(const char* name);
TRS_EXPORT void sim_cfg_set_vel_curve(const double* d, const double* v, int n);
TRS_EXPORT void sim_cfg_set_ramp_table(const double* rows, int n_rows); /* 6 doubles/row */
TRS_EXPORT void sim_cfg_set_obst_weights(const int* types, const double* w, int n);
TRS_EXPORT void sim_cfg_set_narrow_len_weights(const int* lens, const double* w, int n);
/* type_reject (draw-then-substitute): parallel arrays of prev-type, drawn-type, P(reject).
 * Clears the whole matrix first, so passing n = 0 disables every rejection. */
TRS_EXPORT void sim_cfg_set_type_reject(const int* prev, const int* cur, const double* p, int n);

/* episode_profiles (ported from sim_env_torch.py -- the CPU reference has none) */
TRS_EXPORT void sim_cfg_clear_profiles(void);
TRS_EXPORT int  sim_profile_key_id(const char* name);   /* -1 = unsupported override */
TRS_EXPORT int  sim_cfg_add_profile(double weight, int n, const int* key_ids,
                                    const double* vals);

/* env-level (non-sim) parameters: dt, caps, reward + shaping coefficients */
TRS_EXPORT void sim_set_env_params(double step_dt, int max_episode_steps,
                                   double reward_scale, int reward_mode_score,
                                   double death_penalty, int action_history_len,
                                   int frame_stack, int pj_velocity_invariant);
TRS_EXPORT void sim_set_shaping(const double* coeffs, int n);   /* see SHP_* order */

/* fixed-track injection (the deterministic equivalence tier).
 * cells: 10 ints each -- type, level, obstacles, is_turn, can_l, can_r,
 *        is_coin, wide_gap, stepup, narrow_lane(0 = None).  n<=0 clears. */
TRS_EXPORT void sim_set_track(const int* cells, int n);

/* ---- lifecycle --------------------------------------------------------- */
TRS_EXPORT void sim_init(void* buf, int n_envs, unsigned long long seed0);
TRS_EXPORT void sim_reset_all(void* buf, int n_envs, long long seed, float* obs);
TRS_EXPORT void sim_reset_one(void* buf, int env, long long seed, float* obs);

/* ---- stepping ---------------------------------------------------------- */
/* flags bit0 = terminated (dead), bit1 = elapsed >= max_episode_steps.
 * auto_reset != 0 -> vec-env semantics (dead envs restart, obs is post-reset,
 * term_obs holds the pre-reset observation).  auto_reset == 0 -> gym semantics.
 * Any of the output pointers except obs/rew/flags may be NULL. */
TRS_EXPORT void sim_step(void* buf, int n_envs, const int* actions, int auto_reset,
                         float* obs, double* rew, unsigned char* flags,
                         float* term_obs, double* ep_ret, int* ep_len,
                         double* ep_dist, int* death_type);

/* ---- introspection ----------------------------------------------------- */
TRS_EXPORT void sim_get_raw(void* buf, int env, double* out);      /* sim_raw_len() */
TRS_EXPORT void sim_get_obs(void* buf, int env, float* out);       /* sim_obs_len() */
TRS_EXPORT void sim_approach_turn_hint(void* buf, int n_envs, int* out);
TRS_EXPORT double sim_get_distance(void* buf, int env);
TRS_EXPORT int  sim_get_death_type(void* buf, int env);
TRS_EXPORT int  sim_get_is_dead(void* buf, int env);
TRS_EXPORT unsigned long long sim_get_draws(void* buf, int env);
TRS_EXPORT int  sim_get_profile_idx(void* buf, int env);
TRS_EXPORT void sim_stumble(void* buf, int env);       /* fidelity_scorecard hook */
TRS_EXPORT double sim_get_velocity(void* buf, int env);

/* ---- RNG replay (testing only) ----------------------------------------- */
/* Feed every draw from `vals` instead of the PCG stream so the C++ and the
 * Python reference can share one random sequence -- the draw-ORDER assertion.
 * n <= 0 restores the normal PCG stream. */
TRS_EXPORT void sim_rng_replay_set(const double* vals, int n);
TRS_EXPORT int  sim_rng_replay_pos(void);
TRS_EXPORT int  sim_rng_replay_overrun(void);

#endif /* TEMPLERUN_CPP_SIM_H */
