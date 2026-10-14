/* ============================================================================
 * cpp_sim/sim.cpp -- FULL C++ port of src/templerun_rl/sim_env.py (+ obs.py,
 * reward.py, and episode_profiles from sim_env_torch.py).
 *
 * THE PYTHON CPU SIM IS THE SPECIFICATION.  Every branch below is a
 * transcription of the corresponding branch there, in the same order, with the
 * same rng draw order and the same "check prob > 0 FIRST, only then draw"
 * gating, so a disabled feature consumes no random numbers.
 *
 * Gate: scripts/validate/sim_equivalence.py --backend cpp  (deterministic tier must be
 * step-exact against sim_env.py).
 *
 * Build (GCC 4.6.3 / MinGW, no C++11 stdlib):
 *   g++ -O2 -std=c++0x -shared -o sim.dll sim.cpp
 * ==========================================================================*/
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include "sim.h"

/* ------------------------------------------------------------------ enums */
#define OBST_NONE          0
#define OBST_HOLE          1
#define OBST_NARROW        2
#define OBST_TREE_STUMBLE  3
#define OBST_TREE_JUMP     4
#define OBST_TREE_SLIDE    5
#define OBST_FLAME_TOWER   6
#define OBST_LEVEL_CHANGE  7

#define A_STRAFE_L 0
#define A_STRAFE_R 1
#define A_JUMP     2
#define A_SLIDE    3
#define A_TURN_L   4
#define A_TURN_R   5
#define A_NOOP     6
#define N_ACTIONS  7

/* cell flag bits */
#define F_TURN   0x01
#define F_CANL   0x02
#define F_CANR   0x04
#define F_COIN   0x08
#define F_WIDE   0x10
#define F_STEPUP 0x20
#define F_NARROW 0x40
/* step-DOWN to low ground -> crossable ON FOOT (device-measured; see the Python ref's
   stepdown_grounded_ok). Mirrors F_STEPUP through generation, injection and the death rule. */
#define F_SDLOW  0x80

#define TRACK_RING 64
#define TRACK_MASK (TRACK_RING - 1)
#define QUEUE_CAP  64            /* _pending_sequence / _pending_post_gap      */
#define MAX_FS     8
#define MAX_K      8
/* MUST track obs.HORIZON_OBS. Reverted to 0 on 2026-07-28: three arms (10M, 10M, 40M) left the
   horizon cue unlearned (d4-d6 response ~0 against a +0.0030 bar), so the pipeline is back to
   obs 65. Flip BOTH this and obs.HORIZON_OBS together -- a mismatch shows up immediately as an obs
   shape error, which is exactly what it is for. */
#define HORIZON_OBS 0
/* obs 66 since 2026-07-28: +slide_landing_hazard (obs.py OBS_EXTRA_FIELDS). OBS_BASE = len(OBS_FIELDS). */
#if HORIZON_OBS
#define OBS_BASE   41
#else
#define OBS_BASE   39
#endif
#define HORIZON_CELLS 6          /* obs.HORIZON_CELLS                             */
#define MAX_OBS    (MAX_FS * OBS_BASE + MAX_K * N_ACTIONS)
#define RAW_LEN    42           /* fields VISIBLE to Python (cpp_vec_env.RAW_FIELDS) */
/* ⚠️ The raw enum continues PAST RAW_LEN with fields the host never reads: R_soft_stepdown(37),
   R_next_soft_stepdown(38) and, under HORIZON_OBS, R_dist_next_must_*(39,40). The buffers were
   sized RAW_LEN, so `snapshot()` wrote 2 doubles OFF THE END of a 37-element stack array on EVERY
   step, and `shaping_reward` READ prev_raw[37]/[38] for the step-down gate -- past the end of that
   member and into the next State fields, i.e. garbage. That is why stepdown_bonus_gate measured
   INERT. Size the storage by the full enum; keep RAW_LEN as the exported prefix. */
#define RAW_CAP    42
#define LATBUF_CAP 16

static inline int must_jump_o(int o) {
    return o == OBST_HOLE || o == OBST_TREE_JUMP ||
           o == OBST_TREE_STUMBLE || o == OBST_LEVEL_CHANGE;
}
static inline int must_slide_o(int o) {
    return o == OBST_TREE_SLIDE || o == OBST_FLAME_TOWER;
}

/* --------------------------------------------------------- obs.py constants */
#define OBS_VEL_MIN        100.0
#define OBS_VEL_MAX        220.0
#define OBS_PATH_LEVEL_MAX 4.0
#define OBS_FACING_MAX     4.0
#define OBS_LANE_MIN       (-8.0)
#define OBS_LANE_MAX       8.0
#define OBS_NARROW_CENTER  4.5
#define DCPROG_PER_VEL     (0.1 / 60.0)
#define LAND_SPAN_A        0.0143
#define LAND_SPAN_B        (-0.11)
/* the arc is LONGER over a drop and SHORTER over a rise (device-measured; mirror of obs.py
   LAND_SPAN_LEVEL_SCALED). landing_hazard was computing the landing cell from a FLAT span, so it
   was wrong by a whole cell exactly at crevasses -- where chains kill. */
#define LAND_SPAN_LEVEL_SCALED 1
#define LAND_SPAN_STEPDOWN 1.17
#define LAND_SPAN_STEPUP   0.74
#define PJ_JUMP_DURATION   0.8
#define PJ_CELL_LENGTH     60.0
#define LEDGE_CELLS_AHEAD  2.0
#define PJ_NONE            1.0
#define TURN_DEADLINE_CP   0.56
#define TURN_SEND_FRAMES   1.0

/* ============================================================== CONFIG ==== */
/* One flat double array addressed by name.  Booleans are 0/1, ints truncate. */
#define CFG_LIST(X) \
    X(start_velocity) X(max_velocity) X(accel_per_sec) \
    X(stumble_slowdown) X(stumble_recover_rate) X(lane_edge) X(lat_tau) \
    X(lane_recenter_time) X(turn_center_tol) X(stumble_kill_window) \
    X(compound_prob) X(compound_after_distance) X(compound_land_cells) \
    X(jumpslide_prob) X(jumpslide_after_distance) X(jumpslide_gap_cells) X(jumpslide_flame_frac) \
    X(jumpnarrow_prob) X(jumpnarrow_after_distance) X(jumpnarrow_gap_cells) X(jumpnarrow_strip_cells) \
    X(jumpgap_prob) X(jumpgap_after_distance) X(jumpgap_gap_cells) X(jumpgap_hole_cells) X(jumpgap_chain_max) \
    X(slidegap_prob) X(slidegap_after_distance) X(slidegap_gap_cells) X(slidegap_hole_cells) \
    X(turnobstacle_prob) \
    X(min_space_between_obstacles) X(min_space_between_turns) \
    X(device_difficulty_ramp) X(double_obstacle_percent) \
    X(min_spaces_between_bonus) X(obstacle_prob) X(max_back_to_back_obstacles) \
    X(turn_prob) X(ramp_distance) X(min_space_between_turns_late) X(turn_prob_late) \
    X(min_clear_after_turn) \
    X(turn_exec_early) X(turn_exec_late) X(turn_center_cprog) X(turn_offset_scale) \
    X(turn_lane_carry_buffered) X(turn_lane_carry) X(turn_land_kill) X(turn_window_frac) \
    X(turn_tail_cp_lo) X(turn_tail_cp_hi) X(turn_tail_pmax) X(turn_exec_lat_frames) \
    X(turn_approach_buffer) X(hook_turn_buffer) X(turn_buffer_ttl) X(turn_buffer_late_cp) \
    X(turn_send_lat_frames) X(turn_offjunction_drop) X(turn_drop_debounce_s) \
    X(post_land_gap_rejump_prob) X(held_jump_rejump_solid) X(held_jump_rejump_gap) \
    X(turn_early_frac) X(turn_late_frac) X(turn_buffer_cells) \
    X(slides_after_distance) X(default_path_level) \
    X(zone_change_every_cells) X(zone_after_distance) X(zone_up_prob) \
    X(narrows_after_distance) X(narrow_prob) X(narrow_lane_center) X(narrow_lane_tol) \
    X(strafe_off_edge_kills) X(turn_while_strafing_kills) \
    X(wide_gap_prob) X(wide_gap_extra_cells) X(wide_gap_min_airtime) \
    X(turn_linger_frames) X(zone_min_level) X(zone_max_level) \
    X(zone_gap_extra_cells_high) X(zone_gap_extra_cells_lowmid) \
    X(stepup_min_height) X(stepup_clip_band) X(stepup_min_takeoff_prog) X(gap_entry_grace) \
    X(wide_gap_landing_model) X(wide_gap_span_a) X(wide_gap_span_b) X(wide_gap_land_grace) \
    X(hole_landing_model) X(hole_trap_apex_frac) X(hole_trap_kill_prob) \
    X(chain_coast_kill_prob) X(chain_coast_gap) \
    X(land_on_hazard_kill_prob) X(stack_margin_cells) \
    X(airborne_slide_kill_prob) \
    X(crevasse_single_cell_prob) X(turn_aware_lookahead) \
    X(tight_pair_prob) X(tight_pair_min_cells) \
    X(crevasse_apex_all) X(crevasse_graze_worldy) X(crevasse_jump_peak) \
    X(crevasse_cap_height_hi) X(crevasse_graze_band) X(crevasse_cap_arc_offset) \
    X(crevasse_takeoff_std) \
    X(stepdown_grounded_ok) X(stepdown_grounded_max_level) \
    X(pair_same_type_prob) X(pair_same_type_cells) X(pair_anti_repeat_prob) \
    X(type_reject_model) X(type_reject_redraws) X(turnobstacle3_prob) \
    X(path_type_straight) X(path_type_turn) X(path_type_t_junction) \
    X(score_per_distance) X(start_score) X(coin_score) \
    X(dist_per_velsec) X(dist_per_velsec_slope) X(start_distance) \
    X(curriculum_max_start_dist) X(curriculum_deep_start_frac) X(curriculum_dist_power) \
    X(jump_duration) X(jump_drop_prob) X(jump_refuse_strafing_prob) X(gap_jump_lip_cp) \
    X(strafe_exec_lat_frames) X(jump_exec_lat_frames) X(jump_exec_lat_jitter) \
    X(slide_duration) X(strafe_duration) X(strafe_glide_time) X(strafe_reach) X(stumble_duration) \
    X(cell_length) X(cell_length_real) X(replay_start_into_cell) X(lookahead) \
    X(min_spaces_between_coin_runs) X(max_coins_per_run) \
    X(domain_rand) \
    X(dr_turn_window_frac_lo) X(dr_turn_window_frac_hi) \
    X(dr_turn_early_frac_lo) X(dr_turn_early_frac_hi) \
    X(dr_turn_late_frac_lo) X(dr_turn_late_frac_hi) \
    X(dr_jump_duration_lo) X(dr_jump_duration_hi) \
    X(jump_dur_stepdown_scale) X(jump_dur_stepup_scale) \
    X(dr_slide_duration_lo) X(dr_slide_duration_hi) \
    X(dr_velocity_scale_lo) X(dr_velocity_scale_hi) \
    X(dr_obstacle_prob_lo) X(dr_obstacle_prob_hi) \
    X(dr_turn_prob_lo) X(dr_turn_prob_hi) \
    X(dr_cell_length_lo) X(dr_cell_length_hi) \
    X(cliff_edge_lane) X(cliff_edge_kill_frames) X(jump_hold_window)

enum {
#define X(n) K_##n,
    CFG_LIST(X)
#undef X
    K_COUNT
};
static const char* CFG_NAMES[K_COUNT] = {
#define X(n) #n,
    CFG_LIST(X)
#undef X
};
static double G[K_COUNT];
#define C_(n) G[K_##n]
#define Ci(n) ((int)G[K_##n])
#define Cb(n) (G[K_##n] != 0.0)

/* ------------------------------------------------------------ side tables */
#define MAX_VC 64
static int    VC_N = 0;
static double VC_D[MAX_VC], VC_V[MAX_VC];

#define MAX_RAMP 16
static int    RAMP_N = 0;
static double RAMP[MAX_RAMP][6];   /* from_dist, min_obst, min_turns, dbl, narrows, flame */

#define MAX_OW 12
static int    OW_N = 0;
static int    OW_T[MAX_OW];
static double OW_W[MAX_OW];

#define MAX_NLW 8
static int    NLW_N = 0;
static int    NLW_L[MAX_NLW];
static double NLW_W[MAX_NLW];

/* type_reject: P(reject | prev hazard type, freshly drawn type), indexed [prev][cur].
 * Obstacle types are 1..7, so an 8x8 matrix covers them with the 0 row/column unused.
 * Zero everywhere = no rejection = no rng draw (see sample_obstacle_type_reject). */
#define TR_DIM 8
static double TR[TR_DIM][TR_DIM];

/* --------------------------------------------------- episode_profiles ---- */
/* Ported from sim_env_torch.py (_apply_profiles).  The whitelist below is the
 * torch key_map; anything else is rejected by sim_profile_key_id(). */
enum {
    P_obstacle_prob = 0, P_turn_prob, P_turn_prob_late,
    P_min_space_between_turns, P_min_space_between_turns_late,
    P_narrows_after_distance, P_zone_after_distance, P_zone_change_every_cells, P_zone_up_prob,
    P_min_space_between_obstacles, P_compound_prob, P_wide_gap_prob,
    /* SCENARIO knobs [2026-07-28]. Without these an episode profile can only change DENSITY and
       SPACING, so no dedicated EDGE-CASE DRILL is expressible -- you cannot write "an episode full
       of jump->slide traps". The faithful base world and the targeted drill are two different jobs;
       the profiles were only able to do the first. */
    P_jumpslide_prob, P_jumpnarrow_prob, P_jumpgap_prob, P_slidegap_prob,
    P_airborne_slide_kill_prob,
    P_KEY_COUNT
};
static const char* PROFILE_KEYS[P_KEY_COUNT] = {
    "obstacle_prob", "turn_prob", "turn_prob_late",
    "min_space_between_turns", "min_space_between_turns_late",
    "narrows_after_distance", "zone_after_distance", "zone_change_every_cells", "zone_up_prob",
    "min_space_between_obstacles", "compound_prob", "wide_gap_prob",
    "jumpslide_prob", "jumpnarrow_prob", "jumpgap_prob", "slidegap_prob",
    "airborne_slide_kill_prob"
};
/* which profile keys are integer-valued in torch (t.dtype == int64) */
static const int PROFILE_IS_INT[P_KEY_COUNT] = {0,0,0, 1,1, 0,0,1, 1,0,0, 0,0,0,0,0};

#define MAX_PROFILES 16
typedef struct {
    double weight;
    int    n;
    int    key[P_KEY_COUNT];
    double val[P_KEY_COUNT];
} Profile;
static int     PROF_N = 0;
static Profile PROF[MAX_PROFILES];
static double  PROF_CDF[MAX_PROFILES];

/* --------------------------------------------------------- env parameters */
static double EP_step_dt = 0.1;
static int    EP_max_steps = 1000;
static double EP_reward_scale = 0.1;
static int    EP_reward_score = 0;      /* 0 = "distance" */
static double EP_death_penalty = 1.0;
static int    EP_K = 0;                 /* action_history_len */
static int    EP_FS = 1;                /* frame_stack        */
static int    EP_pj_vel_inv = 0;

enum {
    SHP_action_cost = 0, SHP_reaction_bonus, SHP_turn_spam_penalty,
    SHP_idle_action_penalty, SHP_redundant_action_penalty,
    SHP_airborne_action_penalty, SHP_jump_hold_penalty, SHP_strafe_penalty,
    SHP_narrow_hold_bonus, SHP_aim_center_weight, SHP_velocity_drop_penalty,
    SHP_off_center_penalty, SHP_slide_jump_penalty, SHP_turn_buffer_bonus,
    SHP_stepdown_bonus_gate, SHP_landing_hazard_penalty,
    SHP_slide_landing_hazard_penalty,
    SHP_stepdown_jump_penalty,
    SHP_margin_bonus,
    SHP_stack_margin_penalty,
    SHP_turn_takeoff_penalty,
    SHP_COUNT
};
static double SHP[SHP_COUNT];

/* ---------------------------------------------------------- injected track */
typedef struct {
    short type, level, obst, nlane;
    unsigned char flags, _pad[3];
} Cell;

#define MAX_INJ 4096
static int  INJ_N = 0;
static Cell INJ[MAX_INJ];

/* ================================================================ STATE == */
typedef struct {
    /* rng */
    uint64_t rng_state, rng_inc;
    uint64_t draws;
    int      norm_has_spare; double norm_spare;

    /* per-episode config (domain randomisation + episode_profiles) */
    double e_jump_duration, e_slide_duration, e_cell_length, e_vel_scale;
    double e_turn_window_frac, e_turn_early_frac, e_turn_late_frac;
    double e_obstacle_prob, e_turn_prob, e_turn_prob_late;
    double e_min_turns, e_min_turns_late;
    double e_narrows_after, e_zone_after, e_zone_every, e_min_obst, e_zone_up;
    double e_compound_prob, e_wide_gap_prob;
    double e_jumpslide_prob, e_jumpnarrow_prob, e_jumpgap_prob, e_slidegap_prob;
    double e_airborne_slide_kill;
    int    profile_idx;

    /* runner */
    double lane, lat_cmd, lat_target, velocity, stumble_vel;
    int    has_stumble_vel;
    double distance, distance_real, score, dist_into_cell;
    double jump_t, slide_t, stumble_t, stumble_kill_t, strafe_t, turn_debounce_t;
    double jump_takeoff_pos; int has_takeoff;
    int    tilt_dir, facing;
    int    cell_index, gen_head;
    int    is_dead, death_type;
    int    frames_since_land, cliff_edge_frames, pending_jump_frames;
    int    slide_air_rolled;   /* cell whose airborne-slide kill roll was already made */
    int    jump_held_airborne;
    int    gap_grace_cell;           /* -1 = None */
    int    turn_landed;
    int    stumble_count;
    int    post_turn_linger, linger_can_l, linger_can_r;
    int    pending_turn;             /* 0 none, 1 left, 2 right */
    int    pending_turn_ttl;
    double lat_buf[LATBUF_CAP]; int lat_head, lat_len;
    int    jump_recency, slide_recency, last_action, has_last_action;

    /* generation */
    int cells_since_obstacle, cells_since_turn, cells_since_zone, cells_since_bonus;
    int zone_level, b2b_obstacles, pending_double;
    int pending_gap, pending_gap_type, pending_gap_level, pending_gap_obst, pending_gap_stepup;
    int pending_gap_sdlow;
    int last_obst_type, cells_since_hazard;   /* pair_same_type_prob */
    int last_hz_end_cell;      /* cell where the previous hazard STRUCTURE ended */
    int last_hz_type;          /* its obstacle type, to collapse multi-cell structures */
    int acted_since_hazard;    /* a jump/slide STARTED since that hazard -> not coasting */
    int pending_narrow, pending_narrow_lane, pending_narrow_type;
    Cell qseq[QUEUE_CAP];  int qseq_head, qseq_len;   /* _pending_sequence  */
    Cell qpg[QUEUE_CAP];   int qpg_head,  qpg_len;    /* _pending_post_gap  */
    int  inj_cursor;

    /* episode bookkeeping */
    int    elapsed, episodes;
    double ep_ret; int ep_len; double ep_dist;

    /* obs history */
    float  frames[MAX_FS][OBS_BASE]; int n_frames;
    int    act_hist[MAX_K]; int act_hist_n;

    /* previous snapshot (raw) for shaping */
    double prev_raw[RAW_CAP]; int has_prev;
    double prev_distance, prev_score;

    Cell track[TRACK_RING];
} State;

TRS_EXPORT int sim_state_size(void) { return (int)sizeof(State); }
TRS_EXPORT int sim_obs_base(void)   { return OBS_BASE; }
TRS_EXPORT int sim_raw_len(void)    { return RAW_LEN; }
TRS_EXPORT int sim_obs_len(void) {
    int fs = EP_FS < 1 ? 1 : EP_FS;
    return fs * OBS_BASE + EP_K * N_ACTIONS;
}

/* ================================================================= RNG === */
static inline uint32_t pcg32(State* s) {
    uint64_t old = s->rng_state;
    s->rng_state = old * 6364136223846793005ULL + s->rng_inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}
/* RNG REPLAY (testing only, cpp_sim/test_rng_order.py): when a value array is
 * installed every draw is taken from it instead of the PCG stream, so the C++
 * and the Python reference can be driven off the SAME random sequence.  Any
 * difference in draw ORDER or draw COUNT then shows up immediately as a state
 * divergence, which is the only way to test the procedural generator (the
 * deterministic equivalence tier injects fixed tracks and never generates). */
static const double* RPL_V = 0;
static int RPL_N = 0, RPL_I = 0, RPL_OVERRUN = 0;

/* one "draw" == one numpy Generator call (random / choice / integers / uniform) */
static inline double rnd(State* s) {
    s->draws++;
    if (RPL_V) {
        if (RPL_I < RPL_N) return RPL_V[RPL_I++];
        RPL_OVERRUN = 1;
        return 0.5;
    }
    uint64_t hi = (uint64_t)pcg32(s) >> 5;
    uint64_t lo = (uint64_t)pcg32(s) >> 6;
    return (double)((hi << 26) + lo) * (1.0 / 9007199254740992.0);
}
static double rnd_normal(State* s) {
    if (s->norm_has_spare) { s->norm_has_spare = 0; return s->norm_spare; }
    double u1, u2;
    do { u1 = rnd(s); } while (u1 <= 1e-12);
    u2 = rnd(s);
    double r = sqrt(-2.0 * log(u1)), th = 6.283185307179586 * u2;
    s->norm_spare = r * sin(th); s->norm_has_spare = 1;
    return r * cos(th);
}
static inline int rnd_int(State* s, int lo, int hi) {   /* inclusive, like integers(lo,hi+1) */
    int n = hi - lo + 1;
    if (n <= 1) return lo;
    int v = lo + (int)(rnd(s) * (double)n);
    return v > hi ? hi : v;
}
/* numpy Generator.choice(values, p=w/w.sum()): ONE random double, cdf searchsorted-right */
static int rnd_choice_w(State* s, const double* w, int n) {
    double tot = 0.0, u, c = 0.0;
    int i;
    for (i = 0; i < n; ++i) tot += w[i];
    u = rnd(s);
    for (i = 0; i < n; ++i) {
        c += w[i];
        if (c / tot > u) return i;
    }
    return n - 1;
}

/* ====================================================== velocity curve ==== */
/* Mirrors numpy.interp's exact arithmetic: slope = (dy1-dy0)/(dx1-dx0), then
 * slope*(x-dx0)+dy0, with numpy's edge cases. */
static double vel_curve(double d) {
    int n = VC_N;
    if (n <= 0) return 0.0;
    if (d < VC_D[0])      return VC_V[0];
    if (d > VC_D[n - 1])  return VC_V[n - 1];
    /* binary search: largest j with VC_D[j] <= d */
    int lo = 0, hi = n - 1, j;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (VC_D[mid] <= d) lo = mid; else hi = mid;
    }
    j = lo;
    if (j == n - 1)      return VC_V[j];
    if (VC_D[j] == d)    return VC_V[j];
    {
        double slope = (VC_V[j + 1] - VC_V[j]) / (VC_D[j + 1] - VC_D[j]);
        return slope * (d - VC_D[j]) + VC_V[j];
    }
}

/* ======================================================== track helpers === */
static inline Cell* cell_at(State* s, int idx) { return &s->track[idx & TRACK_MASK]; }
static double jump_airtime(State* s);   /* defined below; used by the takeoff sites above it */

static inline double cell_progress(State* s) {
    double cl = s->e_cell_length;
    double cp = s->dist_into_cell / (cl > 1e-6 ? cl : 1e-6);
    if (cp < 0.0) cp = 0.0;
    if (cp > 1.0) cp = 1.0;
    return cp;
}
static inline double long_pos(State* s) { return (double)s->cell_index + cell_progress(s); }

static void do_kill(State* s, int dt) { s->is_dead = 1; s->death_type = dt; }

static void do_stumble(State* s) {
    if (s->is_dead) return;
    if (s->stumble_t > 0.0 || s->stumble_kill_t < C_(stumble_kill_window)) { do_kill(s, 1); return; }
    s->velocity *= C_(stumble_slowdown);
    s->stumble_vel = s->velocity; s->has_stumble_vel = 1;
    s->lane = 0.0; s->tilt_dir = 0; s->strafe_t = 0.0;
    s->stumble_t = C_(stumble_duration);
    s->stumble_kill_t = 0.0;
    s->stumble_count++;
}

static inline void cell_clear(Cell* c) {
    c->type = 1; c->level = 2; c->obst = 0; c->nlane = 0; c->flags = 0;
}
static inline void cell_make(Cell* c, int type, int level, int obst) {
    c->type = (short)type; c->level = (short)level; c->obst = (short)obst;
    c->nlane = 0; c->flags = 0;
}

/* queue helpers (FIFO over a fixed ring) */
static inline void q_push(Cell* q, int* head, int* len, const Cell* c) {
    if (*len >= QUEUE_CAP) return;         /* defensive: never hit with sane configs */
    q[(*head + *len) % QUEUE_CAP] = *c;
    (*len)++;
}
static inline void q_pop(Cell* q, int* head, int* len, Cell* out) {
    *out = q[*head];
    *head = (*head + 1) % QUEUE_CAP;
    (*len)--;
}

/* ======================================================== GENERATION ====== */
static int narrow_ptype(int lane) { return lane == -1 ? 19 : 22; }

/* _ramp(): returns 0 when off, else fills r[5] = row[1:] */
static int ramp_row(State* s, double* r) {
    if (!Cb(device_difficulty_ramp)) return 0;
    int found = -1, i;
    for (i = 0; i < RAMP_N; ++i) {
        if (s->distance_real >= RAMP[i][0]) found = i; else break;
    }
    if (found < 0) return 0;
    r[0] = RAMP[found][1]; r[1] = RAMP[found][2]; r[2] = RAMP[found][3];
    r[3] = RAMP[found][4]; r[4] = RAMP[found][5];
    return 1;
}

static int sample_obstacle_type(State* s) {
    double w[MAX_OW]; int t[MAX_OW]; int n = 0, i;
    double r[5];
    int slides_ok;
    if (ramp_row(s, r)) slides_ok = (r[4] != 0.0);
    else                slides_ok = (s->distance_real >= C_(slides_after_distance));
    for (i = 0; i < OW_N; ++i) {
        if (!slides_ok && (OW_T[i] == 5 || OW_T[i] == 6)) continue;
        t[n] = OW_T[i]; w[n] = OW_W[i]; n++;
    }
    if (n == 0) return OBST_NONE;
    return t[rnd_choice_w(s, w, n)];
}

/* trees = the game's SEPARATE AddObstacle pass, which the pairwise filter never touches.
   Mirror of the Python ref's _TREE_TYPES. */
static inline int is_tree_o(int t) { return t == 3 || t == 4 || t == 5; }

/* DRAW-THEN-SUBSTITUTE (type_reject_model). Transcription of the Python reference's
   _sample_obstacle_type_reject -- see that docstring for the mechanism read out of dump.cs.
   DRAW ORDER, which is the whole risk: one draw for the pick, then per re-draw round at most one
   rejection roll and at most one further pick, and NO roll at all when the pair carries no
   rejection probability. Must match the reference value-for-value. */
static int sample_obstacle_type_reject(State* s) {
    int t = sample_obstacle_type(s);              /* OriginalPick */
    int prev = s->last_obst_type;
    if (!prev) return t;
    if (prev < 0 || prev >= TR_DIM) return t;
    int n = (int)C_(type_reject_redraws), i;
    for (i = 0; i < n; ++i) {
        if (is_tree_o(t)) break;                  /* Stage B: the second pass is unfiltered */
        if (t < 0 || t >= TR_DIM) break;
        double p = TR[prev][t];
        if (p <= 0.0) break;                      /* admissible pair -> no roll, no draw */
        if (rnd(s) >= p) break;                   /* accepted */
        t = sample_obstacle_type(s);              /* FinalPick (the substitute) */
    }
    return t;
}

/* Draw an obstacle type, sometimes REPEATING the previous one. SLIDES ONLY: on device 73% of all
   same-type pairs are repeated slides, while Hole->Hole is BELOW its independent rate. See the
   Python ref's _sample_obstacle_type_paired. prob>0-FIRST so the default draws no rng. */
static int sample_obstacle_type_paired(State* s) {
    /* MODEL SELECT, before any draw: OFF costs zero rng, so the pre-2026-07-28 stream is exact.
       CORRECTED 2026-07-28: this does NOT replace pair_same_type_prob. Stage A (geometry) is the
       rejection above; Stage B (trees) is the repeat below, and the game runs both -- rejection can
       only SUPPRESS, so it cannot produce the device's 6.5x-ELEVATED TreeSlide->TreeSlide. Treating
       them as competing models dropped slide clustering 7.9% -> 3.8% against a 17.9% target.
       Draw order: reject() first, then at most one repeat roll. Mirrors the Python reference. */
    if (Cb(type_reject_model)) {
        int tr = sample_obstacle_type_reject(s);
        double pr = C_(pair_same_type_prob);
        int lr = s->last_obst_type;
        if (pr > 0.0 && lr && (lr == 5 || lr == 6)
                && s->cells_since_hazard <= (int)C_(pair_same_type_cells)
                && rnd(s) < pr)
            return lr;
        return tr;
    }
    double p = C_(pair_same_type_prob);
    int lt = s->last_obst_type;
    if (p > 0.0 && (lt == 5 || lt == 6)
            && s->cells_since_hazard <= (int)C_(pair_same_type_cells)
            && rnd(s) < p)
        return lt;
    int t = sample_obstacle_type(s);
    /* ANTI-REPETITION for JUMP obstacles (device Hole->Hole 6.1% vs an independent 10.3%). The
       types are HARDCODED to (1, 4) here, exactly as pair_same_type_types is hardcoded to (5, 6);
       cpp_vec_env.py rejects any other list loudly rather than diverging silently. */
    double q = C_(pair_anti_repeat_prob);
    if (q > 0.0 && lt && t == lt && (t == 1 || t == 4)
            && s->cells_since_hazard <= (int)C_(pair_same_type_cells)
            && rnd(s) < q)
        return sample_obstacle_type(s);           /* one re-draw; may land on t again */
    return t;
}

static void queue_compound(State* s, int lane_a, int level, int strip_a) {
    Cell e; int i;
    int rest = strip_a - 1; if (rest < 0) rest = 0;
    for (i = 0; i < rest; ++i) {
        cell_make(&e, narrow_ptype(lane_a), level, OBST_NONE);
        e.nlane = (short)lane_a; e.flags |= F_NARROW;
        q_push(s->qseq, &s->qseq_head, &s->qseq_len, &e);
    }
    cell_make(&e, 2, level, OBST_TREE_JUMP);
    q_push(s->qseq, &s->qseq_head, &s->qseq_len, &e);
    int land = Ci(compound_land_cells); if (land < 1) land = 1;
    for (i = 0; i < land; ++i) {
        cell_make(&e, 2, level, OBST_NONE);
        q_push(s->qseq, &s->qseq_head, &s->qseq_len, &e);
    }
    int lane_b = -lane_a;
    int strip_b = NLW_L[rnd_choice_w(s, NLW_W, NLW_N)];
    if (strip_b < 1) strip_b = 1;
    for (i = 0; i < strip_b; ++i) {
        cell_make(&e, narrow_ptype(lane_b), level, OBST_NONE);
        e.nlane = (short)lane_b; e.flags |= F_NARROW;
        q_push(s->qseq, &s->qseq_head, &s->qseq_len, &e);
    }
}

static void queue_jumpslide(State* s, int level) {
    Cell e; int i;
    int gap = Ci(jumpslide_gap_cells); if (gap < 0) gap = 0;
    for (i = 0; i < gap; ++i) {
        cell_make(&e, Ci(path_type_straight), level, OBST_NONE);
        q_push(s->qpg, &s->qpg_head, &s->qpg_len, &e);
    }
    int is_flame = (rnd(s) < C_(jumpslide_flame_frac));
    int sob = is_flame ? OBST_FLAME_TOWER : OBST_TREE_SLIDE;
    int spt = is_flame ? 31 : Ci(path_type_straight);
    cell_make(&e, spt, level, sob);
    q_push(s->qpg, &s->qpg_head, &s->qpg_len, &e);
}

static void queue_jumpnarrow(State* s, int level) {
    Cell e; int i;
    int gap = Ci(jumpnarrow_gap_cells); if (gap < 0) gap = 0;
    for (i = 0; i < gap; ++i) {
        cell_make(&e, Ci(path_type_straight), level, OBST_NONE);
        q_push(s->qpg, &s->qpg_head, &s->qpg_len, &e);
    }
    int lane = (rnd(s) < 0.5) ? -1 : 1;
    int strip = Ci(jumpnarrow_strip_cells); if (strip < 1) strip = 1;
    for (i = 0; i < strip; ++i) {
        cell_make(&e, narrow_ptype(lane), level, OBST_NONE);
        e.nlane = (short)lane; e.flags |= F_NARROW;
        q_push(s->qpg, &s->qpg_head, &s->qpg_len, &e);
    }
}

static void queue_jumpgap(State* s, int level) {
    Cell e; int i, k;
    int n_rep = 1;
    int chain_max = Ci(jumpgap_chain_max);
    if (chain_max > 1) n_rep = 1 + rnd_int(s, 0, chain_max - 1);
    int gap = Ci(jumpgap_gap_cells); if (gap < 0) gap = 0;
    int n_hole = Ci(jumpgap_hole_cells); if (n_hole < 1) n_hole = 1;
    for (k = 0; k < n_rep; ++k) {
        for (i = 0; i < gap; ++i) {
            cell_make(&e, Ci(path_type_straight), level, OBST_NONE);
            q_push(s->qpg, &s->qpg_head, &s->qpg_len, &e);
        }
        for (i = 0; i < n_hole; ++i) {
            cell_make(&e, n_hole > 1 ? 12 : Ci(path_type_straight), level, OBST_HOLE);
            if (n_hole > 1) e.flags |= F_WIDE;
            q_push(s->qpg, &s->qpg_head, &s->qpg_len, &e);
        }
    }
}

static void queue_slidegap(State* s, int level) {
    Cell e; int i;
    int gap = Ci(slidegap_gap_cells); if (gap < 0) gap = 0;
    for (i = 0; i < gap; ++i) {
        cell_make(&e, Ci(path_type_straight), level, OBST_NONE);
        q_push(s->qpg, &s->qpg_head, &s->qpg_len, &e);
    }
    int n_hole = Ci(slidegap_hole_cells); if (n_hole < 1) n_hole = 1;
    for (i = 0; i < n_hole; ++i) {
        cell_make(&e, n_hole > 1 ? 12 : Ci(path_type_straight), level, OBST_HOLE);
        if (n_hole > 1) e.flags |= F_WIDE;
        q_push(s->qpg, &s->qpg_head, &s->qpg_len, &e);
    }
}

/* _next_cell_replay(): injected element, else a neutral straight. */
static void next_cell_injected(State* s, Cell* out) {
    if (s->inj_cursor < INJ_N) {
        *out = INJ[s->inj_cursor];
        s->inj_cursor++;
        return;
    }
    cell_make(out, Ci(path_type_straight), Ci(default_path_level), OBST_NONE);
}

/* _next_cell_procedural() -- transcribed branch for branch. */
static void next_cell_procedural(State* s, Cell* out) {
    int level = s->zone_level;
    s->cells_since_obstacle++;
    s->cells_since_turn++;
    s->cells_since_bonus++;
    s->cells_since_zone++;

    if (s->qseq_len > 0) {
        s->cells_since_obstacle = 0;
        q_pop(s->qseq, &s->qseq_head, &s->qseq_len, out);
        return;
    }
    if (s->pending_gap > 0) {
        s->pending_gap--;
        s->cells_since_obstacle = 0;
        cell_make(out, s->pending_gap_type, s->pending_gap_level, s->pending_gap_obst);
        out->flags |= F_WIDE;
        if (s->pending_gap_sdlow) out->flags |= F_SDLOW;
        if (s->pending_gap == 0 && s->pending_gap_stepup) {
            out->flags |= F_STEPUP;
            s->pending_gap_stepup = 0;
        }
        return;
    }
    if (s->qpg_len > 0) {
        s->cells_since_obstacle = 0;
        q_pop(s->qpg, &s->qpg_head, &s->qpg_len, out);
        return;
    }
    if (s->pending_narrow > 0) {
        s->pending_narrow--;
        s->cells_since_obstacle = 0;
        cell_make(out, s->pending_narrow_type, s->zone_level, OBST_NONE);
        out->nlane = (short)s->pending_narrow_lane;
        out->flags |= F_NARROW;
        return;
    }

    int wide = 0, stepup = 0, has_narrow = 0, narrow_lane = 0, sdlow = 0;

    double r[5];
    int has_ramp = ramp_row(s, r);

    double d = s->distance_real - C_(start_distance);
    double rd = C_(ramp_distance); if (rd < 1.0) rd = 1.0;
    double diff = d / rd;
    if (diff < 0.0) diff = 0.0;
    if (diff > 1.0) diff = 1.0;
    double min_turns = has_ramp ? r[1]
                     : (s->e_min_turns * (1.0 - diff) + s->e_min_turns_late * diff);
    double turn_p = s->e_turn_prob * (1.0 - diff) + s->e_turn_prob_late * diff;

    int is_turn = 0, can_l = 0, can_r = 0;
    int ptype = Ci(path_type_straight);
    if ((double)s->cells_since_turn >= min_turns) {
        if (rnd(s) < turn_p) {
            is_turn = 1;
            s->cells_since_turn = 0;
            double roll = rnd(s);
            if (roll < 0.4)      { can_l = 1; can_r = 0; ptype = Ci(path_type_turn); }
            else if (roll < 0.8) { can_l = 0; can_r = 1; ptype = Ci(path_type_turn); }
            else                 { can_l = 1; can_r = 1; ptype = Ci(path_type_t_junction); }
        }
    }

    int obstacle = OBST_NONE;
    int min_obst_gate = has_ramp ? (int)r[0] : (int)s->e_min_obst;
    if (!is_turn
            && s->cells_since_turn >= Ci(min_clear_after_turn)
            && s->cells_since_obstacle >= min_obst_gate
            && s->cells_since_zone >= (int)s->e_zone_every
            && s->distance_real >= s->e_zone_after) {
        obstacle = OBST_LEVEL_CHANGE;
        s->cells_since_zone = 0;
        s->cells_since_obstacle = 0;
        s->b2b_obstacles = 0;
        int lo = Ci(zone_min_level), hi = Ci(zone_max_level);
        int prev_zone = s->zone_level;
        if (s->zone_level <= lo)      s->zone_level += 1;
        else if (s->zone_level >= hi) s->zone_level -= 1;
        else {
            /* The level walk was hardcoded 65% DOWN / 35% up, so HIGH ground (level 3) -- where the
               user's mid->high cap graze happens -- is reached rarely, and raising
               zone_change_every_cells made level-3 caps per km go DOWN (0.55 -> 0.32) rather than up.
               zone_up_prob exposes the bias so a drill profile can actually target the transition.
               Default 0.35 reproduces the original literal exactly. */
            s->zone_level += (rnd(s) < (1.0 - s->e_zone_up) ? -1 : 1);
        }
        level = s->zone_level;
        double p1 = C_(crevasse_single_cell_prob);
        s->pending_gap_obst  = OBST_LEVEL_CHANGE;
        s->pending_gap_type  = 35;
        s->pending_gap_level = s->zone_level;
        ptype = 35;
        if (p1 > 0.0 && rnd(s) < p1) {
            s->pending_gap = 0;
            stepup = 0;
            wide = 0;
        } else {
            s->pending_gap = 1;
            s->pending_gap_stepup = (s->zone_level > prev_zone) ? 1 : 0;
            stepup = 0;
            wide = 1;
        }
        sdlow = (Cb(stepdown_grounded_ok) && s->zone_level < prev_zone
                 && s->zone_level <= (int)C_(stepdown_grounded_max_level)) ? 1 : 0;
        s->pending_gap_sdlow = sdlow;
    }

    /* POST-TURN OBSTACLE */
    if (obstacle == OBST_NONE && !is_turn && s->cells_since_turn == 1
            && C_(turnobstacle_prob) > 0.0
            && rnd(s) < C_(turnobstacle_prob)) {
        obstacle = sample_obstacle_type_paired(s);
        s->cells_since_obstacle = 0;
    }
    /* the DEVICE concentrates post-turn hazards at +3, not +1 (device +3 = 36.6% of junctions vs
       +1 = 3.2%); turnobstacle_prob can only fill +1, so the peak needs its own knob. Mirror of the
       Python ref. */
    else if (obstacle == OBST_NONE && !is_turn && s->cells_since_turn == 3
            && C_(turnobstacle3_prob) > 0.0
            && rnd(s) < C_(turnobstacle3_prob)) {
        obstacle = sample_obstacle_type_paired(s);
        s->cells_since_obstacle = 0;
    }

    /* TIGHT PAIRS -- prob>0 first so the default draws nothing */
    double tp = C_(tight_pair_prob);
    int tight = 0;
    if (tp > 0.0 && obstacle == OBST_NONE && !is_turn
            && s->cells_since_turn >= Ci(min_clear_after_turn)
            && Ci(tight_pair_min_cells) <= s->cells_since_obstacle
            && s->cells_since_obstacle < (int)s->e_min_obst
            && rnd(s) < tp) {
        tight = 1;
    }

    if (obstacle == OBST_NONE && !is_turn
            && s->cells_since_turn >= Ci(min_clear_after_turn)
            && (s->cells_since_obstacle >= min_obst_gate || tight || s->pending_double)) {
        int place = (rnd(s) < s->e_obstacle_prob);
        if (s->b2b_obstacles >= Ci(max_back_to_back_obstacles)) place = 0;
        if (place) {
            s->cells_since_obstacle = 0;
            s->b2b_obstacles++;
            double dbl = has_ramp ? r[2] : C_(double_obstacle_percent);
            s->pending_double = (dbl > 0.0
                                 && s->b2b_obstacles < Ci(max_back_to_back_obstacles)
                                 && rnd(s) < dbl) ? 1 : 0;
            int narrows_ok = has_ramp ? (r[3] != 0.0)
                                      : (s->distance_real >= s->e_narrows_after);
            if (narrows_ok && rnd(s) < C_(narrow_prob)) {
                narrow_lane = (rnd(s) < 0.5) ? -1 : 1;
                ptype = narrow_ptype(narrow_lane);
                int strip = NLW_L[rnd_choice_w(s, NLW_W, NLW_N)];
                if (s->e_compound_prob > 0.0
                        && s->distance_real >= C_(compound_after_distance)
                        && rnd(s) < s->e_compound_prob) {
                    queue_compound(s, narrow_lane, level, strip);
                    ptype = narrow_ptype(narrow_lane);
                } else {
                    s->pending_narrow = strip - 1;
                    s->pending_narrow_lane = narrow_lane;
                    s->pending_narrow_type = ptype;
                }
                has_narrow = 1;
            } else {
                obstacle = sample_obstacle_type_paired(s);
                if (obstacle == OBST_HOLE && rnd(s) < s->e_wide_gap_prob) {
                    s->pending_gap = Ci(wide_gap_extra_cells);
                    s->pending_gap_obst = OBST_HOLE;
                    s->pending_gap_type = 12;
                    s->pending_gap_level = level;
                    s->pending_gap_sdlow = 0;   /* hole, not a crevasse: clear the step-down mark */
                    ptype = 12;
                    wide = 1;
                }
            }
        } else {
            s->b2b_obstacles = 0;
        }
    } else if (!is_turn) {
        s->b2b_obstacles = 0;
    }

    /* post-wide-gap / post-slide compound tails */
    int queued_tail = 0;
    if (wide && s->e_jumpslide_prob > 0.0
            && s->distance_real >= C_(jumpslide_after_distance)
            && rnd(s) < s->e_jumpslide_prob) {
        queue_jumpslide(s, level); queued_tail = 1;
    }
    if (wide && !queued_tail && s->e_jumpnarrow_prob > 0.0
            && s->distance_real >= C_(jumpnarrow_after_distance)
            && rnd(s) < s->e_jumpnarrow_prob) {
        queue_jumpnarrow(s, level); queued_tail = 1;
    }
    if (wide && !queued_tail && s->e_jumpgap_prob > 0.0
            && s->distance_real >= C_(jumpgap_after_distance)
            && rnd(s) < s->e_jumpgap_prob) {
        queue_jumpgap(s, level); queued_tail = 1;
    }
    if (must_slide_o(obstacle) && !queued_tail && s->e_slidegap_prob > 0.0
            && s->distance_real >= C_(slidegap_after_distance)
            && rnd(s) < s->e_slidegap_prob) {
        queue_slidegap(s, level);
    }

    int is_coin = 0;
    if (!is_turn && obstacle == OBST_NONE
            && s->cells_since_bonus >= Ci(min_spaces_between_bonus)) {
        if (rnd(s) < 0.5) { is_coin = 1; s->cells_since_bonus = 0; }
    }

    cell_make(out, ptype, level, obstacle);
    out->flags = (unsigned char)((is_turn ? F_TURN : 0) | (can_l ? F_CANL : 0)
               | (can_r ? F_CANR : 0) | (is_coin ? F_COIN : 0)
               | (wide ? F_WIDE : 0) | (stepup ? F_STEPUP : 0) | (sdlow ? F_SDLOW : 0)
               | (has_narrow ? F_NARROW : 0));
    out->nlane = (short)narrow_lane;
}

static void next_cell(State* s, Cell* out) {
    if (INJ_N > 0) next_cell_injected(s, out);
    else           next_cell_procedural(s, out);
}

/* _extend_track(): keep lookahead+1 cells ahead of the player (same count as
 * the reference, so procedural cells are generated at the same distances). */
static void extend_track(State* s) {
    int want = s->cell_index + Ci(lookahead) + 1;
    if (want > s->cell_index + TRACK_RING - 1) want = s->cell_index + TRACK_RING - 1;
    while (s->gen_head < want) {
        Cell* e = cell_at(s, s->gen_head);
        next_cell(s, e);
        /* mirror of the Python ref: track the previous HAZARD for pair_same_type_prob here, the
           single point every generated cell passes through (covers all the early returns). */
        if (must_jump_o(e->obst) || must_slide_o(e->obst)) {
            s->last_obst_type = e->obst; s->cells_since_hazard = 0;
        } else s->cells_since_hazard++;
        s->gen_head++;
    }
}

/* =========================================================== DEATH MODELS = */
static void resolve_obstacle_on_enter(State* s, Cell* c) {
    /* ---- TIGHT-CHAIN COAST [DEVICE-MEASURED 2026-07-28] -------------------------------------
       A hazard arriving within `chain_coast_gap` cells of the previous one, with NO new jump/slide
       started since that previous hazard, is the configuration that kills the champion on the phone:
       device COAST 8/17 = 47% lethal vs 5% when a fresh input is taken, and it kills at the SAME
       rate whether the arc lands on the hazard (50%) or past it (45%) -- i.e. it is "no action was
       available", not a landing-position problem. In the sim the identical situation was only ~10%
       lethal (6.2% even when landing ON the hazard), so coasting was nearly FREE and nothing ever
       taught the policy to act twice in quick succession. Same shape as the held-jump gap.
       ⚠️ CONDITIONAL ON PURPOSE: the sim is ALREADY harsher than the device on ISOLATED hazards
       (hole 2.89%/encounter vs 1.64%), so a global dose increase would be wrong -- the error is
       distributional, not a level. Default 0.0 = off. Gated on prob > 0 BEFORE any draw so the
       default RNG stream is bit-identical. */
    if (must_jump_o(c->obst) || must_slide_o(c->obst)) {
        /* SAME STRUCTURE? A 2-cell crevasse / HoleLong occupies two ADJACENT cells of the same
           type and needs ONE input, so its own second cell must not count as a following hazard --
           otherwise the rule fires against the structure itself (measured: episodes collapsed 4.8x).
           This mirrors chain3_forensic.structures(), which collapses adjacent same-type cells. */
        int same_struct = (s->cell_index == s->last_hz_end_cell + 1 && c->obst == s->last_hz_type);
        double ck = C_(chain_coast_kill_prob);
        if (!same_struct && ck > 0.0 && !s->acted_since_hazard && s->last_hz_end_cell >= -100
                && (s->cell_index - s->last_hz_end_cell) <= (int)C_(chain_coast_gap)) {
            if (rnd(s) < ck) {
                do_kill(s, c->obst == OBST_HOLE ? 1
                          : (c->obst == OBST_LEVEL_CHANGE ? 4
                          : (must_slide_o(c->obst) ? (c->obst == OBST_FLAME_TOWER ? 5 : 3) : 2)));
                return;
            }
        }
        s->last_hz_end_cell = s->cell_index;   /* this hazard becomes the reference for the next */
        s->last_hz_type = c->obst;
        if (!same_struct) s->acted_since_hazard = 0;  /* a fresh input must arrive before the NEXT structure */
    }
    if (c->flags & F_NARROW) {
        double target = (double)c->nlane * C_(narrow_lane_center);
        if (fabs(s->lane - target) > C_(narrow_lane_tol)) { do_kill(s, 1); return; }
    }
    int obst = c->obst;
    int apex_cell = (c->flags & F_STEPUP) ? 1 : 0;
    if (!apex_cell && (Cb(crevasse_apex_all) || Cb(crevasse_graze_worldy))
            && obst == OBST_LEVEL_CHANGE && (c->flags & F_WIDE)) {
        Cell* nxt = cell_at(s, s->cell_index + 1);
        apex_cell = !(nxt->flags & F_WIDE);
    }
    /* step-DOWN to low ground: no raised cap -> not an apex cell (mirror of the Python ref) */
    if (c->flags & F_SDLOW) apex_cell = 0;
    if (must_jump_o(obst) && apex_cell) {
        if (s->jump_t <= 0.0 || !s->has_takeoff) { do_kill(s, 4); return; }  /* Ledge: clipped the raised wall */
        /* a step-UP jump is device-measured ~26% SHORTER, so the apex parabola must use the arc the
           runner is ACTUALLY flying (mirror of the Python ref). */
        double span = s->e_jump_duration * C_(jump_dur_stepup_scale)
                    * s->velocity / s->e_cell_length;
        double d = (double)s->cell_index - s->jump_takeoff_pos;
        d += C_(crevasse_cap_arc_offset);
        double tk = C_(crevasse_takeoff_std);
        if (tk > 0.0) d += rnd_normal(s) * tk;
        double sp = span > 1e-6 ? span : 1e-6;
        double frac = d / sp;
        double t = 2.0 * frac - 1.0;
        double height = 1.0 - t * t;
        if (Cb(crevasse_graze_worldy)) {
            double arc_y = C_(crevasse_jump_peak) * height;
            double cap_h = (c->level >= 3) ? C_(crevasse_cap_height_hi) : 0.0;
            double band = C_(crevasse_graze_band);
            if (arc_y < cap_h - band)  do_kill(s, 4);   /* deep clip of the raised cap -> Ledge */
            else if (arc_y < cap_h)    do_stumble(s);
            return;
        }
        double mh = C_(stepup_min_height);
        if (height < mh) {
            double band = C_(stepup_clip_band);
            if (band > 0.0 && height >= mh - band) do_stumble(s);
            else                                   do_kill(s, 4);   /* Ledge */
        }
        return;
    }
    if (must_jump_o(obst)) {
        double need = (c->flags & F_WIDE) ? C_(wide_gap_min_airtime) : 0.0;
        if (s->jump_t <= need) {
            if (c->flags & F_SDLOW) return;   /* foot-crossable step-down to low ground */
            if ((c->flags & F_WIDE) && C_(gap_entry_grace) > 0.0) s->gap_grace_cell = s->cell_index;
            else do_kill(s, obst == OBST_HOLE ? 1 : (obst == OBST_LEVEL_CHANGE ? 4 : 2));
            return;
        }
        if (Cb(wide_gap_landing_model) && (c->flags & F_WIDE) && obst == OBST_HOLE
                && s->has_takeoff) {
            Cell* nxt = cell_at(s, s->cell_index + 1);
            if (!(nxt->flags & F_WIDE)) {
                double span = C_(wide_gap_span_a) * s->velocity + C_(wide_gap_span_b);
                double d = (double)s->cell_index - s->jump_takeoff_pos;
                if (span < d + 1.0 - C_(wide_gap_land_grace)) { do_kill(s, 4); return; }  /* far-wall clip -> Ledge */
            }
        }
        double hp = C_(hole_trap_kill_prob);
        if (hp > 0.0 && Cb(hole_landing_model) && obst == OBST_HOLE
                && !(c->flags & F_WIDE) && s->has_takeoff) {
            double span = C_(wide_gap_span_a) * s->velocity + C_(wide_gap_span_b);
            double sp = span > 1e-6 ? span : 1e-6;
            double frac = ((double)s->cell_index - s->jump_takeoff_pos) / sp;
            if (frac > C_(hole_trap_apex_frac) && (hp >= 1.0 || rnd(s) < hp)) {
                do_kill(s, 1); return;
            }
        }
        return;
    }
    /* GROUNDED-and-not-sliding is a skill failure and stays fatal. AIRBORNE is not: the runner
       could not have slid, the takeoff was forced by a must_jump, and the DEVICE kills only 24%
       there (23 of 94 over 45 captures) against the sim's 100%. Mirror of the Python reference. */
    if (must_slide_o(obst)) {
        if (s->jump_t > 0.0) {
            double pk = s->e_airborne_slide_kill;
            if (pk >= 1.0) {
                do_kill(s, obst == OBST_FLAME_TOWER ? 5 : 3);
            } else if (s->slide_air_rolled != s->cell_index) {
                s->slide_air_rolled = s->cell_index;      /* ONE roll per cell */
                if (rnd(s) < pk) do_kill(s, obst == OBST_FLAME_TOWER ? 5 : 3);
            }
        } else if (s->slide_t <= 0.0) {
            do_kill(s, obst == OBST_FLAME_TOWER ? 5 : 3);
        }
    }
}

/* ============================================================ TURN MODEL == */
static void execute_turn(State* s, Cell* c, int want_left, double swipe_cp, int from_buffer) {
    double center = C_(turn_center_cprog);
    double scale  = C_(turn_offset_scale);
    double kill   = C_(turn_land_kill);
    double carry  = from_buffer ? C_(turn_lane_carry_buffered) : C_(turn_lane_carry);
    if (!from_buffer) {
        double lat_frames = C_(turn_exec_lat_frames);
        if (lat_frames > 0.0)
            swipe_cp = swipe_cp + s->velocity * lat_frames * EP_step_dt / s->e_cell_length;
    }
    double sign = want_left ? 1.0 : -1.0;
    double offset = from_buffer ? 0.0
                  : (sign * (swipe_cp - center) * scale + s->lane * carry);
    static const int LEFT_OF[5]  = {0, 4, 1, 2, 3};   /* N->W E->N S->E W->S */
    static const int RIGHT_OF[5] = {0, 2, 3, 4, 1};   /* N->E E->S S->W W->N */
    s->facing = want_left ? LEFT_OF[s->facing] : RIGHT_OF[s->facing];
    if (fabs(offset) > kill) { do_kill(s, 1); return; }
    double lo = C_(turn_tail_cp_lo), hi = C_(turn_tail_cp_hi), pmax = C_(turn_tail_pmax);
    if (pmax > 0.0 && swipe_cp > lo) {
        double den = hi - lo; if (den < 1e-6) den = 1e-6;
        double p = (swipe_cp - lo) / den * pmax;
        if (p > pmax) p = pmax;
        if (rnd(s) < p) { do_kill(s, 1); return; }
    }
    s->lane = offset;
    s->tilt_dir = 0;
    s->turn_landed = 1;
    c->flags &= (unsigned char)~(F_TURN | F_CANL | F_CANR);
    s->cells_since_turn = 0;
    s->pending_turn = 0;
    int n = Ci(turn_linger_frames);
    if (n > 0) {
        s->post_turn_linger = n;
        s->linger_can_l = want_left ? 1 : 0;
        s->linger_can_r = want_left ? 0 : 1;
    }
}

static void attempt_turn(State* s, int a) {
    if (s->post_turn_linger > 0) { do_stumble(s); return; }
    double deb = C_(turn_drop_debounce_s);
    if (deb > 0.0) {
        if (s->turn_debounce_t > 0.0) return;
        s->turn_debounce_t = deb;
    }
    int want_left = (a == A_TURN_L);
    Cell* c = cell_at(s, s->cell_index);
    double cp = cell_progress(s);
    int side_ok = want_left ? ((c->flags & F_CANL) != 0) : ((c->flags & F_CANR) != 0);

    if (Cb(hook_turn_buffer) && (c->flags & F_TURN) && side_ok) {
        double slat = C_(turn_send_lat_frames);
        double proc_cp = cp + (slat > 0.0
            ? (s->velocity * slat * EP_step_dt / s->e_cell_length) : 0.0);
        if (proc_cp <= C_(turn_buffer_late_cp))
            execute_turn(s, c, want_left, C_(turn_center_cprog), 1);
        return;
    }
    double early = C_(turn_exec_early), late = C_(turn_exec_late);
    if ((c->flags & F_TURN) && cp >= early && cp <= late && side_ok) {
        execute_turn(s, c, want_left, cp, 0);
        return;
    }
    Cell* nxt = cell_at(s, s->cell_index + 1);
    if (Cb(turn_approach_buffer) && !(c->flags & F_TURN)) {
        int nside = want_left ? ((nxt->flags & F_CANL) != 0) : ((nxt->flags & F_CANR) != 0);
        if ((nxt->flags & F_TURN) && nside) {
            s->pending_turn = want_left ? 1 : 2;
            s->pending_turn_ttl = Ci(turn_buffer_ttl);
            return;
        }
    }
    if (Cb(turn_offjunction_drop) && Cb(hook_turn_buffer)) return;
    int near_junction = ((c->flags & F_TURN) != 0) || ((nxt->flags & F_TURN) != 0);
    if (near_junction) do_stumble(s);
    else               do_kill(s, 1);
}

/* ======================================================= _cross_into_next = */
static void cross_into_next_cell(State* s) {
    Cell* leaving = cell_at(s, s->cell_index);
    if ((leaving->flags & F_TURN) && (leaving->flags & (F_CANL | F_CANR))) {
        do_kill(s, 1);
        return;
    }
    s->cell_index++;
    extend_track(s);
    Cell* entered = cell_at(s, s->cell_index);
    if (s->pending_turn != 0) {
        int wl = (s->pending_turn == 1);
        int side = wl ? ((entered->flags & F_CANL) != 0) : ((entered->flags & F_CANR) != 0);
        if ((entered->flags & F_TURN) && side) {
            s->pending_turn = 0;
            execute_turn(s, entered, wl, C_(turn_center_cprog), 1);
            if (s->is_dead) return;
        } else {
            s->pending_turn_ttl--;
            if (s->pending_turn_ttl <= 0) s->pending_turn = 0;
        }
    }
    resolve_obstacle_on_enter(s, entered);
    if (s->is_dead) return;
    if (entered->flags & F_COIN) {
        s->score += C_(coin_score);
        entered->flags &= (unsigned char)~F_COIN;
    }
}

/* ========================================================== _apply_action = */
static void apply_action(State* s, int a) {
    if (a != A_JUMP) s->jump_held_airborne = 0;
    if (a == A_JUMP) {
        if (s->jump_t > 0.0) s->jump_held_airborne = 1;
        if (s->jump_t <= 0.0 && s->slide_t <= 0.0) {
            double jdp = C_(jump_drop_prob);
            if (jdp > 0.0 && rnd(s) < jdp) return;
            if (fabs(s->lat_target - s->lane) > 1.8
                    && rnd(s) < C_(jump_refuse_strafing_prob)) return;
            Cell* now = cell_at(s, s->cell_index);
            if ((now->flags & F_WIDE) && cell_progress(s) > C_(gap_jump_lip_cp)) {
                double plr = C_(post_land_gap_rejump_prob);
                int ok = (plr > 0.0 && s->frames_since_land <= 1 && rnd(s) < plr);
                if (!ok) return;
            }
            int lat = Ci(jump_exec_lat_frames);
            if (lat > 0 && s->pending_jump_frames <= 0) {
                int jit = Ci(jump_exec_lat_jitter);
                int f = lat + (jit > 0 ? rnd_int(s, -jit, jit) : 0);
                s->pending_jump_frames = (f < 1) ? 1 : f;
            } else if (lat <= 0) {
                s->jump_t = jump_airtime(s); s->acted_since_hazard = 1;
                s->jump_takeoff_pos = long_pos(s);
                s->has_takeoff = 1;
            }
        }
    } else if (a == A_SLIDE) {
        if (s->slide_t <= 0.0 && s->jump_t <= 0.0) { s->slide_t = s->e_slide_duration; s->acted_since_hazard = 1; }
    } else if (a == A_STRAFE_L) {
        s->lat_cmd = -C_(lane_edge); s->tilt_dir = -1;
    } else if (a == A_STRAFE_R) {
        s->lat_cmd = +C_(lane_edge); s->tilt_dir = +1;
    } else if (a == A_TURN_L || a == A_TURN_R) {
        attempt_turn(s, a);
    } else if (a == A_NOOP) {
        s->lat_cmd = 0.0; s->tilt_dir = 0;
    }
}

/* =============================================================== _advance = */
static void advance(State* s, double dt) {
    if (s->pending_jump_frames > 0) {
        s->pending_jump_frames--;
        if (s->pending_jump_frames == 0 && s->jump_t <= 0.0 && s->slide_t <= 0.0) {
            s->jump_t = jump_airtime(s);
            s->jump_takeoff_pos = long_pos(s);
            s->has_takeoff = 1;
        }
    }
    int was_airborne = (s->jump_t > 0.0);
    s->jump_t -= dt; if (s->jump_t < 0.0) s->jump_t = 0.0;
    if (was_airborne && s->jump_t <= 0.0) {
        s->frames_since_land = 0;
        /* ---- LANDING ONTO A HAZARD [DEVICE-MEASURED 2026-07-30] ---------------------------------
           The sim resolves a hazard ONLY when the runner ENTERS its cell. Enter a hole AIRBORNE and
           you are correctly waved through -- but if the arc then EXPIRES while still on that cell,
           nothing re-checks, so the runner simply stands in the hole and lives. On the phone that is
           a death, and it is the dominant one: a frame-level forensic of v111's 8 fatal frames found
           7 of 8 are "my arc ended inside a hole" (world_y - ground_height -7 to -44).
           MEASURED with the shipped encoder: P(death <=1.5s | landing_hazard=1 at takeoff) is 16.0%
           pooled / 25.9% today on DEVICE vs 2.14% / 0.68% when it is 0 -- a 7.5-38x lift. The sim
           discriminated only 1.2x (4.56% vs 3.65%), i.e. landing on a hazard was barely worse than
           not. Default 0.0 = off; gated on prob > 0 before any draw. */
        double lh = C_(land_on_hazard_kill_prob);
        if (lh > 0.0 && !s->is_dead) {
            Cell* lc = cell_at(s, s->cell_index);
            if (must_jump_o(lc->obst) && !(lc->flags & F_SDLOW)) {
                if (rnd(s) < lh) {
                    do_kill(s, lc->obst == OBST_HOLE ? 1
                              : (lc->obst == OBST_LEVEL_CHANGE ? 4 : 2));
                }
            }
        }
        double hjs = C_(held_jump_rejump_solid), hjg = C_(held_jump_rejump_gap);
        if (s->jump_held_airborne && (hjs > 0.0 || hjg > 0.0) && !s->is_dead) {
            Cell* c = cell_at(s, s->cell_index);
            double p = (c->flags & F_WIDE) ? hjg : hjs;
            if (p > 0.0 && rnd(s) < p) {
                s->jump_t = jump_airtime(s); s->acted_since_hazard = 1;
                s->jump_takeoff_pos = long_pos(s);
                s->has_takeoff = 1;
            }
        }
        s->jump_held_airborne = 0;
    } else {
        if (s->frames_since_land < 99) s->frames_since_land++;
    }
    s->slide_t -= dt;         if (s->slide_t < 0.0) s->slide_t = 0.0;
    s->turn_debounce_t -= dt; if (s->turn_debounce_t < 0.0) s->turn_debounce_t = 0.0;
    s->strafe_t -= dt;        if (s->strafe_t < 0.0) s->strafe_t = 0.0;

    int slat = Ci(strafe_exec_lat_frames);
    if (slat > 0) {
        if (slat > LATBUF_CAP - 1) slat = LATBUF_CAP - 1;
        s->lat_buf[(s->lat_head + s->lat_len) % LATBUF_CAP] = s->lat_cmd;
        s->lat_len++;
        if (s->lat_len > slat) {
            s->lat_target = s->lat_buf[s->lat_head];
            s->lat_head = (s->lat_head + 1) % LATBUF_CAP;
            s->lat_len--;
        }
    } else {
        s->lat_target = s->lat_cmd;
    }

    double edge = C_(lane_edge);
    if (s->turn_landed) {
        s->turn_landed = 0;
    } else {
        double tau = C_(lat_tau); if (tau < 1e-3) tau = 1e-3;
        double alpha = 1.0 - exp(-dt / tau);
        s->lane += (s->lat_target - s->lane) * alpha;
        if (s->lane < -edge) s->lane = -edge;
        if (s->lane >  edge) s->lane =  edge;
    }
    s->stumble_kill_t += dt;
    if (s->stumble_t > 0.0) { s->stumble_t -= dt; if (s->stumble_t < 0.0) s->stumble_t = 0.0; }

    if (fabs(s->lane) > edge) {
        if (cell_at(s, s->cell_index)->level >= 3) { do_kill(s, 1); return; }
        do_stumble(s);
        if (s->lane < -edge) s->lane = -edge;
        if (s->lane >  edge) s->lane =  edge;
    }
    if (cell_at(s, s->cell_index)->level >= 3 && fabs(s->lane) >= C_(cliff_edge_lane)) {
        s->cliff_edge_frames++;
        if (s->cliff_edge_frames >= Ci(cliff_edge_kill_frames)) { do_kill(s, 1); return; }
    } else {
        s->cliff_edge_frames = 0;
    }
    if (s->post_turn_linger > 0) s->post_turn_linger--;

    double dslope = C_(dist_per_velsec_slope);
    double dpv = (dslope > 0.0) ? (dslope * s->velocity) : C_(dist_per_velsec);
    s->distance_real += s->velocity * dt * dpv;
    double cell_len = s->e_cell_length;

    double curve_v;
    if (VC_N > 0) curve_v = vel_curve(s->distance_real);
    else {
        curve_v = s->velocity + C_(accel_per_sec) * dt;
        if (curve_v > C_(max_velocity)) curve_v = C_(max_velocity);
    }
    if (s->has_stumble_vel) {
        s->stumble_vel += C_(stumble_recover_rate) * dt;
        if (s->stumble_vel > curve_v) s->stumble_vel = curve_v;
        s->velocity = s->stumble_vel;
        if (s->stumble_vel >= curve_v) s->has_stumble_vel = 0;
    } else {
        s->velocity = curve_v;
    }
    s->velocity *= s->e_vel_scale;

    double moved = s->velocity * dt;
    s->distance += moved;
    s->score += moved * C_(score_per_distance);
    s->dist_into_cell += moved;

    int guard = 0;
    while (s->dist_into_cell >= cell_len && !s->is_dead) {
        if (++guard > 64) break;
        s->dist_into_cell -= cell_len;
        cross_into_next_cell(s);
    }

    /* AIRBORNE INTO A SLIDE HAZARD. The sim killed 100% of the time; the DEVICE kills 24%
       (23 of 94 events over 45 captures). That gap matters more than its size suggests:
       these takeoffs are 100% FORCED (a must_jump demanded them, measured in both sim and
       device, 0 discretionary), and in sim NO takeoff timing survives -- 15/15 fatal across
       cp 0.11..0.76. So the world presents a situation with no survivable action, punishes the
       one correct jump, and teaches nothing; it accounts for ~33% of the champion's deaths.
       Probabilistic like hole_trap_kill_prob, because a sharp threshold was the v55 mistake.
       p >= 1.0 short-circuits BEFORE the draw, so the default stream is untouched. */
    if (!s->is_dead && s->jump_t > 0.0) {
        int oc = cell_at(s, s->cell_index)->obst;
        if (must_slide_o(oc)) {
            double pk = s->e_airborne_slide_kill;
            if (pk >= 1.0) {
                do_kill(s, oc == OBST_FLAME_TOWER ? 5 : 3);
            } else if (s->slide_air_rolled != s->cell_index) {
                s->slide_air_rolled = s->cell_index;      /* ONE roll per cell: this runs EVERY
                       tick, and re-rolling turned the 24% device rate into ~75% (1-0.76^n). */
                if (rnd(s) < pk) do_kill(s, oc == OBST_FLAME_TOWER ? 5 : 3);
            }
        }
    }
    if (s->gap_grace_cell >= 0 && !s->is_dead) {
        if (s->cell_index != s->gap_grace_cell || s->jump_t > 0.0) {
            s->gap_grace_cell = -1;
        } else {
            double cl = cell_len > 1e-6 ? cell_len : 1e-6;
            if ((s->dist_into_cell / cl) >= C_(gap_entry_grace)) {
                Cell* c = cell_at(s, s->cell_index);
                do_kill(s, c->obst == OBST_HOLE ? 1 : 2);
                s->gap_grace_cell = -1;
            }
        }
    }
}

/* ============================================================== SNAPSHOT == */
/* raw_state field order (RAW_*) -- must match cpp_vec_env.RAW_FIELDS */
enum {
    R_is_dead = 0, R_is_jumping, R_is_sliding, R_is_on_ground, R_is_stumbling,
    R_is_strafing, R_must_jump, R_must_slide, R_is_game_over,
    R_can_turn_left, R_can_turn_right, R_is_turn,
    R_next_is_turn, R_next_can_turn_left, R_next_can_turn_right,
    R_next_must_jump, R_next_must_slide, R_next_wide_gap,
    R_next_ground_change_higher, R_next2_must_jump, R_next2_must_slide,
    R_next3_must_jump, R_next3_must_slide,
    R_score, R_velocity, R_path_type, R_path_level, R_facing, R_distance,
    R_next_path_level, R_next2_path_level,
    R_narrow_target_lane, R_next2_narrow_lane,
    R_cell_progress, R_lane, R_stumble_kill_timer, R_stumble_count,
    /* SOFT must-jump = step-DOWN to low ground. Reward-only, appended AFTER every observed
       field so the obs layout is untouched. Mirror of the Python ref's snapshot. */
    R_soft_stepdown, R_next_soft_stepdown,
    /* WAVE 17 horizon scalars: cells to the next must-JUMP / must-SLIDE, capped at
       HORIZON_CELLS. The +3 booleans stop one jump short at speed, so a 3-obstacle chain's
       third element is invisible when the first jump must be committed. */
    R_dist_next_must_jump, R_dist_next_must_slide,
    /* OBSTACLE TYPE of the current cell. The device stream carries it and every death forensic
       types hazards by it; without it a C++ rollout could only fall back on the WINDOW-PRESENTED
       must_jump/must_slide flags, which changes how `structures()` collapses adjacent cells and
       silently produced a different chain segmentation than the device (a 20x disagreement in
       tight-chain lethality that was pure basis mismatch, not a real divergence). */
    R_obstacle
};

/* Airtime for a jump starting NOW, scaled by the ground-height change its arc will cross.
   DEVICE-MEASURED (see the Python ref's jump_dur_stepdown_scale): landing LOWER = falling further
   = longer airtime = a 3-cell arc instead of 2; landing RAISED cuts the fall short (~0.74x). The
   landing level is read at the NOMINAL (unscaled) span. Both scales 1.0 -> the old constant. */
static double jump_airtime(State* s) {
    double base = s->e_jump_duration;
    double ds = C_(jump_dur_stepdown_scale), us = C_(jump_dur_stepup_scale);
    if (ds == 1.0 && us == 1.0) return base;
    double cl = s->e_cell_length; if (cl < 1e-6) cl = 1e-6;
    double span = base * s->velocity / cl;
    int off = (int)(span + 0.5); if (off < 1) off = 1;
    int lvl0 = cell_at(s, s->cell_index)->level;
    int lvl1 = cell_at(s, s->cell_index + off)->level;
    if (lvl1 < lvl0) return base * ds;
    if (lvl1 > lvl0) return base * us;
    return base;
}

static int blocked_cell(const Cell* c) {
    if (Cb(turn_aware_lookahead))
        return (c->flags & F_TURN) && (c->flags & F_CANL) && (c->flags & F_CANR);
    return (c->flags & F_TURN) != 0;
}

static void snapshot(State* s, double* o) {
    extend_track(s);
    Cell* cell = cell_at(s, s->cell_index);
    Cell* nxt  = cell_at(s, s->cell_index + 1);
    Cell* nxt2 = cell_at(s, s->cell_index + 2);
    Cell* nxt3 = cell_at(s, s->cell_index + 3);

    int obst = cell->obst;
    int mj = must_jump_o(obst), ms = must_slide_o(obst);

    /* _turn_window_open(): cp is NOT clamped here (matches the reference) */
    double cl = s->e_cell_length; if (cl < 1e-6) cl = 1e-6;
    double cp_raw = s->dist_into_cell / cl;
    int win = (cp_raw >= C_(turn_exec_early) && cp_raw <= C_(turn_exec_late));

    int is_turn = ((cell->flags & F_TURN) != 0) && win;
    int can_l   = ((cell->flags & F_CANL) != 0) && win;
    int can_r   = ((cell->flags & F_CANR) != 0) && win;
    if (s->post_turn_linger > 0) {
        is_turn = 1; can_l = s->linger_can_l; can_r = s->linger_can_r;
    }

    int blk0 = blocked_cell(cell), blk1 = blocked_cell(nxt), blk2 = blocked_cell(nxt2);
    int nobst = nxt->obst;

    o[R_is_dead]       = s->is_dead ? 1 : 0;
    o[R_is_jumping]    = s->jump_t > 0.0;
    o[R_is_sliding]    = s->slide_t > 0.0;
    o[R_is_on_ground]  = (s->jump_t <= 0.0 && !s->is_dead) ? 1 : 0;
    o[R_is_stumbling]  = s->stumble_t > 0.0;
    o[R_is_strafing]   = (fabs(s->lat_target - s->lane) > 0.5) ? 1 : 0;
    o[R_must_jump]     = mj;
    o[R_must_slide]    = ms;
    o[R_is_game_over]  = s->is_dead ? 1 : 0;
    o[R_can_turn_left] = can_l ? 1 : 0;
    o[R_can_turn_right]= can_r ? 1 : 0;
    o[R_is_turn]       = is_turn ? 1 : 0;
    o[R_next_is_turn]  = (nxt->flags & F_TURN) ? 1 : 0;
    o[R_next_can_turn_left]  = (nxt->flags & F_CANL) ? 1 : 0;
    o[R_next_can_turn_right] = (nxt->flags & F_CANR) ? 1 : 0;
    o[R_next_must_jump]  = must_jump_o(nobst);
    o[R_next_must_slide] = must_slide_o(nobst);
    o[R_next_wide_gap]   = (nxt->flags & F_WIDE) ? 1 : 0;
    {   /* scan forward to the next hazard of each kind; STOP at a true T-junction (both
           directions available) because the continuation is undecided there -- mirrors the
           hook's turn-aware walk refusing to resolve it, and the Python ref's _dist_next_hazard. */
        int dj = HORIZON_CELLS, ds = HORIZON_CELLS, d;
        for (d = 0; d <= HORIZON_CELLS; ++d) {
            Cell* q = cell_at(s, s->cell_index + d);
            if (d > 0 && (q->flags & F_CANL) && (q->flags & F_CANR)) break;
            if (dj == HORIZON_CELLS && must_jump_o(q->obst))  dj = d;
            if (ds == HORIZON_CELLS && must_slide_o(q->obst)) ds = d;
            if (dj < HORIZON_CELLS && ds < HORIZON_CELLS) break;
        }
        o[R_dist_next_must_jump]  = dj;
        o[R_dist_next_must_slide] = ds;
    }
    o[R_obstacle]           = (double)cell->obst;
#if !HORIZON_OBS
    o[R_dist_next_must_jump] = 0.0; o[R_dist_next_must_slide] = 0.0;  /* never left uninit */
#endif
    o[R_soft_stepdown]      = (cell->flags & F_SDLOW) ? 1 : 0;
    o[R_next_soft_stepdown] = (nxt->flags & F_SDLOW) ? 1 : 0;
    o[R_next_ground_change_higher] = (nobst == OBST_LEVEL_CHANGE) ? 1 : 0;
    o[R_next2_must_jump]  = (blk0 || blk1) ? 0 : must_jump_o(nxt2->obst);
    o[R_next2_must_slide] = (blk0 || blk1) ? 0 : must_slide_o(nxt2->obst);
    o[R_next3_must_jump]  = (blk0 || blk1 || blk2) ? 0 : must_jump_o(nxt3->obst);
    o[R_next3_must_slide] = (blk0 || blk1 || blk2) ? 0 : must_slide_o(nxt3->obst);
    o[R_score]      = floor(s->score);           /* int(self._score), score >= 0 */
    o[R_velocity]   = s->velocity;
    o[R_path_type]  = cell->type;
    o[R_path_level] = cell->level;
    o[R_facing]     = s->facing;
    o[R_distance]   = s->distance_real;
    o[R_next_path_level]  = blk0 ? -1 : (double)nxt->level;
    o[R_next2_path_level] = (blk0 || blk1) ? -1 : (double)nxt2->level;
    o[R_narrow_target_lane] = (nxt->flags & F_NARROW) ? (double)nxt->nlane : 0.0;
    o[R_next2_narrow_lane]  = (blk0 || blk1) ? 0.0
                            : ((nxt2->flags & F_NARROW) ? (double)nxt2->nlane : 0.0);
    o[R_cell_progress] = cell_progress(s);
    o[R_lane] = s->lane;
    o[R_stumble_kill_timer] = s->stumble_kill_t;
    o[R_stumble_count] = s->stumble_count;
}

/* ================================================================== OBS === */
static inline float clip01f(double x) { return (float)(x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x)); }

static double norm_enum(double v, double maxv) {
    if (v < 0.0) return 0.0;
    double c = v; if (c < 0.0) c = 0.0; if (c > maxv) c = maxv;
    return c / maxv;
}
static double lane_to_obs(double lane) {
    double x = (lane - OBS_LANE_MIN) / (OBS_LANE_MAX - OBS_LANE_MIN);
    if (x < 0.0) x = 0.0; if (x > 1.0) x = 1.0;
    return x;
}
static double narrow_lane_to_obs(double ntl) {
    if (ntl == 0.0) return lane_to_obs(0.0);
    return lane_to_obs(ntl * OBS_NARROW_CENTER);
}
static double path_type_obs(double ptd) {
    int pt = (int)ptd;
    if (pt < 0) return 0.0;
    if (pt <= 2) return 0.16;
    if (pt <= 7) return 0.33;
    if (pt <= 16) return 0.50;
    if (pt <= 30) return 0.66;
    return 0.83;
}
static double PJ_LEAD = 0.5;   /* obs.PJ_LEAD_FRAC, pushed from Python at install() */
TRS_EXPORT void sim_set_pj_lead(double f) { PJ_LEAD = f; }
static double predicted_jump_obs(double cp, double v, int crev_ahead) {
    if (!crev_ahead) return PJ_NONE;
    double span = PJ_JUMP_DURATION * v / PJ_CELL_LENGTH;
    if (EP_pj_vel_inv) {
        double sp = span > 1e-6 ? span : 1e-6;
        double x = (LEDGE_CELLS_AHEAD - cp) / sp / 2.0;
        if (x < 0.0) x = 0.0; if (x > 1.0) x = 1.0;
        return x;
    }
    double raw = (LEDGE_CELLS_AHEAD - cp) - PJ_LEAD * span;
    double x = (raw + 0.5) / 2.0;
    if (x < 0.0) x = 0.0; if (x > 1.0) x = 1.0;
    return x;
}
static double predicted_turn_obs(double cp, double v, int on_junction, int next_is_turn) {
    if (on_junction) {
        double margin = TURN_DEADLINE_CP - cp - TURN_SEND_FRAMES * v * DCPROG_PER_VEL;
        double x = margin / TURN_DEADLINE_CP;
        if (x < 0.0) x = 0.0; if (x > 1.0) x = 1.0;
        return x * 0.75;
    }
    if (next_is_turn) return 1.0;
    return 0.0;
}

/* obs.state_to_obs on a raw-state array */
static void base_obs(const double* r, float* out) {
    int i;
    for (i = 0; i < 23; ++i) out[i] = r[i] != 0.0 ? 1.0f : 0.0f;
    const int b = 23;
    double v = r[R_velocity];
    out[b + 0] = clip01f((v - OBS_VEL_MIN) / (OBS_VEL_MAX - OBS_VEL_MIN));
    out[b + 1] = (float)path_type_obs(r[R_path_type]);
    out[b + 2] = (float)norm_enum(r[R_path_level], OBS_PATH_LEVEL_MAX);
    out[b + 3] = (float)norm_enum(r[R_facing], OBS_FACING_MAX);
    double cp0 = r[R_cell_progress];
    out[b + 4] = clip01f(cp0);
    out[b + 5] = (float)lane_to_obs(r[R_lane]);
    out[b + 6] = (float)narrow_lane_to_obs(r[R_narrow_target_lane]);
    out[b + 7] = clip01f(cp0 + DCPROG_PER_VEL * v);
    out[b + 8] = (float)predicted_jump_obs(cp0, v, r[R_next_ground_change_higher] != 0.0);
    out[b + 9]  = (float)norm_enum(r[R_next_path_level], OBS_PATH_LEVEL_MAX);
    out[b + 10] = (float)norm_enum(r[R_next2_path_level], OBS_PATH_LEVEL_MAX);
    out[b + 11] = (float)narrow_lane_to_obs(r[R_next2_narrow_lane]);
    double span = LAND_SPAN_A * v + LAND_SPAN_B;
    if (span < 0.0) span = 0.0;
#if LAND_SPAN_LEVEL_SCALED
    {   /* level -1 = the hook's junction dropout -> keep the span flat rather than guess */
        int l0 = (int)r[R_path_level];
        int la = (int)r[R_next2_path_level];
        if (la <= 0) la = (int)r[R_next_path_level];
        if (la > 0 && l0 > 0) {
            if (la < l0)      span *= LAND_SPAN_STEPDOWN;
            else if (la > l0) span *= LAND_SPAN_STEPUP;
        }
    }
#endif
    int off = (int)(cp0 + span);
    double lh;
    if (off <= 0)      lh = r[R_must_jump]       != 0.0 ? 1.0 : 0.0;
    else if (off == 1) lh = r[R_next_must_jump]  != 0.0 ? 1.0 : 0.0;
    else if (off == 2) lh = r[R_next2_must_jump] != 0.0 ? 1.0 : 0.0;
    else if (off == 3) lh = r[R_next3_must_jump] != 0.0 ? 1.0 : 0.0;
    else               lh = 0.5;
    out[b + 13] = (float)lh;
    /* slide_landing_hazard: SAME arc, SAME rounding, must-SLIDE ladder. The airborne-into-slide
       trap has no recovery action -- device forensic (n=100 takeoffs spanning a slide hazard):
       survivors' arcs clear it by +1.54 cells, victims' end on it at +0.21, and 89% of the victims
       pressed SLIDE while still in the air. int(landing)==hazard cell IS that discriminator. */
    double slh;
    if (off <= 0)      slh = r[R_must_slide]       != 0.0 ? 1.0 : 0.0;
    else if (off == 1) slh = r[R_next_must_slide]  != 0.0 ? 1.0 : 0.0;
    else if (off == 2) slh = r[R_next2_must_slide] != 0.0 ? 1.0 : 0.0;
    else if (off == 3) slh = r[R_next3_must_slide] != 0.0 ? 1.0 : 0.0;
    else               slh = 0.5;
    out[b + 14] = (float)slh;
    /* landing_margin: signed cells from where the arc comes down to the NEAREST must-jump hazard.
       landing_hazard is BINARY, so +1.05 (one perturbation from fatal) and +1.65 (safe) both read
       0.0. DEVICE: survivors cluster at +1.65 (p25 +1.28, 2% within +-0.5 of the hazard); victims
       scatter (p25 +0.35, 38% within +-0.5) = 19x enrichment. Mirror of obs.py. */
    {
        double land = cp0 + span;
        double lad2[4];
        lad2[0] = r[R_must_jump];  lad2[1] = r[R_next_must_jump];
        lad2[2] = r[R_next2_must_jump]; lad2[3] = r[R_next3_must_jump];
        int best = -1; double bd = 1e9;
        int q;
        for (q = 0; q < 4; ++q) {
            if (lad2[q] == 0.0) continue;
            double dd = fabs(land - (double)q);
            if (dd < bd) { bd = dd; best = q; }
        }
        out[b + 15] = (best < 0) ? 1.0f : clip01f((land - (double)best + 1.0) / 4.0);
    }
    out[b + 12] = (float)predicted_turn_obs(cp0, v, r[R_is_turn] != 0.0,
                                            r[R_next_is_turn] != 0.0);
#if HORIZON_OBS
    out[b + 16] = (float)(r[R_dist_next_must_jump]  / (double)HORIZON_CELLS);
    out[b + 17] = (float)(r[R_dist_next_must_slide] / (double)HORIZON_CELLS);
#endif
}

/* push the frame + emit the stacked obs (frame_stack + one-hot action history) */
static void push_frame(State* s, const double* raw) {
    int fs = EP_FS < 1 ? 1 : EP_FS;
    if (s->n_frames < fs) {
        base_obs(raw, s->frames[s->n_frames]);
        s->n_frames++;
    } else {
        int i;
        for (i = 1; i < fs; ++i) memcpy(s->frames[i - 1], s->frames[i], sizeof(float) * OBS_BASE);
        base_obs(raw, s->frames[fs - 1]);
    }
}

static void full_obs(State* s, float* out) {
    int fs = EP_FS < 1 ? 1 : EP_FS;
    int i, j;
    int n = s->n_frames;
    if (n <= 0) {                          /* shouldn't happen; be safe */
        for (i = 0; i < fs * OBS_BASE; ++i) out[i] = 0.0f;
    } else {
        int pad = fs - n;                  /* left-pad by repeating the OLDEST frame */
        for (i = 0; i < pad; ++i)
            memcpy(out + (size_t)i * OBS_BASE, s->frames[0], sizeof(float) * OBS_BASE);
        for (i = 0; i < n; ++i)
            memcpy(out + (size_t)(pad + i) * OBS_BASE, s->frames[i], sizeof(float) * OBS_BASE);
    }
    float* h = out + (size_t)fs * OBS_BASE;
    for (i = 0; i < EP_K * N_ACTIONS; ++i) h[i] = 0.0f;
    if (EP_K > 0) {
        int have = s->act_hist_n < EP_K ? s->act_hist_n : EP_K;
        int pad = EP_K - have;
        for (i = 0; i < pad; ++i) h[i * N_ACTIONS + A_NOOP] = 1.0f;
        for (j = 0; j < have; ++j) {
            int a = s->act_hist[(s->act_hist_n - have + j) % MAX_K];
            if (a >= 0 && a < N_ACTIONS) h[(pad + j) * N_ACTIONS + a] = 1.0f;
        }
    }
}

/* ============================================================== SHAPING === */
static double shaping_reward(State* s, int action, const double* p, int has_prev,
                             double prev_velocity, double velocity) {
    /* Fast path: nothing to shape. This used to be a HAND-MAINTAINED list of knobs, and it was
       never updated when stepdown_bonus_gate, landing_hazard_penalty or turn_takeoff_penalty were
       added -- so any of those, set ALONE, returned 0.0 here and the knob silently did nothing.
       Recipes that also set a listed knob (v111 sets narrow_hold_bonus / airborne_action_penalty /
       strafe_penalty / turn_buffer_bonus) were unaffected, which is why it survived: it only bites
       ISOLATED ablations -- exactly the experiment you run to attribute a result to one knob.
       Scanning the array cannot go stale. */
    {
        int _k, _any = 0;
        for (_k = 0; _k < SHP_COUNT; ++_k) if (SHP[_k] != 0.0) { _any = 1; break; }
        if (!_any) return 0.0;
    }
    double r = 0.0;
    double lane = has_prev ? p[R_lane] : 0.0;

    if (SHP[SHP_slide_jump_penalty] && action == A_JUMP && has_prev) {
        if (p[R_must_slide] != 0.0 || p[R_next_must_slide] != 0.0 || p[R_is_sliding] != 0.0)
            r -= SHP[SHP_slide_jump_penalty];
    }
    if (SHP[SHP_velocity_drop_penalty] && (prev_velocity - velocity) > 3.0)
        r -= SHP[SHP_velocity_drop_penalty] * (prev_velocity - velocity);

    int is_strafe = (action == A_STRAFE_L || action == A_STRAFE_R);
    int on_narrow = 0;
    if (has_prev) {
        int pt = (int)p[R_path_type];
        on_narrow = ((pt >= 17 && pt <= 30) || p[R_narrow_target_lane] != 0.0);
    }
    if (SHP[SHP_action_cost] && action != A_NOOP) {
        if (!(is_strafe && on_narrow)) r -= SHP[SHP_action_cost];
    }
    if (SHP[SHP_narrow_hold_bonus] && on_narrow) {
        double ntl = p[R_narrow_target_lane];
        if (ntl != 0.0) {
            double sign = ntl > 0 ? 1.0 : -1.0;
            double closeness = (lane * sign) / 3.0;
            if (closeness < 0.0) closeness = 0.0;
            if (closeness > 1.0) closeness = 1.0;
            r += SHP[SHP_narrow_hold_bonus] * closeness;
        } else {
            r += SHP[SHP_narrow_hold_bonus];
        }
    }
    if (SHP[SHP_off_center_penalty] && !on_narrow) {
        double off = fabs(lane) - 1.5;
        if (off > 0.0) r -= SHP[SHP_off_center_penalty] * off;
    }
    /* LANDING-HAZARD PENALTY: dock a JUMP whose arc is predicted to land on a must-jump hazard.
       Amplifier for a correct-but-quiet cue (v107 counterfactual dP only -0.0039 on a 0.036 base
       while 5 of 7 of its device deaths are dt=4 Ledge). Mirror of reward.py. */
    if (SHP[SHP_landing_hazard_penalty] != 0.0 && has_prev && action == A_JUMP) {
        double sp = LAND_SPAN_A * p[R_velocity] + LAND_SPAN_B;
        if (sp < 0.0) sp = 0.0;
        int off = (int)(p[R_cell_progress] + sp);
        double lad[4];
        lad[0] = p[R_must_jump];  lad[1] = p[R_next_must_jump];
        lad[2] = p[R_next2_must_jump]; lad[3] = p[R_next3_must_jump];
        if (off >= 0 && off <= 3 && lad[off] != 0.0) r -= SHP[SHP_landing_hazard_penalty];
    }
    /* SLIDE-LANDING-HAZARD PENALTY: dock a JUMP whose arc lands on a must-SLIDE hazard. The cue
       (obs `slide_landing_hazard`) IS learned on its own -- dP(JUMP) -0.00450 @2M -> -0.00631 @10M
       in v117 -- but it moved no outcome, which is the exact signature the held-jump and
       landing_hazard fixes both showed: a correct-but-quiet cue needs an explicit PRICE. Device
       evidence for the counterfactual: over 26,854 takeoffs the cue carries a 52.6x death-rate lift
       (15.69% vs 0.30%) and survivors take off at cp 0.53 vs victims' 0.62, so an EARLIER takeoff
       is available and is what separates living from dying. Mirror of reward.py. */
    /* STACK-MARGIN PENALTY [USER 2026-07-30: "the reward lacks an adaptability when obstacles are
       stacked ... it gave it the same amount of points for just jumping over an obstacle whether
       there is an obstacle behind or not"]. EXACTLY RIGHT and it is visible in reward.py:
       reaction_bonus pays a flat +0.1 for a JUMP whenever must_jump/next_must_jump fires, IDENTICAL
       for an isolated obstacle and for one with another right behind it. So the policy is fully paid
       for reusing ONE learnt trajectory in a situation that needs a different one, and the only
       thing that ever says otherwise is dying (0.03% of frames).
       This charges ONLY the stacked case: when the arc clears hazard A but comes down within
       `stack_margin_cells` of the NEXT hazard B, dock in proportion to how tight it is. Isolated
       obstacles are untouched, so there is no gradient pushing takeoff later in general -- which is
       why the symmetric margin_bonus (v134) backfired: tight takeoffs rose 13.6% -> 19.4%.
       Device evidence: survivors land 1.65 cells clear (p25 1.28), victims 0.35 (p25). */
    /* ⚠️ NOT while a step-UP crevasse is in play. v135 showed the stack charge buys the best device
       median (6720 vs v120's 4736) but costs the mid->high transition (graze 10.6% vs 1.6-2.2%):
       the penalty perturbs takeoff timing generally, and crevasse timing is the collateral. The
       crevasse case is already priced by its own apex/graze model, so exclude it here and keep the
       stack charge for hole/tree stacks only. next_ground_change_higher is the streamed crevasse-
       ahead flag the obs already uses. */
    if (SHP[SHP_stack_margin_penalty] != 0.0 && has_prev && action == A_JUMP
            && p[R_next_ground_change_higher] == 0.0) {
        double sp = LAND_SPAN_A * p[R_velocity] + LAND_SPAN_B;
        if (sp < 0.0) sp = 0.0;
        double land = p[R_cell_progress] + sp;
        double lad4[4];
        lad4[0] = p[R_must_jump];  lad4[1] = p[R_next_must_jump];
        lad4[2] = p[R_next2_must_jump]; lad4[3] = p[R_next3_must_jump];
        /* the STACK element: the first hazard at or beyond where the arc comes down */
        int k4, nxt_hz = -1;
        for (k4 = 0; k4 < 4; ++k4) {
            if (lad4[k4] != 0.0 && (double)k4 >= land - 1e-9) { nxt_hz = k4; break; }
        }
        if (nxt_hz >= 0) {
            double gap = (double)nxt_hz - land;          /* room between landing and the next hazard */
            double lim = C_(stack_margin_cells);
            if (lim <= 0.0) lim = 1.0;
            if (gap < lim) {
                double tight = (lim - gap) / lim;        /* 0 at the limit, 1 when landing ON it */
                if (tight > 1.0) tight = 1.0;
                r -= SHP[SHP_stack_margin_penalty] * tight;
            }
        }
    }
    /* MARGIN BONUS [2026-07-30] -- THE MISSING DENSE SIGNAL.
       Measured chain: (1) 93% of deaths had the fatal hazard VISIBLE while the runner was still
       grounded and able to act; (2) fatal states are 17x SEPARATED from safe ones in obs space, so
       it is not aliasing; (3) the survivable window is 0.72 cells = 2-3 control steps, so it is not
       control resolution; (4) the dangerous configuration is 19.6% of training frames, so it is not
       exposure. What is left: distance reward pays the SAME for landing 1.65 cells clear of a hazard
       and 0.35 clear, so the policy only learns the difference on the 0.03% of frames where it
       actually dies. With no gradient toward margin it drifts to the boundary -- device survivors
       sit at margin +1.65 (p25 +1.28) and victims at +0.35 (p25), and the champion takes off with
       a median lead of ONE frame against the human's FOUR.
       This grades every hazard-spanning takeoff by how much room it leaves, turning a 0.03% sparse
       death signal into a dense one on ~20% of frames. Discretionary: takeoff TIMING is free. */
    if (SHP[SHP_margin_bonus] != 0.0 && has_prev && action == A_JUMP) {
        double sp = LAND_SPAN_A * p[R_velocity] + LAND_SPAN_B;
        if (sp < 0.0) sp = 0.0;
        double land = p[R_cell_progress] + sp;
        double lad3[4];
        lad3[0] = p[R_must_jump];  lad3[1] = p[R_next_must_jump];
        lad3[2] = p[R_next2_must_jump]; lad3[3] = p[R_next3_must_jump];
        int b3 = -1; double bd3 = 1e9; int q3;
        for (q3 = 0; q3 < 4; ++q3) {
            if (lad3[q3] == 0.0) continue;
            double dd = fabs(land - (double)q3);
            if (dd < bd3) { bd3 = dd; b3 = q3; }
        }
        if (b3 >= 0) {
            double m = land - (double)b3;              /* cells past the hazard */
            double t = (m - 1.0) / 1.0;                /* 1.0 cell = neutral, +1.65 device-safe */
            if (t < -1.0) t = -1.0; if (t > 1.0) t = 1.0;
            r += SHP[SHP_margin_bonus] * t;            /* graded: tight -> negative, clear -> positive */
        }
    }
    /* STEPDOWN-JUMP PENALTY [USER 2026-07-30: "penalize the policy from random jumping the mid to
       low transition. It is useless and it can be a source of death."]. A step-DOWN to lower ground
       is FOOT-CROSSABLE (stepdown_grounded_ok) -- the human walks 58% of them and never dies -- but
       every policy jumps ~100% (v120 95/95, v126 103/103, v111 25/26 on device vs the human's
       17/40). The jump is not merely wasted: it launches a 2.5-3 cell arc that can terminate on a
       hazard, which is the project's dominant death mode. DISCRETIONARY by construction (walking is
       survivable in-sim), which is the check that sank slide_landing_hazard_penalty. Only fires when
       NO hard must-jump is present, so a real hazard is never taxed. */
    if (SHP[SHP_stepdown_jump_penalty] != 0.0 && has_prev && action == A_JUMP) {
        int soft = (p[R_soft_stepdown] != 0.0) || (p[R_next_soft_stepdown] != 0.0);
        int hard = (p[R_must_jump] != 0.0 && p[R_soft_stepdown] == 0.0)
                || (p[R_next_must_jump] != 0.0 && p[R_next_soft_stepdown] == 0.0);
        if (soft && !hard) r -= SHP[SHP_stepdown_jump_penalty];
    }
    if (SHP[SHP_slide_landing_hazard_penalty] != 0.0 && has_prev && action == A_JUMP) {
        double sp = LAND_SPAN_A * p[R_velocity] + LAND_SPAN_B;
        if (sp < 0.0) sp = 0.0;
        int off = (int)(p[R_cell_progress] + sp);
        double lad[4];
        lad[0] = p[R_must_slide];  lad[1] = p[R_next_must_slide];
        lad[2] = p[R_next2_must_slide]; lad[3] = p[R_next3_must_slide];
        if (off >= 0 && off <= 3 && lad[off] != 0.0) r -= SHP[SHP_slide_landing_hazard_penalty];
    }
    /* TURN-TAKEOFF PENALTY: dock a JUMP still airborne when an unresolved junction arrives.
       MEASURED: 5.0% of takeoffs have a junction within 2 cells and 8.7% of THOSE kill, vs ~1%
       for takeoffs generally. Structural, not a training gap: the lookahead is computed along the
       CURRENT heading with a junction dropout, so a turn firing mid-air drops the runner onto a
       path the obs never modelled. Pricing the takeoff is the learnable half. Mirror of
       reward.py -- the span arithmetic and the >= 1.0 reach test must match exactly. */
    if (SHP[SHP_turn_takeoff_penalty] != 0.0 && has_prev && action == A_JUMP) {
        double sp = LAND_SPAN_A * p[R_velocity] + LAND_SPAN_B;
        if (sp < 0.0) sp = 0.0;
        double reach = p[R_cell_progress] + sp;
        if (p[R_is_turn] != 0.0 || (reach >= 1.0 && p[R_next_is_turn] != 0.0))
            r -= SHP[SHP_turn_takeoff_penalty];
    }
    if (SHP[SHP_reaction_bonus] && has_prev) {
        if ((p[R_must_jump] != 0.0 || p[R_next_must_jump] != 0.0) && action == A_JUMP) {
            /* a jump credited SOLELY by a soft step-down earns nothing (mirror of reward.py) */
            int hard = (p[R_must_jump] != 0.0 && p[R_soft_stepdown] == 0.0)
                    || (p[R_next_must_jump] != 0.0 && p[R_next_soft_stepdown] == 0.0);
            if (hard || SHP[SHP_stepdown_bonus_gate] == 0.0)
                r += SHP[SHP_reaction_bonus];
        } else if ((p[R_must_slide] != 0.0 || p[R_next_must_slide] != 0.0) && action == A_SLIDE) {
            r += SHP[SHP_reaction_bonus];
        } else if (p[R_is_turn] != 0.0) {
            /* CPU quirk: `prev.get("cell_progress") or 0.44` maps cp == 0 -> 0.44 */
            double cpv = p[R_cell_progress];
            if (cpv == 0.0) cpv = 0.44;
            double cp = cpv + DCPROG_PER_VEL * prev_velocity;
            double aim = 1.0 - SHP[SHP_aim_center_weight] * fabs(cp - 0.44) / 0.16;
            if (aim < 0.0) aim = 0.0;
            int side_ok = ((action == A_TURN_L && p[R_can_turn_left] != 0.0)
                        || (action == A_TURN_R && p[R_can_turn_right] != 0.0));
            if (side_ok) r += SHP[SHP_reaction_bonus] * aim;
        }
    }
    if (SHP[SHP_turn_buffer_bonus] && has_prev) {
        if ((action == A_TURN_L || action == A_TURN_R)
                && p[R_is_turn] == 0.0 && p[R_next_is_turn] != 0.0) {
            int ok = (action == A_TURN_L) ? (p[R_next_can_turn_left] != 0.0)
                                          : (p[R_next_can_turn_right] != 0.0);
            if (ok) r += SHP[SHP_turn_buffer_bonus];
        }
    }
    if (SHP[SHP_turn_spam_penalty] && s->has_last_action) {
        int is_t = (action == A_TURN_L || action == A_TURN_R);
        int last_t = (s->last_action == A_TURN_L || s->last_action == A_TURN_R);
        if (is_t && last_t) r -= SHP[SHP_turn_spam_penalty];
    }
    if (SHP[SHP_idle_action_penalty] && has_prev) {
        if (action == A_JUMP && p[R_must_jump] == 0.0 && p[R_next_must_jump] == 0.0) {
            r -= SHP[SHP_idle_action_penalty];
        } else if (action == A_SLIDE && p[R_must_slide] == 0.0 && p[R_next_must_slide] == 0.0) {
            r -= SHP[SHP_idle_action_penalty];
        } else if ((action == A_TURN_L || action == A_TURN_R)
                   && p[R_is_turn] == 0.0 && p[R_next_is_turn] == 0.0) {
            r -= SHP[SHP_idle_action_penalty];
        }
    }
    if (SHP[SHP_redundant_action_penalty] && has_prev) {
        if (action == A_JUMP && p[R_must_jump] != 0.0 && p[R_next_must_jump] == 0.0)
            r -= SHP[SHP_redundant_action_penalty];
        else if (action == A_SLIDE && p[R_must_slide] != 0.0 && p[R_next_must_slide] == 0.0)
            r -= SHP[SHP_redundant_action_penalty];
    }
    if (SHP[SHP_airborne_action_penalty] && has_prev) {
        if (action == A_JUMP && p[R_is_jumping] != 0.0)
            r -= SHP[SHP_airborne_action_penalty];
        else if (action == A_SLIDE && p[R_is_sliding] != 0.0)
            r -= SHP[SHP_airborne_action_penalty];
    }
    if (SHP[SHP_jump_hold_penalty] && has_prev) {
        int win = Ci(jump_hold_window);
        if (action == A_JUMP && s->jump_recency < win && p[R_must_jump] == 0.0)
            r -= SHP[SHP_jump_hold_penalty];
        else if (action == A_SLIDE && s->slide_recency < win && p[R_must_slide] == 0.0)
            r -= SHP[SHP_jump_hold_penalty];
    }
    double dock = SHP[SHP_strafe_penalty] ? SHP[SHP_strafe_penalty] : SHP[SHP_idle_action_penalty];
    if (dock && is_strafe && (!has_prev || p[R_narrow_target_lane] == 0.0))
        r -= dock;
    return r;
}

/* ================================================================ RESET === */
static void apply_profiles(State* s) {
    s->profile_idx = -1;
    if (PROF_N <= 0) return;
    double u = rnd(s);
    int idx = PROF_N - 1, i;
    for (i = 0; i < PROF_N; ++i) {
        if (PROF_CDF[i] > u) { idx = i; break; }
    }
    s->profile_idx = idx;
    Profile* pr = &PROF[idx];
    for (i = 0; i < pr->n; ++i) {
        double v = pr->val[i];
        if (PROFILE_IS_INT[pr->key[i]]) v = (double)(int)v;
        switch (pr->key[i]) {
        case P_obstacle_prob:                s->e_obstacle_prob = v; break;
        case P_turn_prob:                    s->e_turn_prob = v; break;
        case P_turn_prob_late:               s->e_turn_prob_late = v; break;
        case P_min_space_between_turns:      s->e_min_turns = v; break;
        case P_min_space_between_turns_late: s->e_min_turns_late = v; break;
        case P_narrows_after_distance:       s->e_narrows_after = v; break;
        case P_zone_after_distance:          s->e_zone_after = v; break;
        case P_zone_up_prob:                 s->e_zone_up = v; break;
        case P_zone_change_every_cells:      s->e_zone_every = v; break;
        case P_min_space_between_obstacles:  s->e_min_obst = v; break;
        case P_compound_prob:                s->e_compound_prob = v; break;
        case P_wide_gap_prob:                s->e_wide_gap_prob = v; break;
        case P_jumpslide_prob:               s->e_jumpslide_prob = v; break;
        case P_jumpnarrow_prob:              s->e_jumpnarrow_prob = v; break;
        case P_jumpgap_prob:                 s->e_jumpgap_prob = v; break;
        case P_slidegap_prob:                s->e_slidegap_prob = v; break;
        case P_airborne_slide_kill_prob:     s->e_airborne_slide_kill = v; break;
        default: break;
        }
    }
}

/* _sample_domain(): 9 uniform draws in the reference's order (three of them
 * feed knobs nothing reads any more -- kept so the stream matches). */
static void sample_domain(State* s) {
    s->e_jump_duration  = C_(jump_duration);
    s->e_slide_duration = C_(slide_duration);
    s->e_cell_length    = C_(cell_length);
    s->e_obstacle_prob  = C_(obstacle_prob);
    s->e_turn_prob      = C_(turn_prob);
    s->e_turn_prob_late = C_(turn_prob_late);
    s->e_turn_window_frac = C_(turn_window_frac);
    s->e_turn_early_frac  = C_(turn_early_frac);
    s->e_turn_late_frac   = C_(turn_late_frac);
    s->e_vel_scale = 1.0;
    if (Cb(domain_rand)) {
        #define U(lo, hi) (C_(lo) + (C_(hi) - C_(lo)) * rnd(s))
        s->e_turn_window_frac = U(dr_turn_window_frac_lo, dr_turn_window_frac_hi);
        s->e_turn_early_frac  = U(dr_turn_early_frac_lo,  dr_turn_early_frac_hi);
        s->e_turn_late_frac   = U(dr_turn_late_frac_lo,   dr_turn_late_frac_hi);
        s->e_jump_duration    = U(dr_jump_duration_lo,    dr_jump_duration_hi);
        s->e_slide_duration   = U(dr_slide_duration_lo,   dr_slide_duration_hi);
        s->e_obstacle_prob    = U(dr_obstacle_prob_lo,    dr_obstacle_prob_hi);
        s->e_cell_length      = U(dr_cell_length_lo,      dr_cell_length_hi);
        {
            double tp = U(dr_turn_prob_lo, dr_turn_prob_hi);
            s->e_turn_prob = tp;
            s->e_turn_prob_late = tp > C_(turn_prob_late) ? tp : C_(turn_prob_late);
        }
        s->e_vel_scale = U(dr_velocity_scale_lo, dr_velocity_scale_hi);
        #undef U
    }
    /* generation knobs back to base, then episode_profiles on top */
    s->e_min_turns      = C_(min_space_between_turns);
    s->e_min_turns_late = C_(min_space_between_turns_late);
    s->e_narrows_after  = C_(narrows_after_distance);
    s->e_zone_after     = C_(zone_after_distance);
    s->e_zone_every     = C_(zone_change_every_cells);
    s->e_zone_up        = C_(zone_up_prob);
    s->e_min_obst       = C_(min_space_between_obstacles);
    s->e_compound_prob  = C_(compound_prob);
    s->e_wide_gap_prob  = C_(wide_gap_prob);
    s->e_jumpslide_prob  = C_(jumpslide_prob);
    s->e_jumpnarrow_prob = C_(jumpnarrow_prob);
    s->e_jumpgap_prob    = C_(jumpgap_prob);
    s->e_slidegap_prob   = C_(slidegap_prob);
    s->e_airborne_slide_kill = C_(airborne_slide_kill_prob);
    apply_profiles(s);
}

static void reset_env(State* s, long long seed) {
    uint64_t rs = s->rng_state, ri = s->rng_inc, dr = s->draws;
    int eps = s->episodes;
    memset(s, 0, sizeof(State));
    if (seed >= 0) {
        s->rng_state = 0u;
        s->rng_inc = (((uint64_t)seed) << 1u) | 1u;
        pcg32(s); s->rng_state += 0x853c49e6748fea9bULL; pcg32(s);
        s->draws = 0;
    } else {
        s->rng_state = rs; s->rng_inc = ri; s->draws = dr;
    }
    s->episodes = eps;

    sample_domain(s);

    s->velocity = C_(start_velocity);
    s->facing = 1;                        /* DIR_N */
    s->lane = 0.0; s->lat_cmd = 0.0; s->lat_target = 0.0;
    s->distance = 0.0;
    s->distance_real = C_(start_distance);
    double cmax = C_(curriculum_max_start_dist);
    if (cmax > 0.0 && INJ_N <= 0) {
        double frac = C_(curriculum_deep_start_frac);
        if (rnd(s) < frac) {
            double pw = C_(curriculum_dist_power); if (pw < 0.1) pw = 0.1;
            double u = pow(rnd(s), pw);
            s->distance_real = C_(start_distance) + u * cmax;
            if (VC_N > 0) s->velocity = vel_curve(s->distance_real);
        }
    }
    s->dist_into_cell = 0.0;
    s->cell_index = 0;
    s->is_dead = 0; s->death_type = 0;
    s->jump_recency = 99; s->slide_recency = 99;
    s->last_action = A_NOOP; s->has_last_action = 0;
    s->jump_t = 0.0; s->slide_t = 0.0; s->strafe_t = 0.0; s->stumble_t = 0.0;
    s->has_takeoff = 0; s->jump_takeoff_pos = 0.0;
    s->pending_jump_frames = 0;
    s->cliff_edge_frames = 0;
    s->turn_debounce_t = 0.0;
    s->frames_since_land = 99;
    s->slide_air_rolled = -1;
    s->jump_held_airborne = 0;
    s->stumble_kill_t = C_(stumble_kill_window);
    s->stumble_count = 0;
    s->has_stumble_vel = 0;
    s->turn_landed = 0;
    s->gap_grace_cell = -1;
    s->post_turn_linger = 0;
    s->pending_turn = 0; s->pending_turn_ttl = 0;
    s->score = C_(start_score);
    s->cells_since_obstacle = 0; s->cells_since_turn = 0;
    s->cells_since_bonus = 0;    s->cells_since_zone = 0;
    s->zone_level = Ci(default_path_level);
    s->b2b_obstacles = 0; s->pending_double = 0;
    s->pending_gap = 0; s->pending_gap_stepup = 0; s->pending_gap_sdlow = 0;
    s->pending_gap_obst = 0; s->pending_gap_type = 1; s->pending_gap_level = 2;
    s->pending_narrow = 0; s->pending_narrow_lane = 0; s->pending_narrow_type = 17;
    s->qseq_head = s->qseq_len = 0;
    s->qpg_head = s->qpg_len = 0;
    s->inj_cursor = 0;
    s->lat_head = s->lat_len = 0;
    s->elapsed = 0;
    s->ep_ret = 0.0; s->ep_len = 0; s->ep_dist = 0.0;
    s->n_frames = 0; s->act_hist_n = 0;
    s->last_hz_end_cell = -999; s->last_hz_type = 0; s->acted_since_hazard = 1;
    s->has_prev = 0;

    /* _build_initial_track(): cell 0 = plain straight, then fill */
    s->gen_head = 0;
    cell_make(cell_at(s, 0), Ci(path_type_straight), Ci(default_path_level), OBST_NONE);
    s->gen_head = 1;
    extend_track(s);

    double raw[RAW_CAP];
    snapshot(s, raw);
    memcpy(s->prev_raw, raw, sizeof(raw));
    s->has_prev = 1;
    s->prev_score = raw[R_score];
    s->prev_distance = raw[R_distance];
    push_frame(s, raw);
}

/* ================================================================== API === */
static void cfg_defaults(void);

TRS_EXPORT void sim_cfg_reset(void) { cfg_defaults(); }

TRS_EXPORT int sim_cfg_set(const char* name, double v) {
    int i;
    for (i = 0; i < K_COUNT; ++i)
        if (strcmp(CFG_NAMES[i], name) == 0) { G[i] = v; return 1; }
    return 0;
}
TRS_EXPORT double sim_cfg_get(const char* name) {
    int i;
    for (i = 0; i < K_COUNT; ++i)
        if (strcmp(CFG_NAMES[i], name) == 0) return G[i];
    return 0.0;
}
TRS_EXPORT void sim_cfg_set_vel_curve(const double* d, const double* v, int n) {
    if (n > MAX_VC) n = MAX_VC;
    if (n < 0) n = 0;
    VC_N = n;
    int i; for (i = 0; i < n; ++i) { VC_D[i] = d[i]; VC_V[i] = v[i]; }
}
TRS_EXPORT void sim_cfg_set_ramp_table(const double* rows, int n_rows) {
    if (n_rows > MAX_RAMP) n_rows = MAX_RAMP;
    if (n_rows < 0) n_rows = 0;
    RAMP_N = n_rows;
    int i, j; for (i = 0; i < n_rows; ++i) for (j = 0; j < 6; ++j) RAMP[i][j] = rows[i * 6 + j];
}
TRS_EXPORT void sim_cfg_set_obst_weights(const int* types, const double* w, int n) {
    if (n > MAX_OW) n = MAX_OW;
    if (n < 0) n = 0;
    OW_N = n;
    int i; for (i = 0; i < n; ++i) { OW_T[i] = types[i]; OW_W[i] = w[i]; }
}
/* type_reject: parallel arrays of (prev, cur, p_reject). ALWAYS clears the matrix first, so a
 * caller that installs an empty table restores "no rejection" instead of inheriting the previous
 * env's. Entries outside 0..7 are dropped here; cpp_vec_env.py has already rejected them loudly
 * (and rejected any entry whose CUR is a tree, which the sampler could never consult). */
TRS_EXPORT void sim_cfg_set_type_reject(const int* prev, const int* cur, const double* p, int n) {
    int a, b, i;
    for (a = 0; a < TR_DIM; ++a) for (b = 0; b < TR_DIM; ++b) TR[a][b] = 0.0;
    for (i = 0; i < n; ++i) {
        a = prev[i]; b = cur[i];
        if (a < 0 || a >= TR_DIM || b < 0 || b >= TR_DIM) continue;
        TR[a][b] = p[i];
    }
}
TRS_EXPORT void sim_cfg_set_narrow_len_weights(const int* lens, const double* w, int n) {
    if (n > MAX_NLW) n = MAX_NLW;
    if (n < 1) n = 1;
    NLW_N = n;
    int i; for (i = 0; i < n; ++i) { NLW_L[i] = lens[i]; NLW_W[i] = w[i]; }
}
TRS_EXPORT void sim_cfg_clear_profiles(void) { PROF_N = 0; }
TRS_EXPORT int sim_profile_key_id(const char* name) {
    int i;
    for (i = 0; i < P_KEY_COUNT; ++i)
        if (strcmp(PROFILE_KEYS[i], name) == 0) return i;
    return -1;
}
TRS_EXPORT int sim_cfg_add_profile(double weight, int n, const int* key_ids, const double* vals) {
    if (PROF_N >= MAX_PROFILES) return 0;
    if (n > P_KEY_COUNT) n = P_KEY_COUNT;
    Profile* p = &PROF[PROF_N];
    p->weight = weight; p->n = n;
    int i; for (i = 0; i < n; ++i) { p->key[i] = key_ids[i]; p->val[i] = vals[i]; }
    PROF_N++;
    double tot = 0.0, c = 0.0;
    for (i = 0; i < PROF_N; ++i) tot += PROF[i].weight;
    for (i = 0; i < PROF_N; ++i) { c += PROF[i].weight; PROF_CDF[i] = c / tot; }
    return 1;
}
TRS_EXPORT void sim_set_env_params(double step_dt, int max_episode_steps,
                                   double reward_scale, int reward_mode_score,
                                   double death_penalty, int action_history_len,
                                   int frame_stack, int pj_velocity_invariant) {
    EP_step_dt = step_dt;
    EP_max_steps = max_episode_steps;
    EP_reward_scale = reward_scale;
    EP_reward_score = reward_mode_score;
    EP_death_penalty = death_penalty;
    EP_K = action_history_len < 0 ? 0 : (action_history_len > MAX_K ? MAX_K : action_history_len);
    EP_FS = frame_stack < 1 ? 1 : (frame_stack > MAX_FS ? MAX_FS : frame_stack);
    EP_pj_vel_inv = pj_velocity_invariant;
}
/* DIAGNOSTIC: read back a shaping coefficient as the C++ actually holds it. Added after a knob
   read INERT and the plumbing looked correct at every Python-side layer -- guessing across an FFI
   boundary is how you burn an afternoon. */
TRS_EXPORT double sim_get_shaping(int i) { return (i >= 0 && i < SHP_COUNT) ? SHP[i] : -1.0; }
TRS_EXPORT int sim_shaping_count(void) { return SHP_COUNT; }
TRS_EXPORT void sim_set_shaping(const double* coeffs, int n) {
    int i;
    for (i = 0; i < SHP_COUNT; ++i) SHP[i] = (i < n) ? coeffs[i] : 0.0;
}
TRS_EXPORT void sim_set_track(const int* cells, int n) {
    if (n <= 0) { INJ_N = 0; return; }
    if (n > MAX_INJ) n = MAX_INJ;
    INJ_N = n;
    int i;
    for (i = 0; i < n; ++i) {
        const int* c = cells + i * 11;
        INJ[i].type  = (short)c[0];
        INJ[i].level = (short)c[1];
        INJ[i].obst  = (short)c[2];
        INJ[i].nlane = (short)c[9];
        INJ[i].flags = (unsigned char)((c[3] ? F_TURN : 0) | (c[4] ? F_CANL : 0)
                     | (c[5] ? F_CANR : 0) | (c[6] ? F_COIN : 0)
                     | (c[7] ? F_WIDE : 0) | (c[8] ? F_STEPUP : 0)
                     | (c[9] != 0 ? F_NARROW : 0) | (c[10] ? F_SDLOW : 0));
    }
}

TRS_EXPORT void sim_init(void* buf, int n_envs, unsigned long long seed0) {
    State* st = (State*)buf;
    int i;
    for (i = 0; i < n_envs; ++i) {
        memset(&st[i], 0, sizeof(State));
        reset_env(&st[i], (long long)(seed0 + (unsigned long long)i * 7919ULL));
    }
}
TRS_EXPORT void sim_reset_all(void* buf, int n_envs, long long seed, float* obs) {
    State* st = (State*)buf;
    int i, ol = sim_obs_len();
    for (i = 0; i < n_envs; ++i) {
        reset_env(&st[i], seed >= 0 ? (seed + (long long)i * 7919LL) : -1);
        if (obs) full_obs(&st[i], obs + (size_t)i * ol);
    }
}
TRS_EXPORT void sim_reset_one(void* buf, int env, long long seed, float* obs) {
    State* s = ((State*)buf) + env;
    reset_env(s, seed);
    if (obs) full_obs(s, obs);
}

TRS_EXPORT void sim_step(void* buf, int n_envs, const int* actions, int auto_reset,
                         float* obs, double* rew, unsigned char* flags,
                         float* term_obs, double* ep_ret, int* ep_len,
                         double* ep_dist, int* death_type) {
    State* st = (State*)buf;
    const int ol = sim_obs_len();
    const double dt = EP_step_dt;
    int e;
    for (e = 0; e < n_envs; ++e) {
        State* s = &st[e];
        int a = actions[e];
        if (a < 0) a = 0;
        if (a >= N_ACTIONS) a = N_ACTIONS - 1;

        if (!s->is_dead) {
            apply_action(s, a);
            advance(s, dt);
        }
        double raw[RAW_CAP];
        snapshot(s, raw);

        double reward;
        if (EP_reward_score) reward = (raw[R_score] - s->prev_score) * EP_reward_scale;
        else {
            double d = raw[R_distance] - s->prev_distance;
            if (d < 0.0) d = 0.0;
            reward = d * EP_reward_scale;
        }
        reward += shaping_reward(s, a, s->prev_raw, s->has_prev,
                                 s->has_prev ? s->prev_raw[R_velocity] : 0.0,
                                 raw[R_velocity]);
        int terminated = (raw[R_is_dead] != 0.0 || raw[R_is_game_over] != 0.0);
        if (terminated) reward -= EP_death_penalty;

        s->prev_score = raw[R_score];
        s->prev_distance = raw[R_distance];
        memcpy(s->prev_raw, raw, sizeof(raw));
        s->has_prev = 1;
        s->last_action = a; s->has_last_action = 1;
        s->jump_recency  = (a == A_JUMP)  ? 0 : (s->jump_recency  < 99 ? s->jump_recency + 1 : 99);
        s->slide_recency = (a == A_SLIDE) ? 0 : (s->slide_recency < 99 ? s->slide_recency + 1 : 99);
        s->elapsed++;
        int over_cap = (s->elapsed >= EP_max_steps);

        if (EP_K > 0) {
            s->act_hist[s->act_hist_n % MAX_K] = a;
            s->act_hist_n++;
        }
        push_frame(s, raw);

        s->ep_ret += reward;
        s->ep_len++;
        s->ep_dist = raw[R_distance];

        if (rew) rew[e] = reward;
        if (flags) flags[e] = (unsigned char)((terminated ? 1 : 0) | (over_cap ? 2 : 0));
        if (death_type) death_type[e] = s->death_type;
        if (ep_ret) ep_ret[e] = s->ep_ret;
        if (ep_len) ep_len[e] = s->ep_len;
        if (ep_dist) ep_dist[e] = s->ep_dist;

        float* to = term_obs ? term_obs + (size_t)e * ol : 0;
        if (to) full_obs(s, to);
        if (auto_reset && (terminated || over_cap)) {
            s->episodes++;
            reset_env(s, -1);
        }
        if (obs) full_obs(s, obs + (size_t)e * ol);
    }
}

TRS_EXPORT void sim_get_raw(void* buf, int env, double* out) {
    State* s = ((State*)buf) + env;
    if (s->has_prev) memcpy(out, s->prev_raw, sizeof(double) * RAW_LEN);
    else snapshot(s, out);
}
TRS_EXPORT void sim_get_obs(void* buf, int env, float* out) {
    full_obs(((State*)buf) + env, out);
}
TRS_EXPORT void sim_approach_turn_hint(void* buf, int n_envs, int* out) {
    State* st = (State*)buf;
    int e;
    for (e = 0; e < n_envs; ++e) {
        State* s = &st[e];
        Cell* c = cell_at(s, s->cell_index);
        Cell* n = cell_at(s, s->cell_index + 1);
        int approach = !(c->flags & F_TURN) && (n->flags & F_TURN)
                       && ((n->flags & F_CANL) || (n->flags & F_CANR));
        out[e] = approach ? ((n->flags & F_CANL) ? A_TURN_L : A_TURN_R) : -1;
    }
}
/* CASCADE HINT (guided exploration, 2026-07-28) — the same device that broke the turn dead-letter.
   MEASURED: v111 dies on 9.62% of three-obstacle chains on DEVICE and 8.16% in SIM (matched, so this
   is NOT a fidelity or coverage gap), while the HUMAN dies on 0 of 45. The sim already presents
   cascades at the right rate AND the right lethality and the policy still cannot do them -- i.e. the
   survivable line exists but is assigned ~no probability, so it is never sampled and no reward can
   reinforce it. That is the turn-dead-letter shape (P(turn|approach) was pinned at 0.000000 through
   five arms until exploring-starts forced the action).
   The oracle: if a must-jump hazard is imminent (within 2 cells) and taking off THIS frame puts the
   landing cell on clear ground, hint JUMP. Otherwise -1 (no opinion) -- notably when NO takeoff is
   safe, so it never teaches a doomed jump. Read-only: reads state, mutates nothing, draws no rng,
   so it cannot affect the equivalence gate (same contract as sim_approach_turn_hint). */
/* STEP-DOWN WALK ORACLE [2026-07-30]. A step-DOWN to lower ground is foot-crossable -- forcing
   NoOp at 600 of them cost the champion 2.6% of median (1437 -> 1400) = FREE -- yet every policy
   jumps ~100% (human 42%). A penalty alone could not move it at doses 0.20 / 1.20 / 8.0 even though
   the coefficient was verified across the FFI (SHP[17]=8.0) and fires on 100% of those takeoffs:
   at a 100% jump rate the ALTERNATIVE IS NEVER SAMPLED, so the penalty is a constant offset that
   advantage normalisation removes. Same hard exploration deadlock as the turn dead-letter, and the
   same fix: inject the counterfactual with probability eps, annealed to 0, so PPO can see it. */
TRS_EXPORT void sim_stepdown_walk_hint(void* buf, int n_envs, int* out) {
    State* st = (State*)buf;
    int e;
    for (e = 0; e < n_envs; ++e) {
        State* s = &st[e];
        out[e] = -1;
        if (s->is_dead || s->jump_t > 0.0 || s->slide_t > 0.0) continue;
        Cell* c  = cell_at(s, s->cell_index);
        Cell* n1 = cell_at(s, s->cell_index + 1);
        int soft = ((c->flags & F_SDLOW) != 0) || ((n1->flags & F_SDLOW) != 0);
        if (!soft) continue;
        /* never suppress a jump a REAL hazard demands */
        int hard = (must_jump_o(c->obst)  && !(c->flags  & F_SDLOW))
                || (must_jump_o(n1->obst) && !(n1->flags & F_SDLOW));
        if (hard) continue;
        out[e] = A_NOOP;                       /* walk it */
    }
}

TRS_EXPORT void sim_cascade_hint(void* buf, int n_envs, int* out) {
    State* st = (State*)buf;
    int e, d;
    for (e = 0; e < n_envs; ++e) {
        State* s = &st[e];
        out[e] = -1;
        if (s->is_dead || s->jump_t > 0.0 || s->slide_t > 0.0) continue;
        /* nearest must-JUMP hazard ahead (0 = the cell we are on) */
        int hz = -1;
        for (d = 0; d <= 2; ++d) {
            if (must_jump_o(cell_at(s, s->cell_index + d)->obst)) { hz = d; break; }
        }
        if (hz < 0) continue;                     /* nothing imminent -> no opinion */
        double cl = s->e_cell_length; if (cl < 1e-6) cl = 1e-6;
        double sp = LAND_SPAN_A * s->velocity + LAND_SPAN_B;
        if (sp < 0.0) sp = 0.0;
        double cp = s->dist_into_cell / cl;
        int land = (int)(cp + sp);
        if (land <= hz) continue;                 /* would come down on/before the hazard */
        Cell* lc = cell_at(s, s->cell_index + land);
        if (must_jump_o(lc->obst) || must_slide_o(lc->obst)) continue;  /* lands on the NEXT one */
        out[e] = A_JUMP;
    }
}
TRS_EXPORT void sim_rng_replay_set(const double* vals, int n) {
    RPL_V = (n > 0) ? vals : 0; RPL_N = n; RPL_I = 0; RPL_OVERRUN = 0;
}
TRS_EXPORT int sim_rng_replay_pos(void)     { return RPL_I; }
TRS_EXPORT int sim_rng_replay_overrun(void) { return RPL_OVERRUN; }

TRS_EXPORT double sim_get_distance(void* buf, int env) { return ((State*)buf)[env].distance_real; }
TRS_EXPORT int sim_get_death_type(void* buf, int env)  { return ((State*)buf)[env].death_type; }
TRS_EXPORT int sim_get_is_dead(void* buf, int env)     { return ((State*)buf)[env].is_dead; }
TRS_EXPORT unsigned long long sim_get_draws(void* buf, int env) { return ((State*)buf)[env].draws; }
TRS_EXPORT int sim_get_profile_idx(void* buf, int env) { return ((State*)buf)[env].profile_idx; }
TRS_EXPORT double sim_get_velocity(void* buf, int env) { return ((State*)buf)[env].velocity; }

/* CALIBRATION PROBE. Drives the generator directly -- no policy, no episode, no deaths -- so the
   EMITTED obstacle mix can be measured on the trainer itself. This matters because the emitted
   share is NOT the weight table: holes also arrive via the wide-gap and compound paths, which
   bypass the type draw entirely, so weights must be fitted against what actually comes out.
   `start_dist` seeds distance_real, which selects the difficulty-ramp row (cascades live late).
   `buf` is a caller-owned State of sim_state_size() bytes. */
TRS_EXPORT double sim_probe_cells(void* buf, long long seed, double start_dist, int n,
                                  int* out_obst, int* out_type) {
    State* s = (State*)buf;
    Cell e;
    int i;
    reset_env(s, seed);
    s->distance_real = start_dist;
    for (i = 0; i < n; ++i) {
        next_cell(s, &e);
        if (out_obst) out_obst[i] = e.obst;
        if (out_type) out_type[i] = e.type;
    }
    return s->distance_real;   /* what the generator actually saw at the end */
}

/* PAIRED-EVALUATION SUPPORT: generate a whole track and hand it back in sim_set_track's 11-int
   layout, so every policy can be scored on the IDENTICAL world.
   WHY: the sim median has a +-20-30% noise floor at n=128 because TRACK LUCK dominates -- one arm
   read +29.6% and -28.9% on two seeds of the same comparison. Track luck is a common random number:
   fix the track and it cancels exactly, which is the only way a 20% effect becomes readable.
   Seeding the same RNG for two policies is NOT enough -- generation and the event rolls share one
   stream, so a different action sequence shifts the draws and the worlds diverge.
   distance_real is advanced by one real cell length per cell, because the difficulty ramp is
   distance-keyed: a track spanning D metres must be generated with the ramp at those distances,
   otherwise it is all early-ramp and never turns slides on. */
TRS_EXPORT double sim_probe_track(void* buf, long long seed, int n, double dist_per_cell,
                                  int* out) {
    State* s = (State*)buf;
    Cell e;
    int i;
    double step = dist_per_cell > 0.0 ? dist_per_cell : 8.7;
    reset_env(s, seed);
    for (i = 0; i < n; ++i) {
        next_cell(s, &e);
        s->distance_real += step;
        out[i * 11 + 0] = e.type;
        out[i * 11 + 1] = e.level;
        out[i * 11 + 2] = e.obst;
        out[i * 11 + 3] = (e.flags & F_TURN)   ? 1 : 0;
        out[i * 11 + 4] = (e.flags & F_CANL)   ? 1 : 0;
        out[i * 11 + 5] = (e.flags & F_CANR)   ? 1 : 0;
        out[i * 11 + 6] = (e.flags & F_COIN)   ? 1 : 0;
        out[i * 11 + 7] = (e.flags & F_WIDE)   ? 1 : 0;
        out[i * 11 + 8] = (e.flags & F_STEPUP) ? 1 : 0;
        out[i * 11 + 9] = e.nlane;
        out[i * 11 + 10] = (e.flags & F_SDLOW) ? 1 : 0;
    }
    return s->distance_real;
}
TRS_EXPORT void sim_stumble(void* buf, int env) { do_stumble(((State*)buf) + env); }

/* -------------------------------------------------- DEFAULT_SIM_CONFIG ---- */
static void cfg_defaults(void) {
    int i;
    for (i = 0; i < K_COUNT; ++i) G[i] = 0.0;
    C_(start_velocity) = 100.0;   C_(max_velocity) = 210.0;  C_(accel_per_sec) = 6.0;
    C_(stumble_slowdown) = 0.85;  C_(stumble_recover_rate) = 1.0;
    C_(lane_edge) = 7.0;          C_(lat_tau) = 0.169;
    C_(lane_recenter_time) = 100.0; C_(turn_center_tol) = 4.0;
    C_(stumble_kill_window) = 2.0;
    C_(compound_prob) = 0.0; C_(compound_after_distance) = 300.0; C_(compound_land_cells) = 2;
    C_(jumpslide_prob) = 0.0; C_(jumpslide_after_distance) = 300.0;
    C_(jumpslide_gap_cells) = 1; C_(jumpslide_flame_frac) = 0.5;
    C_(jumpnarrow_prob) = 0.0; C_(jumpnarrow_after_distance) = 300.0;
    C_(jumpnarrow_gap_cells) = 1; C_(jumpnarrow_strip_cells) = 2;
    C_(jumpgap_prob) = 0.0; C_(jumpgap_after_distance) = 300.0;
    C_(jumpgap_gap_cells) = 2; C_(jumpgap_hole_cells) = 2; C_(jumpgap_chain_max) = 1;
    C_(slidegap_prob) = 0.0; C_(slidegap_after_distance) = 300.0;
    C_(slidegap_gap_cells) = 2; C_(slidegap_hole_cells) = 1;
    C_(turnobstacle_prob) = 0.0;
    C_(min_space_between_obstacles) = 6; C_(min_space_between_turns) = 12;
    C_(device_difficulty_ramp) = 0.0; C_(double_obstacle_percent) = 0.0;
    C_(min_spaces_between_bonus) = 50;
    C_(obstacle_prob) = 0.52; C_(max_back_to_back_obstacles) = 2;
    C_(turn_prob) = 0.18; C_(ramp_distance) = 6000.0;
    C_(min_space_between_turns_late) = 5; C_(turn_prob_late) = 0.5;
    C_(min_clear_after_turn) = 2;
    C_(turn_exec_early) = 0.15; C_(turn_exec_late) = 0.60;
    C_(turn_center_cprog) = 0.44; C_(turn_offset_scale) = 45.0;
    C_(turn_lane_carry_buffered) = 1.0; C_(turn_lane_carry) = 3.0;
    C_(turn_land_kill) = 9.0; C_(turn_window_frac) = 0.60;
    C_(turn_tail_cp_lo) = 0.50; C_(turn_tail_cp_hi) = 0.64; C_(turn_tail_pmax) = 0.0;
    C_(turn_exec_lat_frames) = 0.0;
    C_(turn_approach_buffer) = 0.0; C_(hook_turn_buffer) = 0.0;
    C_(turn_buffer_ttl) = 3; C_(turn_buffer_late_cp) = 0.56;
    C_(turn_send_lat_frames) = 0.0; C_(turn_offjunction_drop) = 0.0;
    C_(turn_drop_debounce_s) = 0.0;
    C_(post_land_gap_rejump_prob) = 0.0;
    C_(held_jump_rejump_solid) = 0.0; C_(held_jump_rejump_gap) = 0.0;
    C_(turn_early_frac) = 0.0; C_(turn_late_frac) = 1.0; C_(turn_buffer_cells) = 2;
    C_(slides_after_distance) = 300.0; C_(default_path_level) = 2;
    C_(zone_change_every_cells) = 32; C_(zone_after_distance) = 550.0;
    C_(narrows_after_distance) = 1600.0; C_(narrow_prob) = 0.70;
    C_(narrow_lane_center) = 4.5; C_(narrow_lane_tol) = 2.5;
    C_(strafe_off_edge_kills) = 1.0; C_(turn_while_strafing_kills) = 1.0;
    C_(wide_gap_prob) = 0.40; C_(wide_gap_extra_cells) = 1; C_(wide_gap_min_airtime) = 0.0;
    C_(turn_linger_frames) = 0; C_(zone_min_level) = 1; C_(zone_max_level) = 3;
    C_(zone_gap_extra_cells_high) = 1; C_(zone_gap_extra_cells_lowmid) = 0;
    C_(stepup_min_height) = 0.95; C_(stepup_clip_band) = 0.0;
    C_(stepup_min_takeoff_prog) = 0.85; C_(gap_entry_grace) = 0.2;
    C_(wide_gap_landing_model) = 0.0;
    C_(wide_gap_span_a) = 0.0143; C_(wide_gap_span_b) = -0.11;
    C_(wide_gap_land_grace) = 0.35;
    C_(hole_landing_model) = 0.0; C_(hole_trap_apex_frac) = 0.5; C_(hole_trap_kill_prob) = 0.35;
    C_(chain_coast_kill_prob) = 0.0; C_(chain_coast_gap) = 2.0;
    C_(land_on_hazard_kill_prob) = 0.0; C_(stack_margin_cells) = 1.0;
    C_(airborne_slide_kill_prob) = 1.0;   /* 1.0 = the old always-fatal rule */
    C_(crevasse_single_cell_prob) = 0.0; C_(turn_aware_lookahead) = 0.0;
    C_(zone_up_prob) = 0.35;   /* the hardcoded literal was `rnd < 0.65 ? down : up` */
    C_(tight_pair_prob) = 0.0; C_(tight_pair_min_cells) = 1;
    C_(crevasse_apex_all) = 0.0; C_(crevasse_graze_worldy) = 0.0;
    C_(crevasse_jump_peak) = 19.2; C_(crevasse_cap_height_hi) = 15.0;
    C_(crevasse_graze_band) = 10.0; C_(crevasse_cap_arc_offset) = 0.0;
    C_(crevasse_takeoff_std) = 0.0;
    C_(path_type_straight) = 1; C_(path_type_turn) = 3; C_(path_type_t_junction) = 7;
    C_(score_per_distance) = 2.7; C_(start_score) = 162; C_(coin_score) = 50;
    C_(dist_per_velsec) = 0.107; C_(dist_per_velsec_slope) = 0.0; C_(start_distance) = 5.3;
    C_(curriculum_max_start_dist) = 0.0; C_(curriculum_deep_start_frac) = 0.35;
    C_(curriculum_dist_power) = 1.5;
    C_(jump_duration) = 0.8; C_(jump_drop_prob) = 0.0;
    C_(jump_dur_stepdown_scale) = 1.0; C_(jump_dur_stepup_scale) = 1.0;
    C_(pair_same_type_prob) = 0.0; C_(pair_same_type_cells) = 4.0;
    C_(pair_anti_repeat_prob) = 0.0;
    C_(type_reject_model) = 0.0; C_(type_reject_redraws) = 1.0;
    { int a, b; for (a = 0; a < TR_DIM; ++a) for (b = 0; b < TR_DIM; ++b) TR[a][b] = 0.0; }
    C_(turnobstacle3_prob) = 0.0;
    C_(jump_refuse_strafing_prob) = 0.6; C_(gap_jump_lip_cp) = 0.2;
    C_(strafe_exec_lat_frames) = 0; C_(jump_exec_lat_frames) = 0; C_(jump_exec_lat_jitter) = 0;
    C_(slide_duration) = 0.7; C_(strafe_duration) = 0.25;
    C_(strafe_glide_time) = 0.6; C_(strafe_reach) = 7.0; C_(stumble_duration) = 0.5;
    C_(cell_length) = 60.0; C_(cell_length_real) = 6.4;
    C_(replay_start_into_cell) = 3.2; C_(lookahead) = 12;
    C_(min_spaces_between_coin_runs) = 12; C_(max_coins_per_run) = 5;
    C_(domain_rand) = 0.0;
    C_(dr_turn_window_frac_lo) = 0.40; C_(dr_turn_window_frac_hi) = 0.52;
    C_(dr_turn_early_frac_lo) = 0.0;   C_(dr_turn_early_frac_hi) = 0.0;
    C_(dr_turn_late_frac_lo) = 1.0;    C_(dr_turn_late_frac_hi) = 1.0;
    C_(dr_jump_duration_lo) = 0.72;    C_(dr_jump_duration_hi) = 0.88;
    C_(dr_slide_duration_lo) = 0.62;   C_(dr_slide_duration_hi) = 0.80;
    C_(dr_velocity_scale_lo) = 0.94;   C_(dr_velocity_scale_hi) = 1.08;
    C_(dr_obstacle_prob_lo) = 0.35;    C_(dr_obstacle_prob_hi) = 0.55;
    C_(dr_turn_prob_lo) = 0.12;        C_(dr_turn_prob_hi) = 0.28;
    C_(dr_cell_length_lo) = 56.0;      C_(dr_cell_length_hi) = 64.0;
    C_(cliff_edge_lane) = 6.3; C_(cliff_edge_kill_frames) = 6; C_(jump_hold_window) = 6;

    /* velocity_curve (DEFAULT_SIM_CONFIG) */
    { static const double D0[32] = {5,10,30,50,70,90,110,130,150,170,190,225,275,325,375,425,
                                    475,525,575,750,1050,1350,1650,1950,2250,2550,2850,3150,
                                    3450,3750,4200,6750};
      static const double V0[32] = {100.0,110.9,125.7,137.1,138.6,139.8,141.3,142.6,144.1,
                                    145.5,146.3,148.7,151.6,154.3,157.6,159.7,162.6,164.9,
                                    167.3,174.0,180.7,184.5,187.5,190.6,194.0,196.8,200.0,
                                    202.7,205.5,207.4,210.0,210.0};
      int a; VC_N = 32; for (a = 0; a < 32; ++a) { VC_D[a] = D0[a]; VC_V[a] = V0[a]; }
    }
    /* obstacle_type_weights, in DEFAULT_SIM_CONFIG insertion order */
    OW_N = 6;
    OW_T[0] = 1; OW_W[0] = 0.22;
    OW_T[1] = 4; OW_W[1] = 0.22;
    OW_T[2] = 5; OW_W[2] = 0.22;
    OW_T[3] = 6; OW_W[3] = 0.14;
    OW_T[4] = 3; OW_W[4] = 0.12;
    OW_T[5] = 2; OW_W[5] = 0.08;
    /* narrow_len_weights */
    NLW_N = 4;
    NLW_L[0] = 1; NLW_W[0] = 0.25;
    NLW_L[1] = 2; NLW_W[1] = 0.34;
    NLW_L[2] = 3; NLW_W[2] = 0.27;
    NLW_L[3] = 4; NLW_W[3] = 0.14;
    /* difficulty_ramp_table */
    RAMP_N = 8;
    { static const double R0[8][6] = {
        {0.0,    6, 12, 0.00, 0, 0},
        {250.0,  5, 10, 0.00, 0, 0},
        {500.0,  4,  8, 0.05, 0, 1},
        {1100.0, 4,  6, 0.05, 0, 1},
        {1500.0, 3,  5, 0.10, 1, 1},
        {2000.0, 3,  4, 0.15, 1, 1},
        {3250.0, 3,  3, 0.17, 1, 1},
        {4500.0, 2,  2, 0.20, 1, 1}};
      int a, b; for (a = 0; a < 8; ++a) for (b = 0; b < 6; ++b) RAMP[a][b] = R0[a][b];
    }
    PROF_N = 0;
}

/* run cfg_defaults at load so a caller that forgets sim_cfg_reset still works */
struct _CfgInit { _CfgInit() { cfg_defaults(); } };
static _CfgInit _cfg_init_instance;
