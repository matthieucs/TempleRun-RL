/* ============================================================================
 * experiments/experiments/cpp_bench/sim_core.cpp  --  BOUNDED FEASIBILITY BENCHMARK, NOT a finished port.
 *
 * A C port of the HOT INNER STEP of src/templerun_rl/sim_env.py:
 *     step() -> _apply_action + _advance + _cross_into_next_cell
 *               + _resolve_obstacle_on_enter + _stumble + _kill
 *     plus the MAIN PATH of _next_cell_procedural (track generation), which the
 *     inner step triggers on every cell crossing and therefore cannot be omitted
 *     without making the benchmark unrepresentative.
 *
 * WHAT IS *NOT* HERE (deliberately, see the report):
 *   - _snapshot / obs.state_to_obs / reward.shaping_reward  (the other ~59% of a
 *     full env.step() in the Python reference)
 *   - the compound-scenario machinery (_queue_compound/_jumpslide/_jumpnarrow/
 *     _jumpgap/_slidegap, _pending_sequence)
 *   - the device difficulty-ramp table, domain randomisation, replay/ext_distance
 *     path, episode_profiles, turn_approach_buffer / hook_turn_buffer branches
 *   - held_jump_rejump, tight_pair, double_obstacle, crevasse_single_cell
 *
 * DETERMINISM: explicit seeded PCG32 per env.  Every rng consumer keeps the
 * Python "prob > 0 first" gating so a disabled feature draws NO random number --
 * that is what keeps a track stream reproducible across config changes, and it
 * is the property a real port must preserve to pass scripts/validate/sim_equivalence.py.
 *
 * Build (GCC 4.6.3 / MinGW):
 *   g++ -O3 -std=c++0x -shared -o sim_core.dll sim_core.cpp
 * ==========================================================================*/

#include <cmath>
#include <cstring>
#include <cstdint>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

/* --------------------------------------------------------------- constants */
#define TRACK_RING 64            /* power of two; must exceed the lookahead */
#define TRACK_MASK (TRACK_RING - 1)
#define GEN_AHEAD  8             /* cells kept generated ahead of cell_index  */

/* ObstacleType enum (sim_env.py) */
#define OBST_NONE          0
#define OBST_HOLE          1
#define OBST_NARROW        2
#define OBST_TREE_STUMBLE  3
#define OBST_TREE_JUMP     4
#define OBST_TREE_SLIDE    5
#define OBST_FLAME_TOWER   6
#define OBST_LEVEL_CHANGE  7

/* Action bytes */
#define A_STRAFE_L 0
#define A_STRAFE_R 1
#define A_JUMP     2
#define A_SLIDE    3
#define A_TURN_L   4
#define A_TURN_R   5
#define A_NOOP     6

/* cell flag bits */
#define F_IS_TURN 0x01
#define F_CAN_L   0x02
#define F_CAN_R   0x04
#define F_COIN    0x08
#define F_WIDE    0x10
#define F_STEPUP  0x20
#define F_NARROW  0x40           /* narrow_lane is not None */

static inline int must_jump(int o) {
    return o == OBST_HOLE || o == OBST_TREE_JUMP ||
           o == OBST_TREE_STUMBLE || o == OBST_LEVEL_CHANGE;
}
static inline int must_slide(int o) {
    return o == OBST_TREE_SLIDE || o == OBST_FLAME_TOWER;
}

/* ------------------------------------------------------------------ config */
/* Mirrors DEFAULT_SIM_CONFIG + the device-true knobs the Python benchmark
 * enables, so the two sides run the SAME branches. */
typedef struct {
    double step_dt, jump_duration, slide_duration, stumble_duration;
    double stumble_slowdown, stumble_recover_rate, stumble_kill_window;
    double lane_edge, lat_tau, cell_length, dist_per_velsec, score_per_distance;
    double start_distance, max_velocity, accel_per_sec;
    double gap_entry_grace, wide_gap_min_airtime;
    double narrow_lane_center, narrow_lane_tol;
    double cliff_edge_lane;  int cliff_edge_kill_frames;
    double gap_jump_lip_cp, jump_refuse_strafing_prob, jump_drop_prob;
    double post_land_gap_rejump_prob;
    int    jump_exec_lat_frames, jump_exec_lat_jitter;
    int    strafe_exec_lat_frames;
    /* turn model */
    double turn_exec_early, turn_exec_late, turn_center_cprog, turn_offset_scale;
    double turn_land_kill, turn_lane_carry, turn_exec_lat_frames;
    double turn_tail_cp_lo, turn_tail_cp_hi, turn_tail_pmax, turn_drop_debounce_s;
    /* crevasse / hole death models */
    int    crevasse_apex_all, crevasse_graze_worldy, wide_gap_landing_model, hole_landing_model;
    double crevasse_cap_arc_offset, crevasse_takeoff_std;
    double crevasse_jump_peak, crevasse_cap_height_hi, crevasse_graze_band;
    double stepup_min_height, stepup_clip_band;
    double wide_gap_span_a, wide_gap_span_b, wide_gap_land_grace;
    double hole_trap_apex_frac, hole_trap_kill_prob;
    /* generation */
    double obstacle_prob, turn_prob, turn_prob_late, ramp_distance;
    double narrow_prob, wide_gap_prob, narrows_after_distance, slides_after_distance;
    int    min_space_between_obstacles, min_space_between_turns, min_space_between_turns_late;
    int    min_clear_after_turn, max_back_to_back_obstacles, zone_change_every_cells;
    int    zone_min_level, zone_max_level, wide_gap_extra_cells, min_spaces_between_bonus;
    double coin_score;
} Config;

static Config G;

static void cfg_defaults(void) {
    memset(&G, 0, sizeof(G));
    G.step_dt = 0.1;            G.jump_duration = 0.8;     G.slide_duration = 0.7;
    G.stumble_duration = 0.5;   G.stumble_slowdown = 0.85; G.stumble_recover_rate = 1.0;
    G.stumble_kill_window = 2.0;
    G.lane_edge = 7.0;          G.lat_tau = 0.169;         G.cell_length = 60.0;
    G.dist_per_velsec = 0.107;  G.score_per_distance = 2.7;
    G.start_distance = 5.3;     G.max_velocity = 210.0;    G.accel_per_sec = 6.0;
    G.gap_entry_grace = 0.2;    G.wide_gap_min_airtime = 0.0;
    G.narrow_lane_center = 4.5; G.narrow_lane_tol = 2.5;
    G.cliff_edge_lane = 6.3;    G.cliff_edge_kill_frames = 6;
    G.gap_jump_lip_cp = 0.2;    G.jump_refuse_strafing_prob = 0.6; G.jump_drop_prob = 0.0;
    G.post_land_gap_rejump_prob = 0.0;
    G.jump_exec_lat_frames = 0;  G.jump_exec_lat_jitter = 0;  G.strafe_exec_lat_frames = 0;
    G.turn_exec_early = 0.15;   G.turn_exec_late = 0.60;   G.turn_center_cprog = 0.44;
    G.turn_offset_scale = 45.0; G.turn_land_kill = 9.0;    G.turn_lane_carry = 3.0;
    G.turn_exec_lat_frames = 0.0;
    G.turn_tail_cp_lo = 0.50;   G.turn_tail_cp_hi = 0.64;  G.turn_tail_pmax = 0.0;
    G.turn_drop_debounce_s = 0.0;
    /* device-true death models ON (same as the Python side of this benchmark) */
    G.crevasse_apex_all = 1;    G.crevasse_graze_worldy = 1;
    G.wide_gap_landing_model = 1; G.hole_landing_model = 1;
    G.crevasse_cap_arc_offset = 0.0; G.crevasse_takeoff_std = 0.0;
    G.crevasse_jump_peak = 19.2; G.crevasse_cap_height_hi = 15.0; G.crevasse_graze_band = 10.0;
    G.stepup_min_height = 0.95; G.stepup_clip_band = 0.25;
    G.wide_gap_span_a = 0.0143; G.wide_gap_span_b = -0.11; G.wide_gap_land_grace = 0.35;
    G.hole_trap_apex_frac = 0.5; G.hole_trap_kill_prob = 0.35;
    G.obstacle_prob = 0.52;     G.turn_prob = 0.18;        G.turn_prob_late = 0.5;
    G.ramp_distance = 6000.0;   G.narrow_prob = 0.70;      G.wide_gap_prob = 0.40;
    G.narrows_after_distance = 1600.0; G.slides_after_distance = 300.0;
    G.min_space_between_obstacles = 6;  G.min_space_between_turns = 12;
    G.min_space_between_turns_late = 5; G.min_clear_after_turn = 2;
    G.max_back_to_back_obstacles = 2;   G.zone_change_every_cells = 32;
    G.zone_min_level = 1;       G.zone_max_level = 3;      G.wide_gap_extra_cells = 1;
    G.min_spaces_between_bonus = 50;    G.coin_score = 10.0;
}

/* calibrated velocity curve (DistanceTraveled -> velocity), sim_env.py */
#define NVC 32
static const double VC_D[NVC] = {5,10,30,50,70,90,110,130,150,170,190,225,275,325,375,425,475,
                                 525,575,750,1050,1350,1650,1950,2250,2550,2850,3150,3450,3750,
                                 4200,6750};
static const double VC_V[NVC] = {100.0,110.9,125.7,137.1,138.6,139.8,141.3,142.6,144.1,145.5,
                                 146.3,148.7,151.6,154.3,157.6,159.7,162.6,164.9,167.3,174.0,
                                 180.7,184.5,187.5,190.6,194.0,196.8,200.0,202.7,205.5,207.4,
                                 210.0,210.0};

/* np.interp equivalent; plain forward scan (no binary search) so this is if
 * anything MORE work than numpy does, not less. */
static double vel_curve(double d) {
    if (d <= VC_D[0]) return VC_V[0];
    for (int i = 1; i < NVC; ++i) {
        if (d <= VC_D[i]) {
            double t = (d - VC_D[i-1]) / (VC_D[i] - VC_D[i-1]);
            return VC_V[i-1] + t * (VC_V[i] - VC_V[i-1]);
        }
    }
    return VC_V[NVC-1];
}

/* -------------------------------------------------------------------- cell */
typedef struct {
    signed char type;
    signed char level;
    signed char obst;
    signed char narrow_lane;    /* -1 / +1, meaningful only when F_NARROW */
    unsigned char flags;
    unsigned char _pad[3];
} Cell;

/* ------------------------------------------------------------------- state */
typedef struct {
    /* --- deterministic RNG (PCG32) --- */
    uint64_t rng_state, rng_inc;
    int   norm_has_spare; double norm_spare;

    /* --- runner --- */
    double lane, lat_cmd, lat_target, velocity, stumble_vel;
    int    has_stumble_vel;             /* stumble_vel is None in Python */
    double distance, distance_real, score, dist_into_cell;
    double jump_t, slide_t, stumble_t, stumble_kill_t, strafe_t, turn_debounce_t;
    double jump_takeoff_pos; int has_takeoff;
    int    tilt_dir, facing;
    int    cell_index, gen_head;
    int    is_dead, death_type;
    int    frames_since_land, cliff_edge_frames, pending_jump_frames;
    int    gap_grace_cell;              /* -1 = None */
    int    turn_landed;
    double lat_buf[8]; int lat_buf_n;   /* strafe_exec_lat_frames FIFO */

    /* --- generation --- */
    int cells_since_obstacle, cells_since_turn, cells_since_zone, cells_since_bonus;
    int zone_level, b2b_obstacles;
    int pending_gap, pending_gap_type, pending_gap_level, pending_gap_obst, pending_gap_stepup;
    int pending_narrow, pending_narrow_lane, pending_narrow_type;

    /* --- episode bookkeeping (for the benchmark's auto-reset) --- */
    int    steps, episodes; double last_ep_distance;

    Cell track[TRACK_RING];
} State;

EXPORT int sim_state_size(void) { return (int)sizeof(State); }

/* ------------------------------------------------------------------- PCG32 */
static inline uint32_t pcg32(State* s) {
    uint64_t old = s->rng_state;
    s->rng_state = old * 6364136223846793005ULL + s->rng_inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}
static inline double rnd(State* s) {           /* [0,1) with 53 bits */
    uint64_t hi = (uint64_t)pcg32(s) >> 5;
    uint64_t lo = (uint64_t)pcg32(s) >> 6;
    return (double)((hi << 26) + lo) * (1.0 / 9007199254740992.0);
}
static double rnd_normal(State* s) {           /* Box-Muller, explicit spare */
    if (s->norm_has_spare) { s->norm_has_spare = 0; return s->norm_spare; }
    double u1, u2;
    do { u1 = rnd(s); } while (u1 <= 1e-12);
    u2 = rnd(s);
    double r = sqrt(-2.0 * log(u1)), th = 6.283185307179586 * u2;
    s->norm_spare = r * sin(th); s->norm_has_spare = 1;
    return r * cos(th);
}
static inline int rnd_int(State* s, int lo, int hi) {   /* inclusive */
    return lo + (int)(rnd(s) * (double)(hi - lo + 1));
}

/* ------------------------------------------------------------- track access */
static inline Cell* cell_at(State* s, int idx) { return &s->track[idx & TRACK_MASK]; }

/* ------------------------------------------------- procedural generation */
/* Main path of _next_cell_procedural: pending gap / pending narrow tails, the
 * junction decision, the zone crevasse, the normal obstacle placement with the
 * min-space + back-to-back gates, narrows, wide gaps, coins.  Same rng draw
 * ORDER and same gating shape as the Python. */
static void gen_cell(State* s, Cell* out) {
    s->cells_since_obstacle++;
    s->cells_since_turn++;
    s->cells_since_bonus++;
    s->cells_since_zone++;

    out->flags = 0; out->narrow_lane = 0; out->obst = OBST_NONE;
    out->level = (signed char)s->zone_level; out->type = 1;

    /* --- wide-gap continuation cell (2nd cell of a 2-cell gap) --- */
    if (s->pending_gap > 0) {
        s->pending_gap--;
        s->cells_since_obstacle = 0;
        out->type  = (signed char)s->pending_gap_type;
        out->level = (signed char)s->pending_gap_level;
        out->obst  = (signed char)s->pending_gap_obst;
        out->flags = F_WIDE;
        if (s->pending_gap == 0 && s->pending_gap_stepup) {
            out->flags |= F_STEPUP;
            s->pending_gap_stepup = 0;
        }
        return;
    }
    /* --- narrow-strip continuation --- */
    if (s->pending_narrow > 0) {
        s->pending_narrow--;
        s->cells_since_obstacle = 0;
        out->type = (signed char)s->pending_narrow_type;
        out->level = (signed char)s->zone_level;
        out->narrow_lane = (signed char)s->pending_narrow_lane;
        out->flags = F_NARROW;
        return;
    }

    int wide = 0, stepup = 0, has_narrow = 0, narrow_lane = 0;
    int level = s->zone_level;

    /* difficulty ramp (lerp) */
    double diff = (s->distance_real - G.start_distance) / (G.ramp_distance > 1.0 ? G.ramp_distance : 1.0);
    if (diff < 0.0) diff = 0.0; if (diff > 1.0) diff = 1.0;
    double min_turns = (double)G.min_space_between_turns * (1.0 - diff)
                     + (double)G.min_space_between_turns_late * diff;
    double turn_p = G.turn_prob * (1.0 - diff) + G.turn_prob_late * diff;

    /* --- junction decision --- */
    int is_turn = 0, can_l = 0, can_r = 0, ptype = 1;
    if ((double)s->cells_since_turn >= min_turns) {
        if (rnd(s) < turn_p) {
            is_turn = 1;
            s->cells_since_turn = 0;
            double roll = rnd(s);
            if (roll < 0.4)      { can_l = 1; can_r = 0; ptype = 3; }
            else if (roll < 0.8) { can_l = 0; can_r = 1; ptype = 3; }
            else                 { can_l = 1; can_r = 1; ptype = 8; }
        }
    }

    int obstacle = OBST_NONE;
    /* --- zone transition crevasse (takes priority over a normal obstacle) --- */
    if (!is_turn
            && s->cells_since_turn >= G.min_clear_after_turn
            && s->cells_since_obstacle >= G.min_space_between_obstacles
            && s->cells_since_zone >= G.zone_change_every_cells
            && s->distance_real >= 0.0) {
        obstacle = OBST_LEVEL_CHANGE;
        s->cells_since_zone = 0;
        s->cells_since_obstacle = 0;
        s->b2b_obstacles = 0;
        int prev_zone = s->zone_level;
        if (s->zone_level <= G.zone_min_level)      s->zone_level += 1;
        else if (s->zone_level >= G.zone_max_level) s->zone_level -= 1;
        else                                        s->zone_level += (rnd(s) < 0.65 ? -1 : 1);
        level = s->zone_level;
        s->pending_gap_obst  = OBST_LEVEL_CHANGE;
        s->pending_gap_type  = 35;
        s->pending_gap_level = s->zone_level;
        ptype = 35;
        s->pending_gap = 1;                      /* -> 1 continuation cell = the raised ledge */
        s->pending_gap_stepup = (s->zone_level > prev_zone) ? 1 : 0;
        stepup = 0;
        wide = 1;
    }

    /* --- normal obstacle placement --- */
    if (obstacle == OBST_NONE && !is_turn
            && s->cells_since_turn >= G.min_clear_after_turn
            && s->cells_since_obstacle >= G.min_space_between_obstacles) {
        int place = (rnd(s) < G.obstacle_prob);
        if (s->b2b_obstacles >= G.max_back_to_back_obstacles) place = 0;
        if (place) {
            s->cells_since_obstacle = 0;
            s->b2b_obstacles++;
            int narrows_ok = (s->distance_real >= G.narrows_after_distance);
            if (narrows_ok && rnd(s) < G.narrow_prob) {
                narrow_lane = (rnd(s) < 0.5) ? -1 : 1;
                ptype = (narrow_lane < 0) ? 19 : 22;
                /* device-measured strip length weights {1:.25, 2:.34, 3:.27, 4:.14} */
                double u = rnd(s); int strip;
                if (u < 0.25)      strip = 1;
                else if (u < 0.59) strip = 2;
                else if (u < 0.86) strip = 3;
                else               strip = 4;
                s->pending_narrow = strip - 1;
                s->pending_narrow_lane = narrow_lane;
                s->pending_narrow_type = ptype;
                has_narrow = 1;
            } else {
                /* _sample_obstacle_type: weighted draw, slides gated by distance */
                int slides_ok = (s->distance_real >= G.slides_after_distance);
                double u = rnd(s);
                if (slides_ok) {
                    /* 1:.22 4:.22 5:.22 6:.14 3:.12 2:.08 */
                    if      (u < 0.22) obstacle = OBST_HOLE;
                    else if (u < 0.44) obstacle = OBST_TREE_JUMP;
                    else if (u < 0.66) obstacle = OBST_TREE_SLIDE;
                    else if (u < 0.80) obstacle = OBST_FLAME_TOWER;
                    else if (u < 0.92) obstacle = OBST_TREE_STUMBLE;
                    else               obstacle = OBST_NARROW;
                } else {
                    /* renormalised without 5/6: 1:.3667 4:.3667 3:.2 2:.1333 */
                    if      (u < 0.366667) obstacle = OBST_HOLE;
                    else if (u < 0.733333) obstacle = OBST_TREE_JUMP;
                    else if (u < 0.933333) obstacle = OBST_TREE_STUMBLE;
                    else                   obstacle = OBST_NARROW;
                }
                if (obstacle == OBST_HOLE && rnd(s) < G.wide_gap_prob) {
                    s->pending_gap = G.wide_gap_extra_cells;
                    s->pending_gap_obst = OBST_HOLE;
                    s->pending_gap_type = 12;
                    s->pending_gap_level = level;
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

    /* --- coin / bonus cell --- */
    int is_coin = 0;
    if (!is_turn && obstacle == OBST_NONE && s->cells_since_bonus >= G.min_spaces_between_bonus) {
        if (rnd(s) < 0.5) { is_coin = 1; s->cells_since_bonus = 0; }
    }

    out->type  = (signed char)(ptype > 127 ? 127 : ptype);
    out->level = (signed char)level;
    out->obst  = (signed char)obstacle;
    out->narrow_lane = (signed char)narrow_lane;
    out->flags = (unsigned char)((is_turn ? F_IS_TURN : 0) | (can_l ? F_CAN_L : 0)
                | (can_r ? F_CAN_R : 0) | (is_coin ? F_COIN : 0)
                | (wide ? F_WIDE : 0) | (stepup ? F_STEPUP : 0)
                | (has_narrow ? F_NARROW : 0));
}

static void extend_track(State* s) {
    while (s->gen_head < s->cell_index + GEN_AHEAD) {
        gen_cell(s, cell_at(s, s->gen_head));
        s->gen_head++;
    }
}

/* -------------------------------------------------------------- primitives */
static inline double cell_progress(State* s) {
    double cp = s->dist_into_cell / G.cell_length;
    if (cp < 0.0) cp = 0.0; if (cp > 1.0) cp = 1.0;
    return cp;
}
static inline double long_pos(State* s) { return (double)s->cell_index + cell_progress(s); }

static void do_kill(State* s, int dt) { s->is_dead = 1; s->death_type = dt; }

static void do_stumble(State* s) {
    if (s->is_dead) return;
    if (s->stumble_t > 0.0 || s->stumble_kill_t < G.stumble_kill_window) { do_kill(s, 1); return; }
    s->velocity *= G.stumble_slowdown;
    s->stumble_vel = s->velocity; s->has_stumble_vel = 1;
    s->lane = 0.0; s->tilt_dir = 0; s->strafe_t = 0.0;
    s->stumble_t = G.stumble_duration;
    s->stumble_kill_t = 0.0;
}

/* ----------------------------------------------- _resolve_obstacle_on_enter */
static void resolve_obstacle_on_enter(State* s, Cell* c) {
    if (c->flags & F_NARROW) {
        double target = (double)c->narrow_lane * G.narrow_lane_center;
        if (fabs(s->lane - target) > G.narrow_lane_tol) { do_kill(s, 1); return; }
    }
    int obst = c->obst;
    int apex_cell = (c->flags & F_STEPUP) ? 1 : 0;
    if (!apex_cell && (G.crevasse_apex_all || G.crevasse_graze_worldy)
            && obst == OBST_LEVEL_CHANGE && (c->flags & F_WIDE)) {
        Cell* nxt = cell_at(s, s->cell_index + 1);
        apex_cell = !(nxt->flags & F_WIDE);
    }
    if (must_jump(obst) && apex_cell) {
        if (s->jump_t <= 0.0 || !s->has_takeoff) { do_kill(s, 2); return; }
        double span = G.jump_duration * s->velocity / G.cell_length;
        double d = (double)s->cell_index - s->jump_takeoff_pos;
        d += G.crevasse_cap_arc_offset;
        if (G.crevasse_takeoff_std > 0.0) d += rnd_normal(s) * G.crevasse_takeoff_std;
        double frac = d / (span > 1e-6 ? span : 1e-6);
        double t = 2.0 * frac - 1.0;
        double height = 1.0 - t * t;
        if (G.crevasse_graze_worldy) {
            double arc_y = G.crevasse_jump_peak * height;
            double cap_h = (c->level >= 3) ? G.crevasse_cap_height_hi : 0.0;
            if (arc_y < cap_h - G.crevasse_graze_band)  do_kill(s, 2);
            else if (arc_y < cap_h)                     do_stumble(s);
            return;
        }
        if (height < G.stepup_min_height) {
            if (G.stepup_clip_band > 0.0 && height >= G.stepup_min_height - G.stepup_clip_band)
                do_stumble(s);
            else
                do_kill(s, 2);
        }
        return;
    }
    if (must_jump(obst)) {
        double need = (c->flags & F_WIDE) ? G.wide_gap_min_airtime : 0.0;
        if (s->jump_t <= need) {
            if ((c->flags & F_WIDE) && G.gap_entry_grace > 0.0) s->gap_grace_cell = s->cell_index;
            else do_kill(s, obst == OBST_HOLE ? 1 : 2);
            return;
        }
        if (G.wide_gap_landing_model && (c->flags & F_WIDE) && obst == OBST_HOLE && s->has_takeoff) {
            Cell* nxt = cell_at(s, s->cell_index + 1);
            if (!(nxt->flags & F_WIDE)) {
                double span = G.wide_gap_span_a * s->velocity + G.wide_gap_span_b;
                double d = (double)s->cell_index - s->jump_takeoff_pos;
                if (span < d + 1.0 - G.wide_gap_land_grace) { do_kill(s, 2); return; }
            }
        }
        if (G.hole_trap_kill_prob > 0.0 && G.hole_landing_model && obst == OBST_HOLE
                && !(c->flags & F_WIDE) && s->has_takeoff) {
            double span = G.wide_gap_span_a * s->velocity + G.wide_gap_span_b;
            double frac = ((double)s->cell_index - s->jump_takeoff_pos) / (span > 1e-6 ? span : 1e-6);
            if (frac > G.hole_trap_apex_frac
                    && (G.hole_trap_kill_prob >= 1.0 || rnd(s) < G.hole_trap_kill_prob)) {
                do_kill(s, 1); return;
            }
        }
        return;
    }
    if (must_slide(obst)) {
        if (s->slide_t <= 0.0 || s->jump_t > 0.0)
            do_kill(s, obst == OBST_FLAME_TOWER ? 5 : 3);
    }
}

/* ------------------------------------------------- _cross_into_next_cell */
static void cross_into_next_cell(State* s) {
    Cell* leaving = cell_at(s, s->cell_index);
    if ((leaving->flags & F_IS_TURN) && (leaving->flags & (F_CAN_L | F_CAN_R))) {
        do_kill(s, 1);            /* missed turn -> Fall */
        return;
    }
    s->cell_index++;
    extend_track(s);
    Cell* entered = cell_at(s, s->cell_index);
    resolve_obstacle_on_enter(s, entered);
    if (s->is_dead) return;
    if (entered->flags & F_COIN) {
        s->score += G.coin_score;
        entered->flags &= (unsigned char)~F_COIN;
    }
}

/* -------------------------------------------------------------- turn model */
static void execute_turn(State* s, Cell* c, int want_left, double swipe_cp) {
    double carry = G.turn_lane_carry;
    if (G.turn_exec_lat_frames > 0.0)
        swipe_cp += s->velocity * G.turn_exec_lat_frames * G.step_dt / G.cell_length;
    double sign = want_left ? 1.0 : -1.0;
    double offset = sign * (swipe_cp - G.turn_center_cprog) * G.turn_offset_scale + s->lane * carry;
    /* facing rotation (N=1,E=2,S=3,W=4) */
    static const int LEFT_OF[5]  = {0, 4, 1, 2, 3};
    static const int RIGHT_OF[5] = {0, 2, 3, 4, 1};
    s->facing = want_left ? LEFT_OF[s->facing] : RIGHT_OF[s->facing];
    if (fabs(offset) > G.turn_land_kill) { do_kill(s, 1); return; }
    if (G.turn_tail_pmax > 0.0 && swipe_cp > G.turn_tail_cp_lo) {
        double p = (swipe_cp - G.turn_tail_cp_lo) / (G.turn_tail_cp_hi - G.turn_tail_cp_lo) * G.turn_tail_pmax;
        if (p > G.turn_tail_pmax) p = G.turn_tail_pmax;
        if (rnd(s) < p) { do_kill(s, 1); return; }
    }
    s->lane = offset;
    s->tilt_dir = 0;
    s->turn_landed = 1;
    c->flags &= (unsigned char)~(F_IS_TURN | F_CAN_L | F_CAN_R);
    s->cells_since_turn = 0;
}

static void attempt_turn(State* s, int a) {
    if (G.turn_drop_debounce_s > 0.0) {
        if (s->turn_debounce_t > 0.0) return;
        s->turn_debounce_t = G.turn_drop_debounce_s;
    }
    int want_left = (a == A_TURN_L);
    Cell* c = cell_at(s, s->cell_index);
    double cp = cell_progress(s);
    int side_ok = want_left ? (c->flags & F_CAN_L) : (c->flags & F_CAN_R);
    if ((c->flags & F_IS_TURN) && cp >= G.turn_exec_early && cp <= G.turn_exec_late && side_ok) {
        execute_turn(s, c, want_left, cp);
        return;
    }
    Cell* nxt = cell_at(s, s->cell_index + 1);
    int near_junction = (c->flags & F_IS_TURN) || (nxt->flags & F_IS_TURN);
    if (near_junction) do_stumble(s);
    else               do_kill(s, 1);    /* turn on a plain straight -> off the path */
}

/* ------------------------------------------------------------ _apply_action */
static void apply_action(State* s, int a) {
    if (a == A_JUMP) {
        if (s->jump_t <= 0.0 && s->slide_t <= 0.0) {
            if (G.jump_drop_prob > 0.0 && rnd(s) < G.jump_drop_prob) return;
            if (fabs(s->lat_target - s->lane) > 1.8 && rnd(s) < G.jump_refuse_strafing_prob) return;
            Cell* now = cell_at(s, s->cell_index);
            if ((now->flags & F_WIDE) && cell_progress(s) > G.gap_jump_lip_cp) {
                double plr = G.post_land_gap_rejump_prob;
                if (!(plr > 0.0 && s->frames_since_land <= 1 && rnd(s) < plr)) return;
            }
            if (G.jump_exec_lat_frames > 0 && s->pending_jump_frames <= 0) {
                int jit = G.jump_exec_lat_jitter;
                int f = G.jump_exec_lat_frames + (jit > 0 ? rnd_int(s, -jit, jit) : 0);
                s->pending_jump_frames = (f < 1) ? 1 : f;
            } else if (G.jump_exec_lat_frames <= 0) {
                s->jump_t = G.jump_duration;
                s->jump_takeoff_pos = long_pos(s);
                s->has_takeoff = 1;
            }
        }
    } else if (a == A_SLIDE) {
        if (s->slide_t <= 0.0 && s->jump_t <= 0.0) s->slide_t = G.slide_duration;
    } else if (a == A_STRAFE_L) {
        s->lat_cmd = -G.lane_edge; s->tilt_dir = -1;
    } else if (a == A_STRAFE_R) {
        s->lat_cmd = +G.lane_edge; s->tilt_dir = +1;
    } else if (a == A_TURN_L || a == A_TURN_R) {
        attempt_turn(s, a);
    } else if (a == A_NOOP) {
        s->lat_cmd = 0.0; s->tilt_dir = 0;
    }
}

/* ----------------------------------------------------------------- _advance */
static void advance(State* s, double dt) {
    if (s->pending_jump_frames > 0) {
        s->pending_jump_frames--;
        if (s->pending_jump_frames == 0 && s->jump_t <= 0.0 && s->slide_t <= 0.0) {
            s->jump_t = G.jump_duration;
            s->jump_takeoff_pos = long_pos(s);
            s->has_takeoff = 1;
        }
    }
    int was_airborne = (s->jump_t > 0.0);
    s->jump_t -= dt; if (s->jump_t < 0.0) s->jump_t = 0.0;
    if (was_airborne && s->jump_t <= 0.0) s->frames_since_land = 0;
    else if (s->frames_since_land < 99)   s->frames_since_land++;
    s->slide_t -= dt;         if (s->slide_t < 0.0) s->slide_t = 0.0;
    s->turn_debounce_t -= dt; if (s->turn_debounce_t < 0.0) s->turn_debounce_t = 0.0;
    s->strafe_t -= dt;        if (s->strafe_t < 0.0) s->strafe_t = 0.0;

    if (G.strafe_exec_lat_frames > 0) {
        int n = G.strafe_exec_lat_frames; if (n > 8) n = 8;
        s->lat_buf[s->lat_buf_n % 8] = s->lat_cmd;
        s->lat_buf_n++;
        if (s->lat_buf_n > n) s->lat_target = s->lat_buf[(s->lat_buf_n - n - 1) % 8];
    } else {
        s->lat_target = s->lat_cmd;
    }

    double edge = G.lane_edge;
    if (s->turn_landed) {
        s->turn_landed = 0;
    } else {
        double alpha = 1.0 - exp(-dt / (G.lat_tau > 1e-3 ? G.lat_tau : 1e-3));
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
    if (cell_at(s, s->cell_index)->level >= 3 && fabs(s->lane) >= G.cliff_edge_lane) {
        s->cliff_edge_frames++;
        if (s->cliff_edge_frames >= G.cliff_edge_kill_frames) { do_kill(s, 1); return; }
    } else {
        s->cliff_edge_frames = 0;
    }

    s->distance_real += s->velocity * dt * G.dist_per_velsec;
    double cell_len = G.cell_length;

    double curve_v = vel_curve(s->distance_real);
    if (s->has_stumble_vel) {
        s->stumble_vel += G.stumble_recover_rate * dt;
        if (s->stumble_vel > curve_v) s->stumble_vel = curve_v;
        s->velocity = s->stumble_vel;
        if (s->stumble_vel >= curve_v) s->has_stumble_vel = 0;
    } else {
        s->velocity = curve_v;
    }

    double moved = s->velocity * dt;
    s->distance += moved;
    s->score += moved * G.score_per_distance;
    s->dist_into_cell += moved;

    int guard = 0;
    while (s->dist_into_cell >= cell_len && !s->is_dead) {
        if (++guard > 64) break;
        s->dist_into_cell -= cell_len;
        cross_into_next_cell(s);
    }

    if (!s->is_dead && s->jump_t > 0.0) {
        int oc = cell_at(s, s->cell_index)->obst;
        if (must_slide(oc)) do_kill(s, oc == OBST_FLAME_TOWER ? 5 : 3);
    }
    if (s->gap_grace_cell >= 0 && !s->is_dead) {
        if (s->cell_index != s->gap_grace_cell || s->jump_t > 0.0) {
            s->gap_grace_cell = -1;
        } else if ((s->dist_into_cell / cell_len) >= G.gap_entry_grace) {
            Cell* c = cell_at(s, s->cell_index);
            do_kill(s, c->obst == OBST_HOLE ? 1 : 2);
            s->gap_grace_cell = -1;
        }
    }
}

/* -------------------------------------------------------------------- API */
static void reset_env(State* s, uint64_t seed) {
    uint64_t rs = s->rng_state, ri = s->rng_inc;   /* keep the stream across resets */
    int eps = s->episodes, steps = s->steps;
    double lastd = s->last_ep_distance;
    memset(s, 0, sizeof(State));
    if (seed) {
        s->rng_state = 0u; s->rng_inc = (seed << 1u) | 1u;
        pcg32(s); s->rng_state += 0x853c49e6748fea9bULL; pcg32(s);
    } else { s->rng_state = rs; s->rng_inc = ri; }
    s->episodes = eps; s->steps = steps; s->last_ep_distance = lastd;

    s->facing = 1;                 /* DIR_N */
    s->zone_level = 2;
    s->distance_real = G.start_distance;
    s->velocity = vel_curve(G.start_distance);
    s->stumble_kill_t = G.stumble_kill_window;
    s->gap_grace_cell = -1;
    s->cells_since_turn = 100; s->cells_since_obstacle = 100;
    s->cells_since_zone = 0;   s->cells_since_bonus = 0;
    s->cell_index = 0; s->gen_head = 0;
    extend_track(s);
}

EXPORT void sim_init(void* buf, int n_envs, unsigned long long seed0) {
    cfg_defaults();
    State* st = (State*)buf;
    for (int i = 0; i < n_envs; ++i) reset_env(&st[i], seed0 + (unsigned long long)i * 7919ULL);
}

EXPORT void sim_set_cfg_double(int idx, double v) {
    /* byte-offset addressed for the benchmark's convenience; see bench.py */
    double* p = (double*)((char*)&G + idx);
    *p = v;
}

/* The benchmarked entry point.
 * actions: n_steps * n_envs int32, step-major (row t = actions for all envs at step t).
 * Auto-resets a dead env (vec-env semantics), matching the Python rollout_core.
 * Returns the number of episode terminations (a cheap way to prove work happened). */
EXPORT int sim_step_many(void* buf, int n_envs, int n_steps, const int* actions) {
    State* st = (State*)buf;
    int deaths = 0;
    const double dt = G.step_dt;
    for (int t = 0; t < n_steps; ++t) {
        const int* row = actions + (size_t)t * (size_t)n_envs;
        for (int e = 0; e < n_envs; ++e) {
            State* s = &st[e];
            if (s->is_dead) {
                s->last_ep_distance = s->distance;
                s->episodes++;
                reset_env(s, 0);
                deaths++;
            }
            apply_action(s, row[e]);
            advance(s, dt);
            s->steps++;
        }
    }
    return deaths;
}

/* ---------------------------------------------------------------------------
 * REACTIVE DRIVER (benchmark scaffolding, NOT part of the sim).
 * A fixed action stream kills an episode every ~5 steps (a turn on a straight is
 * fatal), which makes the benchmark reset-dominated and unrepresentative of
 * training, where episodes run hundreds of steps and a cell crossing (-> track
 * generation) happens every ~3-6 steps.  This tiny rule-based driver reproduces
 * the REAL workload mix.  It is mirrored exactly in bench.py so both sides pay
 * the same conceptual cost; its own cost is reported separately.
 * -------------------------------------------------------------------------*/
static int oracle_action(State* s) {
    Cell* c = cell_at(s, s->cell_index);
    Cell* n = cell_at(s, s->cell_index + 1);
    double cp = cell_progress(s);
    if ((c->flags & F_IS_TURN) && (c->flags & (F_CAN_L | F_CAN_R))
            && cp >= 0.20 && cp <= 0.50)
        return (c->flags & F_CAN_L) ? A_TURN_L : A_TURN_R;
    if (must_jump(n->obst) && cp > 0.55) return A_JUMP;
    if (must_slide(n->obst) && cp > 0.55) return A_SLIDE;
    if (n->flags & F_NARROW) {
        double tgt = (double)n->narrow_lane * G.narrow_lane_center;
        if (s->lane < tgt - 0.4) return A_STRAFE_R;
        if (s->lane > tgt + 0.4) return A_STRAFE_L;
        return A_NOOP;
    }
    if (c->flags & F_NARROW) {
        double tgt = (double)c->narrow_lane * G.narrow_lane_center;
        if (s->lane < tgt - 0.4) return A_STRAFE_R;
        if (s->lane > tgt + 0.4) return A_STRAFE_L;
        return A_NOOP;
    }
    return A_NOOP;
}

/* Same loop as sim_step_many but the action comes from oracle_action(). */
EXPORT int sim_step_many_oracle(void* buf, int n_envs, int n_steps) {
    State* st = (State*)buf;
    int deaths = 0;
    const double dt = G.step_dt;
    for (int t = 0; t < n_steps; ++t) {
        for (int e = 0; e < n_envs; ++e) {
            State* s = &st[e];
            if (s->is_dead) {
                s->last_ep_distance = s->distance; s->episodes++;
                reset_env(s, 0); deaths++;
            }
            apply_action(s, oracle_action(s));
            advance(s, dt);
            s->steps++;
        }
    }
    return deaths;
}

/* ---------------------------------------------------------------------------
 * REPRESENTATIVE OBSERVATION BUILDER (obs.py state_to_obs, 65-wide).
 * Not field-exact -- it exists only to price what porting the OTHER 59% of a
 * full env.step() would cost in C, since obs.state_to_obs is 52% of the Python
 * step's self-time.  Same shape: ~23 booleans off the current/next/next2 cells,
 * ~12 clipped/normalised scalars, then a K=4 one-hot action history.
 * -------------------------------------------------------------------------*/
static inline float clip01(double x) { return (float)(x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x)); }

static void build_obs(State* s, float* o, const int* hist4) {
    Cell* c  = cell_at(s, s->cell_index);
    Cell* n1 = cell_at(s, s->cell_index + 1);
    Cell* n2 = cell_at(s, s->cell_index + 2);
    int i = 0;
    o[i++] = (float)(s->is_dead != 0);
    o[i++] = (float)(s->jump_t > 0.0);
    o[i++] = (float)(s->slide_t > 0.0);
    o[i++] = (float)(s->stumble_t > 0.0);
    o[i++] = (float)((c->flags & F_IS_TURN) != 0);
    o[i++] = (float)((c->flags & F_CAN_L) != 0);
    o[i++] = (float)((c->flags & F_CAN_R) != 0);
    o[i++] = (float)must_jump(c->obst);
    o[i++] = (float)must_slide(c->obst);
    o[i++] = (float)((c->flags & F_NARROW) != 0);
    o[i++] = (float)((c->flags & F_WIDE) != 0);
    o[i++] = (float)((n1->flags & F_IS_TURN) != 0);
    o[i++] = (float)((n1->flags & F_CAN_L) != 0);
    o[i++] = (float)((n1->flags & F_CAN_R) != 0);
    o[i++] = (float)must_jump(n1->obst);
    o[i++] = (float)must_slide(n1->obst);
    o[i++] = (float)((n1->flags & F_NARROW) != 0);
    o[i++] = (float)((n1->flags & F_WIDE) != 0);
    o[i++] = (float)must_jump(n2->obst);
    o[i++] = (float)must_slide(n2->obst);
    o[i++] = (float)((n2->flags & F_IS_TURN) != 0);
    o[i++] = (float)(n1->level > c->level);
    o[i++] = (float)(s->gap_grace_cell >= 0);
    /* scalars */
    o[i++] = clip01((s->velocity - 90.0) / (215.0 - 90.0));
    int pt = c->type;
    o[i++] = (float)(pt <= 2 ? 0.16 : (pt <= 7 ? 0.33 : (pt <= 16 ? 0.50 : (pt <= 30 ? 0.66 : 0.83))));
    o[i++] = clip01((double)c->level / 3.0);
    o[i++] = clip01((double)s->facing / 4.0);
    double cp = cell_progress(s);
    o[i++] = clip01(cp);
    o[i++] = clip01((s->lane + G.lane_edge) / (2.0 * G.lane_edge));
    double ntl = (n1->flags & F_NARROW) ? (double)n1->narrow_lane * G.narrow_lane_center : 0.0;
    o[i++] = clip01((ntl + G.lane_edge) / (2.0 * G.lane_edge));
    o[i++] = clip01(cp + 0.0023 * s->velocity);                      /* predicted_landing */
    double span = G.wide_gap_span_a * s->velocity + G.wide_gap_span_b;
    o[i++] = clip01(span / 4.0);                                     /* predicted_jump */
    o[i++] = clip01((double)n1->level / 3.0);
    o[i++] = clip01((double)n2->level / 3.0);
    /* landing_hazard (v90): 1.0 if the arc lands on a must-jump cell */
    int land = (int)(cp + span);
    Cell* lc = cell_at(s, s->cell_index + (land < 0 ? 0 : (land > 3 ? 3 : land)));
    o[i++] = (float)(land > 3 ? 0.5 : (must_jump(lc->obst) ? 1.0 : 0.0));
    while (i < 37) o[i++] = 0.0f;
    /* K=4 one-hot action history (7 each) => 37 + 28 = 65 */
    for (int k = 0; k < 4; ++k) {
        for (int a = 0; a < 7; ++a) o[i++] = (float)(hist4[k] == a);
    }
}

/* Physics + obs build + a distance-delta reward: the closest this benchmark gets
 * to a FULL env.step().  obs_out is n_envs*65 float32, overwritten every step. */
EXPORT int sim_step_many_oracle_obs(void* buf, int n_envs, int n_steps,
                                    float* obs_out, float* rew_out) {
    State* st = (State*)buf;
    int deaths = 0;
    const double dt = G.step_dt;
    static int hist[4] = {6, 6, 6, 6};
    for (int t = 0; t < n_steps; ++t) {
        for (int e = 0; e < n_envs; ++e) {
            State* s = &st[e];
            if (s->is_dead) {
                s->last_ep_distance = s->distance; s->episodes++;
                reset_env(s, 0); deaths++;
            }
            double d0 = s->distance;
            int a = oracle_action(s);
            apply_action(s, a);
            advance(s, dt);
            hist[3] = hist[2]; hist[2] = hist[1]; hist[1] = hist[0]; hist[0] = a;
            build_obs(s, obs_out + (size_t)e * 65, hist);
            rew_out[e] = (float)((s->distance - d0) * 0.1 - (s->is_dead ? 1.0 : 0.0));
            s->steps++;
        }
    }
    return deaths;
}

/* Oracle ONLY (no physics) -- lets the benchmark subtract the driver's cost. */
EXPORT int sim_oracle_only(void* buf, int n_envs, int n_steps) {
    State* st = (State*)buf;
    int acc = 0;
    for (int t = 0; t < n_steps; ++t)
        for (int e = 0; e < n_envs; ++e) acc += oracle_action(&st[e]);
    return acc;
}

/* ---- read-back helpers (opaque struct on the Python side) ---- */
EXPORT double sim_get_distance(void* buf, int i) { return ((State*)buf)[i].distance; }
EXPORT double sim_get_last_ep_distance(void* buf, int i) { return ((State*)buf)[i].last_ep_distance; }
EXPORT double sim_get_velocity(void* buf, int i) { return ((State*)buf)[i].velocity; }
EXPORT double sim_get_lane(void* buf, int i) { return ((State*)buf)[i].lane; }
EXPORT int    sim_get_cell_index(void* buf, int i) { return ((State*)buf)[i].cell_index; }
EXPORT int    sim_get_episodes(void* buf, int i) { return ((State*)buf)[i].episodes; }
EXPORT int    sim_get_death_type(void* buf, int i) { return ((State*)buf)[i].death_type; }
EXPORT int    sim_get_is_dead(void* buf, int i) { return ((State*)buf)[i].is_dead; }
EXPORT int    sim_get_zone_level(void* buf, int i) { return ((State*)buf)[i].zone_level; }
EXPORT double sim_get_score(void* buf, int i) { return ((State*)buf)[i].score; }

/* single-env single-step, for the correctness trace */
EXPORT void sim_step_one(void* buf, int i, int action) {
    State* s = &((State*)buf)[i];
    if (!s->is_dead) { apply_action(s, action); advance(s, G.step_dt); s->steps++; }
}
EXPORT void sim_reset_one(void* buf, int i, unsigned long long seed) {
    reset_env(&((State*)buf)[i], seed);
}
