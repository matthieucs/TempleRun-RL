"""``TempleRunSim-v0`` — a fast, headless, pure-python Temple Run model.

A simulator that reproduces the *real* env's observation / reward / action contract
(``TempleRunEnv``) so a policy trained here transfers to the phone (and vice-versa).
No phone, no Frida, no sockets, **no sleeps** — thousands of steps/sec.

What it models (per ``docs/SIM_MODEL_SPEC.md``):
  - the track as a linked sequence of ``PathElement``-like cells (type / level / obstacle /
    junction info), generated procedurally per ``config`` OR replayed from an injected list;
  - the runner physics (velocity 100->210 accel, 3 lanes + strafe, jump/slide/strafe/turn as
    timed flag transitions);
  - collision/death (un-actioned obstacle at the player's cell, or a junction passed without a
    valid turn -> Fall);
  - scoring (~ velocity * dt -> reproduces the ~+27/step-at-start trend).

It emits the SAME 22-field raw-state dict (``constants.STATE_FIELDS``) the Frida hook produces,
turns it into an obs via ``obs.state_to_obs`` (reused verbatim), and computes the SAME reward
(score-delta * reward_scale + the shared shaping in ``reward.py``, minus death_penalty).

Two ``track_source`` modes (SIM_CONTRACT §2):
  - ``"procedural"`` — RNG-generate the track (training).
  - an injected ``list[dict]`` — REPLAY mode: consume the provided elements as the player
    advances (fidelity harness). Each dict: ``{type,level,obstacles,is_turn,can_turn_left,
    can_turn_right}`` (extra keys ignored; missing keys defaulted).

All inferred numeric knobs live in ``DEFAULT_SIM_CONFIG`` below (one place to calibrate later).

Run the smoke test:  ``python -m templerun_rl.sim_env``  (or ``python src/templerun_rl/sim_env.py``).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from . import constants as C
from . import obs as O
from . import reward as R

# CONTINUOUS, TILT-CONTROLLED lateral model (device-measured 2026-07-01, authoritative): the lateral
# position is a continuous tilt offset centered on 0, range ~[-LANE_EDGE, +LANE_EDGE]. A STRAFE is a
# tilt NUDGE toward one side (a short glide of one LANE_STEP); when steering stops the position
# RECENTERS to 0 over ~lane_recenter_time. Hitting a side WALL (over-tilting past LANE_EDGE) does NOT
# kill — it STUMBLES (velocity *= stumble_slowdown, recenter, survivable once); a 2nd stumble within
# stumble_kill_window kills. This replaces the old persistent lock-lane target + instant off-edge Fall.
LANE_STEP = O.LANE_STEP        # ~3.7, the lateral distance one strafe (tilt nudge) covers
LANE_EDGE = 7.0               # the side wall (over-tilting past it -> stumble); config: cfg["lane_edge"]


# =============================================================================
# DEFAULT_SIM_CONFIG — every INFERRED numeric knob, in one labeled place.
# [KNOWN] = verbatim from dump.cs / SIM_MODEL_SPEC. [INFERRED] = needs calibration.
# Group A: velocity/physics. Group B: track generation. Group C: scoring. Group D: timings.
# =============================================================================
DEFAULT_SIM_CONFIG: dict[str, Any] = {
    # --- A. Velocity / pacing ------------------------------------------------
    "start_velocity": 100.0,          # [KNOWN] SRV (CharacterPlayer.StartRunVelocity)
    "max_velocity": 210.0,            # [KNOWN] MRV (CharacterPlayer.MaxRunVelocity)
    "accel_per_sec": 6.0,             # [INFERRED] linear fallback if velocity_curve is empty
    # velocity = curve(distance) — the real AccCurve is distance-gated, not linear. Measured from
    # human runs (median velocity per distance bin): rises 100->210 over ~4k distance, then plateaus.
    "velocity_curve": [               # [CALIBRATED] (DistanceTraveled, velocity) median over real runs
        [5, 100.0], [10, 110.9], [30, 125.7], [50, 137.1], [70, 138.6], [90, 139.8], [110, 141.3],
        [130, 142.6], [150, 144.1], [170, 145.5], [190, 146.3], [225, 148.7], [275, 151.6],
        [325, 154.3], [375, 157.6], [425, 159.7], [475, 162.6], [525, 164.9], [575, 167.3],
        [750, 174.0], [1050, 180.7], [1350, 184.5], [1650, 187.5], [1950, 190.6], [2250, 194.0],
        [2550, 196.8], [2850, 200.0], [3150, 202.7], [3450, 205.5], [3750, 207.4], [4200, 210.0],
        [6750, 210.0],
    ],
    "stumble_slowdown": 0.85,         # [KNOWN] StumbleSlowDown — velocity *= this on a stumble
    "stumble_recover_rate": 1.0,      # [MEASURED] u/s the velocity re-climbs after a stumble (device
                                      # 0.5-1.1 across eval_stumble/human captures; NOT a curve snap-back)
    # --- LATERAL / stumble (device-measured 2026-07-01) --------------------------------------------
    "lane_edge": 7.0,                 # [MEASURED] side wall: |lane|>this over-tilts -> stumble (not a Fall)
    # LATERAL HOLD (device-eval 2026-07-01, CORRECTED): the lateral offset does NOT snap back to center.
    # A device capture of a policy showed the runner stuck at lane ~+4.7 through 1.4 s of NOOPs (no drift),
    # then turning from there threw it off the path -> Fall. The OLD fast recenter (0.4 s) centered the
    # policy for FREE, so it never learned to steer back to center before a turn (the dominant device
    # death). Model the HOLD: the lane recenters only very slowly, so the policy must ACTIVELY strafe back
    # to center (like the human, who turns at lane ~0.2). See [[lateral-control-recenter-mechanic]] (the
    # earlier "recenters fast" claim is superseded by this device eval).
    # [DEVICE-MEASURED 2026-07-01] during NOOP frames the device lane barely moves (median Δ|lane|
    # +0.004/frame; 317/430 frames HOLD, only 30 recenter) -> it essentially HOLDS. Model a near-total
    # hold (100 s to drift one edge = ~0.007/frame) so the policy must ACTIVELY strafe back to center.
    # The old 0.4 s and the memory's "recenters over 0.5 s" were both wrong (measured the strafe anim end).
    "lane_recenter_time": 100.0,      # [DEAD 2026-07-02] persistent-px era; see lat_tau (analog tilt)
    # ANALOG-TILT lateral (lateral-accel-15a, device-MEASURED strafe_probe_tilt2 2026-07-02): the lane
    # first-order-lags toward the commanded target (hold +/-edge, NoOp 0); alpha 0.446 per 0.1 s frame
    # both directions -> tau = -0.1/ln(1-0.446) ~ 0.169 s. Full reach ~0.5 s; recenter is the mirror.
    "lat_tau": 0.169,                 # [MEASURED] first-order time constant of the lane lag (seconds)
    # OFF-CENTER TURN death: turning while laterally off-center rotates the offset off the exit path ->
    # Fall. Human turns at lane med 0.22 (p90 4.42) and survives; the policy turned at 4.7 and fell. So a
    # turn executed with |lane| > this throws the runner off the path -> Fall. Forces "centre before you turn".
    "turn_center_tol": 4.0,
    "stumble_kill_window": 2.0,       # [MEASURED] a 2nd stumble within this many seconds of the last -> death
    # --- B. Track generation (PathElement statics) — values READ off the device (Frida, baseline
    # Temple/level-2 section; the game ramps these by distance via section/level changes). -------
    "compound_prob": 0.0,               # [2026-07-06 USER-OBSERVED blindspot] P(a narrow placement becomes
                                        # a STACKED compound: narrow(side A) -> jump-barrier -> narrow(side
                                        # B, OPPOSITE). Survivable only by a MID-AIR STRAFE (jump the
                                        # barrier while sliding A->B). The sim never stacked obstacles
                                        # (min_space 6) so the policy never learned it. 0 default (scorecard
                                        # / existing behaviour unchanged); train ~0.15. Device: real + "not
                                        # uncommon" (user). Bypasses min_space for the sequence's own cells.
    "compound_after_distance": 300.0,   # only compound past this (early run stays simple)
    "compound_land_cells": 2,           # clear cells after the barrier before narrow B (best oracle
                                        # survival 16-17/40 vs 12 @1 / 11 @3 — 2 gives room for the
                                        # A->B mid-air/pre-swing to land on strip B)
    # JUMP->SLIDE compound [DEVICE DEATH 2026-07-07, v51 dt=3 @4991/3916]: a SLIDE obstacle placed a
    # few clear cells after a WIDE-GAP (crevasse/long-hole), WITHIN the jump's ~0.8s airtime -> the
    # crevasse jump lands the runner still AIRBORNE at the slide -> can't slide -> dt=3/5. The sim never
    # stacked jump->slide (min_space forces a clear gap after every wide-gap) so v51 never learned the
    # jump-early-land-then-slide timing (offline death mix = 0 dt=3). Model it as a post-gap tail that
    # bypasses min_space. 0 default (rng gated on prob>0 FIRST -> track stream / equivalence unchanged);
    # train ~0.15. The policy must jump the gap EARLY (land before the slide) + not re-jump.
    "jumpslide_prob": 0.0,              # P(a wide-gap is followed by a close slide obstacle)
    "jumpslide_after_distance": 300.0,  # only past this (slides need slides_after_distance anyway)
    "jumpslide_gap_cells": 1,           # clear cells between the wide-gap end and the slide (tighter = harder)
    "jumpslide_flame_frac": 0.5,        # fraction of jump->slide tails that are FlameTower(6) vs TreeSlide(5)
    # JUMP->STRAFE compound [DEVICE DEATH 2026-07-07, USER-observed + v51/v53 narrow deaths @5148/3281/2793]:
    # a SIDE-NARROW (needs lane +/-1) placed a few clear cells after a WIDE-GAP -> the runner must STRAFE
    # into the narrow's lane DURING/right after the jump. The policy commits the strafe too LATE (cp~0.9)
    # -> the analog-tilt lateral can't reach the lane before landing -> falls off the narrow. The sim never
    # stacked jump->narrow (min_space) so the policy never learned to strafe EARLY after a jump. Same shape
    # as jumpslide: place a narrow tail after the wide-gap (bypasses min_space). 0 default; train ~0.15.
    "jumpnarrow_prob": 0.0,             # P(a wide-gap is followed by a close side-narrow strip)
    "jumpnarrow_after_distance": 300.0, # only past this
    "jumpnarrow_gap_cells": 1,          # clear cells between the wide-gap end and the narrow
    "jumpnarrow_strip_cells": 2,        # narrow strip length (fixed, matched CPU<->torch for the tail)
    # GAP->GAP compound [DEVICE DEATH 2026-07-08, USER-observed high-ground DOUBLE-JUMP + rejump_lethality
    # forensic]: on device, holes sit 2-3 cells apart in high-ground zones -> a blind landing-frame RE-JUMP
    # after gap-1 lands INSIDE gap-2 (dt4 Ledge; 35% of device eps end within 15 frames of a re-jump). The
    # sim's min_space 6 made the re-jump structurally harmless (0/47 lethality -> no learning signal; and
    # 2 reward-shaping attempts v60/v61 failed: gamed by gaps / gate let the killer through). Same shape as
    # jumpslide: place a 2-cell HoleLong tail after a wide-gap (bypasses min_space). Blind re-jump dies,
    # land->run->deliberate-jump-at-the-lip clears. 0 default; train ~0.15.
    "jumpgap_prob": 0.0,                # P(a wide-gap is followed by a close 2nd hole)
    "jumpgap_after_distance": 300.0,    # only past this
    "jumpgap_gap_cells": 2,             # clear cells between the wide-gap end and the 2nd hole
    "jumpgap_hole_cells": 2,            # width of the 2nd hole (2 = HoleLong pt12, landing model applies)
    "jumpgap_chain_max": 1,             # RHYTHM CHAIN: repeat [clear,hole] up to n times (uniform 1..max);
                                        # 1 = single tail, no extra rng (v62-compatible). Device: high-ground
                                        # zones chain long-holes every ~4 cells (ab_v62_r1 ep0, ab_v60_r2 ep4)
    # SLIDE->HOLE compound [WAVE 9 HUMAN-DEMO 2026-07-09, `experiments/slidegap_survival.py`]: a HOLE a couple
    # clear cells after a SLIDE obstacle. The slide COMMITS ~0.73s (device-measured; sim slide_duration 0.7 +
    # jump-while-sliding already refused, sim_env.py _apply_action) — if the hole arrives inside the commit
    # the jump window is EATEN. Killed the HUMAN twice (ep2 @3604 / ep3 @686: sliding when the gap arrived,
    # jump swipe refused mid-fall); human survives spacing >=1.1s via slide-end -> INSTANT jump (takeoff
    # -0.33..+0.07s rel. hole entry, 12/14 events). Generation NEVER stacks slide->hole (min_space 6;
    # jumpslide is the REVERSE order) -> free in-sim -> no learning signal (the 6th such gap). The trainable
    # skill = EARLY slide (commit ends sooner) + instant jump at slide end. 0 default; train ~0.12.
    "slidegap_prob": 0.0,               # P(a slide obstacle is followed by a close hole)
    "slidegap_after_distance": 300.0,   # only past this (slides need slides_after_distance anyway)
    "slidegap_gap_cells": 2,            # clear cells between the slide and the hole (2 ~= the human's ~1.2s limit)
    "slidegap_hole_cells": 1,           # hole width (1 = single pt11 hole, the device-observed form)
    # POST-TURN OBSTACLE [DEVICE-MEASURED 2026-07-08]: the game places obstacles on the FIRST cell after a
    # junction (72 onsets ~1 cell post-turn across 6 captures; bimodal with the normal bulk at 2.5-5 cells);
    # min_clear_after_turn(2) meant the sim NEVER did -> post-turn deaths end the 4-7k runs (7/15 of the
    # pt(-1/1/2) deaths follow a turn by <12 frames). Standard obstacle sampler (holes stay 1-cell). 0=off.
    "turnobstacle_prob": 0.0,           # P(a standard obstacle on the first post-junction cell)
    "turnobstacle3_prob": 0.0,          # P(one at post-junction cell +3) — the DEVICE PEAK, see below
    "min_space_between_obstacles": 6,   # [READ] sMinSpaceBetweenObstacles (cells)
    "min_space_between_turns": 12,      # [READ] sMinSpaceBetweenTurns (cells)
    # ★★★ THE GAME'S OWN DIFFICULTY RAMP [READ LIVE OFF THE DEVICE 2026-07-26, hook 17j `s.gen`:
    # PathElement statics via klass+0xB8 from a live instance, sampled every ~2s over a 10,768 m run].
    # This is an EXACT distance-keyed schedule, not a fitted distribution — implement it, do not
    # calibrate it. The sim previously held min_space_between_obstacles at a CONSTANT 6 for the whole
    # run while the real game ramps it 6 -> 2, so past ~4.5 km the device packs obstacles THREE TIMES
    # tighter than anything the policy ever trained on. That single fact accounts for: the device's
    # tight pairs, the 17 hazard-pair types with ZERO sim coverage (incl. every "-> Crevasse"
    # compound), the sim being ~1.5x too easy (v78: 6786 sim vs 4425 device), and the user-reported
    # "succession of obstacles" death. sMaxBackToBackObstacles is 2 THROUGHOUT (training used 6 = wrong).
    # Rows: (from_distance_m, min_space_obstacles, min_space_turns, double_obstacle_pct, narrows, flame_slide)
    "device_difficulty_ramp": False,    # OFF = the historical flat behaviour (equivalence baseline)
    "difficulty_ramp_table": (
        (0.0,    6, 12, 0.00, False, False),
        (250.0,  5, 10, 0.00, False, False),
        (500.0,  4,  8, 0.05, False, True),
        (1100.0, 4,  6, 0.05, False, True),
        (1500.0, 3,  5, 0.10, True,  True),
        (2000.0, 3,  4, 0.15, True,  True),
        (3250.0, 3,  3, 0.17, True,  True),
        (4500.0, 2,  2, 0.20, True,  True),
    ),
    # sDoubleObstaclePercent (0 -> 0.20 with distance): P(an emitted obstacle is immediately followed
    # by a SECOND one on the very next cell) — capped by max_back_to_back_obstacles(2). NOT previously
    # modelled at all; this is the mechanism behind the device's gap-1 hazard pairs.
    "double_obstacle_percent": 0.0,
    "min_spaces_between_bonus": 50,     # [READ] sMinSpacesBetweenBonusItems
    "obstacle_prob": 0.52,             # [AUDIT 2026-06-26] denser to match real obstacle-frame rate (was 0.45, ~under)
    "double_obstacle_percent": 0.0,     # [READ] sDoubleObstaclePercent (no doubles in the baseline section)
    "max_back_to_back_obstacles": 2,    # [READ] sMaxBackToBackObstacles
    "turn_prob": 0.18,                 # [INFERRED] chance a turn-eligible cell becomes a junction (early game)
    # Difficulty RAMP: the real game densifies turns with distance (measured turns/1k: ~8.5 at the
    # start -> ~21 by distance 5-6k). difficulty = clip((distance-start)/ramp_distance, 0, 1) lerps
    # the turn spacing down + turn prob up; slides only switch on past slides_after_distance (the
    # baseline statics have sAllowTreeSlide/FlameTowers = False -> no slides early).
    "ramp_distance": 6000.0,           # [CALIBRATED] distance over which difficulty goes 0 -> 1
    "min_space_between_turns_late": 5, # [AUDIT 2026-06-26] denser late turns to match real ~6.9%/frame (was 7)
    "turn_prob_late": 0.5,             # [CALIBRATED] turn probability at full difficulty
    # Clear run-out after a junction: the cell immediately after a turn carries NO obstacle. You can't
    # turn AND have a jump/slide active on the same cell-crossing, so a must-jump cell right after a
    # junction is unclearable (turn -> enter obstacle with no jump -> die). Real turn-pieces have a
    # straight run-out too. >=2 -> the player turns, gets 1 clean cell, then the next may be an obstacle
    # (visible via the lookahead while on the clean cell).
    "min_clear_after_turn": 2,         # [CALIBRATED] cells after a junction kept obstacle-free
    # TURN EXECUTION [DEVICE-MEASURED 2026-07-02, scripts/probes/probe_turn_exec.py — one swipe at controlled
    # cprog, 40 turns]: the turn executes INSTANTLY (facing flips next frame; the game's HandleDelayed-
    # Turning buffer is NOT used) and the runner LANDS at a lateral offset ~ (swipe_cprog - CENTER) *
    # SCALE. Swipe at cprog 0.44 -> land centered (lane ~0); swipe at the is_turn flash (0.15-0.23) ->
    # land 8-12 off -> water; swipe late (>=0.61) -> land 11-29 off -> water. Survivable swipe window
    # ~[0.28, 0.60]. The runner then RECENTERS via the analog tilt. This is THE turn mechanic (replaces
    # the old "any cprep in [0,0.45] lands centered", which let the policy swipe at the flash and die).
    "turn_exec_early": 0.15,           # is_turn PRESENTED + a swipe executes from this cprog (the flash)
    "turn_exec_late": 0.60,            # ...until this cprog (past it = ran too far, swipe misses)
    "turn_center_cprog": 0.44,         # swipe here -> land centered (measured offset minimum)
    "turn_offset_scale": 45.0,         # landed |lane| = |swipe_cprog - center| * this (measured slope)
    "turn_lane_carry_buffered": 1.0,   # [2026-07-06 DEVICE-MEASURED] carry for BUFFERED (swipe-queue)
                                       # turns: native corner landing is forgiving (v14: lane<=4.6 swipes,
                                       # 0 landing deaths in 212). Keeps a mild centering preference
                                       # without teaching the fatal recenter-then-swipe hesitation.
    "turn_lane_carry": 3.0,            # [MEASURED 2026-07-02] pre-turn lane AMPLIFICATION through a turn
                                       # (device: strafe-into-turn lane 5 -> lands ~18). Punishes leaning
                                       # into turns -> the policy must center (NoOp) before swiping.
                                       # DO NOT LOWER. 2026-07-03: "fidelity" carry=0.25 (99% replay
                                       # agreement) was DEVICE-REFUTED (v17 median 308, turns landed 15-22).
                                       # carry=3 is a LOAD-BEARING REGULARIZER that keeps the policy centered
                                       # AND times turns well; it is not pure physics. See v14-ceiling memory.
    "turn_land_kill": 9.0,             # [MEASURED] land beyond this |offset| -> off the exit path ->
                                       # water. The runner tolerates offsets up to ~9 (recenters via
                                       # tilt); probe: offset 8-10 survived, >=11 (swipe <=0.14 or
                                       # >=0.61) died. Survivable swipe window ~[0.24, 0.64].
    "turn_window_frac": 0.60,          # [alias] is_turn presentation upper bound = turn_exec_late
    # MARGINAL-ZONE TAIL RISK [DEVICE-MEASURED 2026-07-04]: swipes in [cp_lo, cp_hi] get a probabilistic
    # catastrophic-landing kill ramping 0 -> pmax with cp (device death: cp0.50-0.58 ~8%, 0.58-0.64 ~18%),
    # even when the deterministic offset would survive. Forces the policy to aim EARLY (reliable cp<0.50)
    # instead of the sim-safe-but-device-fatal marginal zone. replay_diff sets pmax=0 (deterministic).
    "turn_tail_cp_lo": 0.50,
    "turn_tail_cp_hi": 0.64,
    "turn_tail_pmax": 0.0,             # DEFAULT OFF (2026-07-04): v19 showed v14 ALREADY aims early
                                       # (sim cp 0.30, 0% marginal) so the tail-risk never bites — the
                                       # real turn gap is the ~2-frame CONTROL LATENCY (stale obs +
                                       # delayed exec), not marginal-swipe choice. Kept for reference.
    "turn_exec_lat_frames": 0.0,       # CONTROL LATENCY on junction-cell swipes (default 0 = off). v20
                                       # sets ~2.0 to model the phone's ~2-frame lag -> junction swipes
                                       # LAND LATE (die at speed) -> forces the approach-buffer (immune).
    # APPROACH-BUFFER (flag, 2026-07-03, device probe_turn_window): the game's HandleDelayedTurning — a
    # swipe on the APPROACH cell (next cell is the junction) is HELD and fires CENTERED on entering the
    # junction (measured: an early approach-swipe flips facing ~4 frames later AT the junction). This is
    # the SAFE play the policy should learn (a late junction-cell swipe lands off-center via the 1-frame
    # lag). OFF by default (keeps the current behavior + scorecard). Distinct from the §14 EXPLOIT (which
    # fired the turn IMMEDIATELY on the approach -> off-path): here the swipe is LATCHED and fires at the
    # junction center. `turn_buffer_ttl` = cells the latch survives before timing out (no-op = the game's
    # TimeSinceShouldTurn timeout). Pre-turn lane STILL carries (turn_lane_carry) -> the buffer helps only
    # a CENTERED approach; leaning in still lands off (correct lesson: center, then approach-swipe).
    "turn_approach_buffer": False,     # gate the buffer mechanic (train_sim --turn-approach-buffer)
    "hook_turn_buffer": False,         # models hook 16j TURN_BUFFER=true: junction-cell turn fires
                                       # CENTERED (timing-free) — removes the 10Hz turn latency. v20 on.
    "turn_buffer_ttl": 3,              # cells a latched approach-swipe persists before timing out
    "turn_buffer_late_cp": 0.56,       # [2026-07-05 v3, DEVICE-MEASURED on 205 first-swipes across
                                       # v26/v27/v14_freshphone captures] junction FIRST swipe fires iff
                                       # observed cp <= ~0.56 at ALL velocities (fired max 0.55-0.56 in
                                       # every band incl v207; missed min 0.59 @v180+). This is the game's
                                       # CanTurn window end (hook guard drops the GI_ShouldTurn write when
                                       # native CanTurn is false), NOT a corner-arrival race: the earlier
                                       # velocity-dependent lag-2 arrival model predicted v210 junction
                                       # swipes ~impossible, but v14 turned 46/46 up to v207 on device
                                       # (5462m fresh-phone run). Second-frame swipes (~0.6 at v>150)
                                       # MISS -> v27's device hesitation deaths; first-frame swipes safe.
    # TURN-SEND LATENCY [DEVICE-MEASURED 2026-07-06, v47 junction/off deaths + USER-observed "action fired
    # but never launched"]: the turn_buffer_late_cp gate (0.56) is measured on the PROCESSING cp (the cp
    # when the game consumes the swipe), which is velocity-INDEPENDENT. But the policy DECIDES on the
    # OBSERVED cp, and the byte arrives ~this many frames LATER -> processing_cp = decision_cp +
    # velocity*lat*dt/cell. At HIGH speed that pushes an on-junction decision (e.g. v47 decided TL @cp0.36
    # @v210) past 0.56 -> DEAD-LETTER -> off-track death (v14 survived by deciding @cp~0.1). The sim gated
    # on the decision cp (no latency) -> let v47 fire on-junction turns that dead-letter on device. lat=1
    # (= the measured 1-frame control latency) reproduces it; the latency-robust play is a PRE-SWIPE on the
    # approach cell (turn_approach_buffer, immune). Default 0 = off (equivalence/scorecard unchanged).
    "turn_send_lat_frames": 0.0,
    # WAVE 10 (2026-07-09) — post-16o turn-byte fidelity pair (see _attempt_turn comments): the hook
    # DROPS off-junction turn bytes (measured drop paths 2026-07-05) instead of executing them, so the
    # 2026-07-02 turn-on-straight=FATAL rule is stale for the current pipeline. Both default OFF.
    "turn_offjunction_drop": False,     # off-junction turn (w/ hook_turn_buffer) = silent drop, not kill
    "turn_drop_debounce_s": 0.0,        # hook anti-double-turn window: a turn byte within this of the
                                        # previous one is EATEN (spam near a junction kills the real swipe)
    # WAVE 12 (2026-07-09): a landing-frame re-jump ON a wide-gap cell fires 46% on device (41/90,
    # landing_lock_forensic context split) — the hole-chain recovery affordance; the lip rule alone
    # refuses it 100% in-sim = unlearnable. Per-attempt draw at the measured rate. Default 0 = off.
    "post_land_gap_rejump_prob": 0.0,
    # ★★ HELD-JUMP -> LANDING RE-JUMP [WAVE 16, USER-reported ~50 times: "on the high-ground it jumps
    # for no reason just after having jumped over a crevasse"]. MEASURED (scripts/analysis/detect_patterns.py):
    # the policy holds JUMP through its whole arc 10.7 times per km; the HUMAN does it ZERO times in
    # 41 km -- they tap. In the sim, a JUMP pressed while airborne is simply DISCARDED by the airborne
    # guard, so holding the button is completely FREE and nothing ever teaches the policy to release
    # it. On DEVICE the still-held press fires a NEW jump the instant the runner touches down
    # (wave-12 landing_lock_forensic: 10% on solid ground, 28-46% on a gap cell) -- and on high ground
    # that second jump drops into the next crevasse/hole. That is the user's complaint, and it is a
    # SIM-FIDELITY gap, not a reward-shaping one (which is why the v58/v60/v61 penalties failed: they
    # taxed a behaviour the world never punished, and got gamed by 1-frame gaps).
    # With this on, holding JUMP has the same consequence it has on the phone, so releasing becomes
    # something the policy has a reason to learn. Gated prob>0-first (0 = old behaviour, stream safe).
    "held_jump_rejump_solid": 0.0,     # device-measured 0.10 on solid ground
    "held_jump_rejump_gap": 0.0,       # device-measured 0.28 on a wide-gap/crevasse cell
    # DEAD (kept only so old sidecars/configs referencing them don't KeyError): the off-centre turn
    # death is DELETED — device shows early turns are how humans play, NOT a trap. _attempt_turn no
    # longer reads these; turning at any cprog in the window (or buffered early) is clean.
    "turn_early_frac": 0.0,            # [DELETED 2026-07-01] no early-turn death (was 0.30 off-center trap)
    "turn_late_frac": 1.0,             # [DELETED 2026-07-01] no late-turn death (was 0.62 off-center trap)
    # BUFFERED EARLY SWIPE (device HandleDelayedTurning): a turn swiped on the approach cell (next_is_turn)
    # is latched and fires on entering the turn area. This is how the human turns reliably despite the
    # 1-frame is_turn flash. The buffer survives this many cell-crossings before expiring.
    "turn_buffer_cells": 2,
    "slides_after_distance": 300.0,    # [CALIBRATED] TreeSlide/FlameTower obstacles only past this distance
    "default_path_level": 2,           # [READ] sPathLevel = 2 (Temple) — the START zone (mid altitude)
    # ZONE / ALTITUDE TRANSITIONS (user-reported + captured live): the track runs through 3 altitude
    # zones (low/mid/high). A transition carries ObstacleType.LevelChange(7) — a CREVASSE that must be
    # JUMPED (captured: path_level 2->3 + obstacle=7 + no jump => fell). Model it: every ~N cells the
    # zone steps to an adjacent altitude and that cell is a must-jump LevelChange. path_level in the obs
    # follows the zone so the policy also sees the altitude.
    # [DEVICE-MEASURED 2026-06-26] crevasse frame-rate ~3.5% vs the old sim's 1.85% (2x too sparse);
    # first crevasse at dist 587. Halve the spacing to match the real density (audit-tuned).
    "zone_change_every_cells": 32,     # [MEASURED] cells between zone transitions (~2x denser to match real 3.5%)
    # [DEVICE-MEASURED] first crevasse at ~587 m (v≈145). Gating the sim to the same onset matters MORE now
    # that the apex check is device-tight (mh 0.95): below v≈140 the short arc physically cannot reach
    # near-apex at the ledge from the approach cell, so early low-speed sim crevasses would be unsurvivable
    # phantoms the device never poses (they were masked when mh was a lenient 0.45).
    "zone_after_distance": 550.0,      # zone crevasses only past this distance (device onset)
    # WIDE GAPS (2-cell): on device, long holes (path_type 12/13/15/16) and zone LevelChange caps
    # (33-36) span TWO cells — a single jump must be timed at the gap EDGE to span both; jumping a cell
    # early (the single-obstacle reflex) lands in the far half -> Fall. Modeled as 2 consecutive
    # must-jump cells + a `next_wide_gap` obs so the NN learns to jump late (via cell_progress).
    # NARROW pieces (path_type 17-30): a one-lane strip; the runner falls off unless in the strip's
    # safe lane. The sim teaches the SKILL "strafe to the lane the obs says" using a RANDOM safe lane
    # per narrow (so the policy MUST read `narrow_target_lane`, not memorize). The real subtype+facing
    # -> safe-lane map lives in the HOOK (from the device sweep); the policy is map-agnostic.
    # [DEVICE-MEASURED 2026-06-26] narrows appear ONLY past ~1800m (first at 1824) and at ~7% frame-rate
    # after that; the old sim spawned them from dist 0 at ~3.5% (wrong onset + 2x too sparse).
    "narrows_after_distance": 1600.0,  # [MEASURED] first narrows at 1605-1752 across human_calib_v2 eps
    "narrow_prob": 0.70,               # [CALIBRATED 2026-07-02] band 1500-3000 narrow%: sim ~8 vs real
                                       # 11.5 (scorecard tol ratio [0.6,1.6]); long-run real avg ~8%
    # NARROW strip geometry [MEASURED 2026-07-02, Region2D polygon sampling, human_narrows.jsonl]:
    # long-narrow safe band = [2,7]/[3,7] (or mirrored) -> strip CENTER ~ +/-4.5 (NOT the +/-3.7
    # coin-lane step), half-width ~2.0-2.5. Validated: every human on-strip death rode OUTSIDE its
    # band. NarrowSmall 17/18 is bimodal (two side strips) — modeled center-safe with the same tol.
    "narrow_lane_center": 4.5,         # [MEASURED] |lane| of a long-narrow strip center
    "narrow_lane_tol": 2.5,            # [MEASURED] half-width of the safe band around that center
    # DEVICE-MEASURED strip length (human capture 2026-06-26): {1:12, 2:16, 3:13, 4:6, 5:2, 7:1} cells.
    # A strip is multiple cells -> the runner must HOLD the safe lane across all of them (a per-cell
    # off-lane Fall), capped at 4 here. The OLD 1-cell model let the policy strafe back too early.
    "narrow_len_weights": {1: 0.25, 2: 0.34, 3: 0.27, 4: 0.14},
    # Strafing past the 3-lane edge drops the runner off the path on device (measured: lane -5.5 on a
    # straight -> Fall). The cap-only sim hid this, so the policy strafe-spammed off-path. Model it.
    "strafe_off_edge_kills": True,
    "turn_while_strafing_kills": True,  # a turn issued mid-strafe lands off-path (strafe-heavy policy fix)
    "wide_gap_prob": 0.40,             # chance a placed HOLE is a WIDE gap (jump-late-or-fall)
    # DEVICE-MEASURED gap spans (human capture 2026-06-26, captures/human_calib_2026-06-26.jsonl):
    # same-level long holes span 1-3 cells (model the wide ones as 2 = extra 1). The OLD value 0 made
    # every gap a single cell, so the policy never learned the 2-cell timing -> jumped early -> fell in.
    "wide_gap_extra_cells": 1,         # 1 => the gap is 2 cells (a single jump must span both)
    # Wide-gap death: a 2-cell gap is self-timing — the runner must be AIRBORNE at BOTH gap-cell entries,
    # and a single jump (airtime ~0.6s) only spans ~2 cells at the crevasse velocity (~210, top speed),
    # so an early jump runs out of air before the 2nd cell -> Fall, a late/no jump isn't airborne at the
    # 1st -> Fall. The span enforces the timing, so the entry check is just "still airborne" (need ~0).
    # (The old need=0.30 was a hack to fake timing on a SINGLE-cell gap; the real 2-cell gap does it.)
    "wide_gap_min_airtime": 0.0,
    # SIM-TO-REAL turn fidelity: on the phone is_turn stays True for ~2 frames AFTER a turn (the
    # facing-change lags the input), so the policy fires the turn on both frames and the 2nd
    # OVER-ROTATES the character off the corner -> Fall (the dominant on-device turn-death). The plain
    # sim drops is_turn instantly so the policy is never tempted -> never learns to resist. Set
    # turn_linger_frames>0 to FORCE is_turn True for N frames post-turn (a 2nd turn during them = death)
    # -> the policy must learn (via action-history memory) to turn ONCE. 0 = off (legacy instant-consume).
    "turn_linger_frames": 0,
    "zone_min_level": 1,               # low altitude
    "zone_max_level": 3,               # high altitude (start = default_path_level = 2 = mid)
    # DEVICE-MEASURED (human capture 2026-06-26): the crevasse gap WIDTH depends on the altitude pair.
    # mid<->high (2<->3) transitions are 2-3 cell gaps (the wide ones the policy jumps too early on);
    # low<->mid (1<->2) are ~1 cell (forgiving). So a transition touching HIGH gets the wide 2-cell gap.
    "zone_gap_extra_cells_high": 1,    # mid<->high transition => 2-cell gap (extra 1)
    "zone_gap_extra_cells_lowmid": 0,  # low<->mid transition => single cell (extra 0)
    # [device-eval 2026-07-01] STEP-UP (level increases, e.g. mid->high 2->3) jump TIMING: a level-UP
    # transition raises the far ground into a WALL — jumping too early lands the runner ONTO/INTO that
    # raised wall (device: mid->high cleared but the policy jumps slightly too early -> hits the wall ->
    # slows/dies). Model it: on a level-UP crevasse the runner must have TAKEN OFF late enough — after
    # this fraction of the pre-gap cell — else the (still-airborne) early jump smacks the wall and FAILS.
    # An early takeoff (the 1-cell-lookahead reflex, or a jump 2 cells before the gap) => Fall; taking
    # off near the gap edge (as the oracle does, cprog>=0.62) clears. Level-DOWN + same-level long holes
    # are UNAFFECTED (only cells flagged stepup are checked). Keep below the oracle's 0.62 jump so it's fair.
    # [DEVICE-FIT 2026-07-05, v26 post-mortem] 0.45 was FAR too lenient on the EARLY side: device takeoffs
    # at v165-190 SURVIVE only at approach-cp 0.61-0.98 (14 clears, levelprobe_16q) and DIE at cp<=0.57
    # (5 v26 Ledge deaths) -> the device demands ~apex (within ~10% of peak) at the ledge. mh=0.95 is the
    # unique fit (covers all survivors, excludes all deaths; @v210 window = 1.8 frames -> the late-game
    # wall). The old 0.45 let the sim reward cp~0.4 takeoffs that the device kills -> v26 learned them.
    "stepup_min_height": 0.95,         # [VELOCITY-DEP APEX model 2026-07-03, human_calib_v2 v210 probe] a
                                       # step-UP crevasse survives only if the jump HEIGHT (parabola, 1 at
                                       # apex) at the raised ledge >= this — auto velocity-dependent (span
                                       # = jump_dur*vel/cell_len). 0.45 -> survive ~[0.13,0.87] of the span.
    # NON-FATAL CLIP [USER-OBSERVED + DEVICE-MEASURED 2026-07-06, 15 events / 4 captures]: a MARGINAL
    # step-up clip does NOT kill — the runner grazes the ledge and STUMBLES: velocity *= 0.85 in ONE
    # frame (exactly StumbleSlowDown; 15/15 airborne, 14/15 on pt33-36/long-holes, recovery ~+1.4 u/s =
    # stumble_recover_rate; ~5 events per 8 device eps). The binary clear-or-die model missed it, and the
    # slowdown RE-TIMES everything after (the low-speed hole-death cluster follows it). Model: apex
    # height in [stepup_min_height - band, stepup_min_height) -> _stumble() instead of Kill(2).
    # 0 = OFF (scorecard/replay/reference behavior unchanged); train ~0.25 (calibrated vs device rate).
    "stepup_clip_band": 0.0,
    "stepup_min_takeoff_prog": 0.85,   # [DEAD — replaced by stepup_min_height apex model] kept for compat
    "gap_entry_grace": 0.2,            # [MEASURED 2026-07-02] solid lip at a wide-gap cell's entry:
                                       # grounded entry survives until this cprog (device takeoffs
                                       # 0.98-1.02 cleared); still grounded past it -> Fall
    # WIDE-GAP LANDING MODEL [DEVICE-MEASURED 2026-07-06, jump_arc.py over 8 captures / 1600 jumps].
    # The old wide-gap check was "still airborne at each cell entry" — which a SHORT arc satisfies
    # while landing INSIDE the far gap cell (device: clips the far wall -> dt4 Ledge). Device 2-cell
    # holes clip 21% (18/87), concentrated at LOW speed where the arc barely spans 2 cells. Fix: the
    # jump must LAND past the far edge. Only the LAST gap cell is checked (next cell solid). OFF by
    # default (no-op -> equivalence stays green); enabled in the device-true / training config.
    # ⚠️ 2026-07-27: the old default (0.0204 / -0.94) is WRONG and was too LENIENT — an over-long span
    # judges a jump to clear the far edge when it actually falls short, under-punishing exactly the
    # "my arc landed me somewhere fatal" class that is 60% of device deaths. The claim it rested on
    # ("the sim's v/75 has the wrong slope") is refuted: measured with ONE estimator on both sides,
    # HUMAN 30 Hz gives 0.0143*v - 0.11 and the SIM gives 0.0147*v + 0.06 -- the sim's arc was right,
    # the fit was built on 10 Hz captures whose 0.1 s quantisation inflates the slope with velocity.
    # See the obs.py LAND_SPAN_A note; keep these two in step with it.
    "wide_gap_landing_model": False,
    "wide_gap_span_a": 0.0143,         # [DEVICE-MEASURED, human 30 Hz n=428] cells of span per velocity
    "wide_gap_span_b": -0.11,          # span intercept (cells)
    # SINGLE-HOLE AIRBORNE TRAP [WAVE 14, DEVICE-MEASURED 2026-07-25,
    # scripts/analysis/hole_landing_forensic.py over 1065 hole crossings / 4 policies]. A 1-cell hole is NOT
    # wide_gap, so the wide-gap landing model above skips it and the ONLY test is binary `_jump_t > 0`
    # -> an almost-expired arc crosses a hole FOR FREE in-sim. On device it does not, and this is
    # ~80% of the device hole deaths (v78 ep3 @612 + ep0 @12779, v74 ep3 @1722, r2 ep0/ep1/ep3): the
    # runner jumps obstacle A, the hole sits INSIDE its arc, it cannot re-take-off, and its DESCENDING
    # arc drops into the hole.
    # THE SEPARATOR IS ARC PHASE, not landing position: every device death entered the hole with
    # frac = d/span >= 0.51 (past the apex, descending) while all 858 free crossings sit at frac <= 0.41
    # (d<=1 cell = a takeoff on the approach cell, which is what the policy already does and MUST stay
    # free -- 0/858 deaths). Physically exact: reach a hole while still rising/at apex and you sail
    # over; reach it descending and you drop in. Same parabola framing as the crevasse apex model.
    # The kill is PROBABILISTIC (device 6/20 = 30% once past apex) rather than a sharp threshold --
    # the v55 lesson. OFF by default (no rng draw -> equivalence stream untouched).
    "hole_landing_model": False,
    "hole_trap_apex_frac": 0.5,        # arc phase past which a 1-cell hole is lethal (0.5 = the apex)
    "hole_trap_kill_prob": 0.35,       # P(fall) once past it — device 6/20 = 30% (1.0 = deterministic)
    # ONE-CELL CREVASSE share [DEVICE-MEASURED WAVE 14, n=409]: 67.5% of real crevasse structures are
    # a single cell; the sim made them all 2-cell, which put every following obstacle outside the jump
    # arc and made the airborne trap unreachable. 0 = off (all 2-cell, the pre-wave-14 behaviour).
    "crevasse_single_cell_prob": 0.0,
    # TURN-AWARE LOOKAHEAD [WAVE 15 PERCEPTION FIX, device-probed 2026-07-26, hook 17g `s.wp`]: the
    # hook's neighbour walk used ONE fixed heading offset per step, so it died past every corner —
    # 72% of on/near-junction device frames streamed next/next2 = -1 and NOBODY could see past a
    # turn. A heading-aware walk resolves the full 6-cell horizon on 100% of the frames where the old
    # walk fell short (56.8% of all frames, mean +3.46 cells further). With this ON, the lookahead
    # only blanks at a genuine T-JUNCTION (both directions available = the continuation is not yet
    # decided, 32.6% of device junction frames); a forced corner now streams true values.
    # REQUIRES hook >= 17h on the device. Default OFF = the historical always-blind behaviour.
    "turn_aware_lookahead": False,
    # TIGHT OBSTACLE PAIRS [DEVICE-MEASURED WAVE 14, n=2684 must-jump pairs]: 19% of real consecutive
    # must-jump obstacles sit CLOSER than min_space_between_obstacles(6) and 6.2% within the jump arc
    # (<=3 cells). The sim's floor is hard, so only the bespoke compound tails ever break it — and
    # they land at gap 3-4, one cell too generous for the airborne trap. 0 = off (hard floor).
    # ★ THE GAME REPEATS THE SAME OBSTACLE TYPE [DEVICE-MEASURED 2026-07-27 night,
    # experiments/pair_coverage.py over human 41 km + v92b + v104]. The device's two MOST COMMON
    # consecutive-obstacle orderings are same-type slide pairs:
    #     TreeSlide -> TreeSlide  9.8% of all pairs      Flame -> Flame  8.1%
    # i.e. 17.9% of device pairs are a repeated slide, and 24.6% are a repeat of ANY type. An
    # INDEPENDENT draw from the (device-matched) obstacle mix predicts only ~4% slide-repeat, so this
    # is real clustering in the game's generator, not a mix artifact -- and the sim emits 2.9%.
    # USER, watching v104: "The struggle is really when you have 3 obstacles in a row. It really
    # handles it poorly ... you really have to randomize the obstacles that follow one another. You
    # shall not skip the 2 obstacles in a row training!" -- the repeated-slide rhythm (slide, recover,
    # slide) is the single most common device sequence the policy has never trained on.
    # When a NEW obstacle is placed within pair_same_type_cells of the previous one, reuse the
    # previous TYPE with this probability instead of drawing from the weight table. prob>0-first, so
    # 0.0 draws no rng and leaves the stream bit-identical.
    "pair_same_type_prob": 0.0,
    "pair_same_type_cells": 4,
    "pair_same_type_types": (5, 6),   # TreeSlide / FlameTower — see _sample_obstacle_type_paired
    # ANTI-REPETITION for JUMP obstacles: device Hole->Hole is 6.1% against an INDEPENDENT 10.3%,
    # so the game avoids repeating a hole while the sim draws i.i.d. and repeats at chance.
    # Suppressing 41% of hole-after-hole draws maps 10.3% -> 6.1%. 0.0 = off (draws no rng).
    # AIRBORNE INTO A SLIDE HAZARD: P(kill). 1.0 = the old always-fatal rule; the DEVICE measures
    # 0.24 (23 fatal of 94 events over 45 captures). See the death rule for why this matters.
    "airborne_slide_kill_prob": 1.0,
    "pair_anti_repeat_prob": 0.0,
    "pair_anti_repeat_types": (1, 4),   # Hole, TreeJump
    # ★★★ DRAW-THEN-SUBSTITUTE OBSTACLE TYPES [READ FROM THE GAME 2026-07-28,
    # il2cpp_work/output/dump.cs; spec docs/design/2026-07-28-obstacle-type-sampler-rewrite.md].
    # The real generator is CONSTRAINT-FILTERED REJECTION SAMPLING, not an i.i.d. weight draw:
    #     PathElement.ACertainTypeCheck(PathType currentType, PathType parentType)  @283485, 84 B
    #         -- an explicit PAIRWISE admissibility rule, and only ~21 instructions, i.e. ONE special
    #            case rather than a full matrix (so it is NOT a blanket anti-repeat: Flame->Flame is
    #            ELEVATED on device at 8.1% while Hole->Hole is SUPPRESSED at 6.1%).
    #     PathType OriginalPick 0x98 / PathType FinalPick 0x9C
    #         -- the generator draws a type and then SUBSTITUTES it. One substitution, hence one
    #            re-draw (type_reject_redraws 1).
    # Confirmed ABSENT file-wide: any authored pattern/template collection, any ObstacleType[] /
    # List<PathType>, and any per-type probability field. The obstacle numbers are compiled-in
    # literals, so the SHAPE comes from the game and the NUMBERS are calibrated to the device.
    #
    # THE STRUCTURAL KEY: holes / narrows / flames / level-changes are PathType (the cell GEOMETRY)
    # and must survive the pairwise + fit + spacing filters; TREES (TreeStumble/TreeJump/TreeSlide)
    # are ObstacleType, dropped onto an already-chosen straight by a SEPARATE AddObstacle pass that
    # those filters never touch (see _TREE_TYPES). One mechanism therefore predicts both device
    # deviations: a filtered type is pushed BELOW chance, an unfiltered one sits ABOVE it.
    #
    # `type_reject` is {"prev,cur": p_reject} (JSON-friendly) or {(prev, cur): p_reject}; `prev` is
    # the previously emitted HAZARD type (_last_obst_type), `cur` the freshly drawn one. An entry
    # whose `cur` is a TREE is REJECTED AT PARSE TIME rather than silently ignored.
    # MASTER SWITCH default False, and it is checked BEFORE any random() -> the OFF path draws
    # nothing at all and every existing track stream / calibration stays bit-identical.
    # ⚠️ When True the new model REPLACES pair_same_type_prob / pair_anti_repeat_prob in this
    # sampler (they are two models of the same draw, not layers); both stay fully usable with
    # type_reject_model False, which is what reproduces every pre-2026-07-28 result.
    "type_reject_model": False,
    "type_reject": {},
    "type_reject_redraws": 1,
    "tight_pair_prob": 0.0,
    "tight_pair_min_cells": 1,         # closest allowed spacing when a tight pair fires (device min = 1)
    # CREVASSE APEX for ALL level-changes [DEVICE-MEASURED 2026-07-06, vel_y arcs + crevasse_clips.py].
    # The step-UP apex model only fired on level-UP crevasses; step-DOWN / high-ground crevasses used
    # the lenient airborne check -> the sim never penalised an EARLY crevasse jump, so v47 drifted to
    # takeoff cp 0.33 (device: 0.63) and took repeated hits (slowdowns) on high-ground crevasses. The
    # vel_y arcs show step-down ALSO needs near-apex at the far cell (pt35 cp0.96 vel_y~4 = apex). When
    # True, the apex check (+ stepup_clip_band graze->stumble) fires on EVERY crevasse ledge cell. OFF
    # by default (no-op -> equivalence green); enabled in the device-true / training config.
    "crevasse_apex_all": False,
    # ★ STEP-DOWN TO LOW GROUND IS NOT A MUST-JUMP [DEVICE-MEASURED 2026-07-27, human 41 km + the
    # v92b/v102 device sets, scripts/analysis/stepdown_forensic.py]. USER, watching v102: "The problem is
    # also the mid to low transition when you have an obstacle right ahead. The policy learnt to jump
    # when it goes from the mid to the low ground. It does not need to. When there is an obstacle,
    # either you jump early or you don't jump, but the policy can't do that right now."
    #
    # MEASURED, split by the exact level transition (P(jump) is a free CHOICE for the human):
    #     transition      HUMAN P(jump)   grounded crossings / deaths     v92b + v102
    #     1->2  UP           100.0%              0  /  -                     100%
    #     2->3  UP           100.0%              0  /  -                     100%
    #     3->2  down          92.9%              1  /  0                     100%
    #     3->1  down          44.4%              5  /  0                     100%
    #     2->1  down          29.4%             12  /  0                     100%
    # The human crosses 17 of 26 step-downs TO LOW GROUND on foot and NEVER dies; the step-UPs they
    # jump 50/50. The sim collapses that asymmetry: the crevasse gap cell demands airborne in BOTH
    # directions, so "always jump" is the only correct sim policy and the phone inherits it as a
    # superstition. The device streams must_jump=True on 100% of crevasse structures INCLUDING mid->low
    # (checked), so the observation is faithful and this is purely a DEATH-RULE gap -- the flag is HARD
    # in the sim and SOFT on the phone.
    # COST OF THE SUPERSTITION: with a hazard within 4 cells after the cap, the forced jump kills 50%
    # (v102, n=4) and 16.7% (v92b, n=6) vs the human's 0% whether they jump or not -- the jump is both
    # too late to clear the follower and too committed to react to it, which is exactly the user's
    # "either you jump early or you don't jump".
    # WHEN ON: a step-DOWN crevasse whose NEW level is <= stepdown_grounded_max_level may be crossed
    # GROUNDED for free (jumping it stays legal and safe -- the human still does it 29%). Step-UPs and
    # 3->2 are untouched. Bool knob, no rng draw -> the OFF stream is bit-identical (equivalence safe).
    "stepdown_grounded_ok": False,
    "stepdown_grounded_max_level": 1,
    # HIGH-GROUND CAP GRAZE — WORLD-Y model [DEVICE-DECODED 2026-07-07, hook 16y source-dig + arcs;
    # USER-confirmed live 2026-07-07: "mid->high transition, the policy does NOT die but takes a hit almost
    # every time"]. The mh(0.95)-proxy above is a NORMALISED-height threshold that ignores the cap's actual
    # world height. The real mechanic: jump peak is a VELOCITY-INDEPENDENT constant ~19.2 world-units
    # (p25 19.1/p75 19.2, n=186); the raised cap top by level is ground_height = 15 @level-3, 0 @level-2.
    # The graze (Stumble x0.85, non-fatal, CHRONIC) fires iff the arc height at the cap < cap top; a deep
    # clip (arc well below the cap) Ledge-kills. Reproduces device 32% graze @lvl3 / 0% @lvl2: a lvl3 cap
    # (15) sits just under the 19.2 peak so the DESCENDING arc drops below it ~1/3 of takeoff timings; lvl2
    # caps (0) are always cleared (arc height >= 0 in flight). When ON, this SUPERSEDES the mh-proxy on the
    # apex cell (and self-extends apex detection to all last-crevasse cells, like crevasse_apex_all). OFF by
    # default (no-op -> equivalence stays green); enabled in the device-true / training config.
    "crevasse_graze_worldy": False,
    "crevasse_jump_peak": 19.2,        # world-units jump apex height (device-measured, velocity-independent)
    "crevasse_cap_height_hi": 15.0,    # raised cap top (ground_height) at level>=3; level<2 caps = 0 (clear)
    "crevasse_graze_band": 10.0,       # world-units below the cap that still GRAZES (stumble); deeper -> kill
    # STEP-UP CAP ARC OFFSET [DEVICE-DECODED 2026-07-07, v51/v53 first-crevasse deaths @598/609/1258]: the
    # sim evaluates the raised lvl-3 cap at frac = d/span where d = cells(takeoff->ledge); a cp~0.9 takeoff
    # puts the ledge at the jump APEX (height 1) so v51 CLEARS. But device vel_y shows the runner DESCENDING
    # (vel_y -33..-62) at the cap -> it clips (dt=4 Ledge) at LOW speed (~v166, the FIRST crevasse ~587m).
    # The device cap sits ~this many cells LATER in the arc than the sim models -> add to d so the cap falls
    # in the DESCENDING arc (height drops) and v51 clips it in-sim like on device -> retrain learns the
    # earlier takeoff. 0 default (no-op -> equivalence/scorecard unchanged); calibrate to the device rate.
    "crevasse_cap_arc_offset": 0.0,    # cells added to the takeoff->cap distance (shifts the cap later in the arc)
    # TAKEOFF-TIMING VARIANCE [DEVICE-MEASURED 2026-07-08, probe_jump_timing + v51_crevasse_16z]: a FIXED
    # cap-arc offset (v55) gave a SHARP all-or-nothing threshold (no clean partial rate) because in-sim v51's
    # takeoff is near-deterministic. On device the takeoff cell_progress VARIES run-to-run (std ~0.06-0.1 cp),
    # so at the descending arc the world-Y clearance spreads around the cap top -> a FRACTION grazes (device
    # ~32% @lvl3) rather than 0% or 100%. Model it as a per-crossing Gaussian shift of d (in cells), drawn
    # from self._rng exactly like jump_exec_lat_jitter (line ~1209). Gated on >0 (default 0 -> NO rng draw ->
    # deterministic-suite RNG stream unchanged -> equivalence baseline green; tested via the STATISTICAL tier
    # when on). Calibrate WITH crevasse_cap_arc_offset so mean(arc_y)~cap+0.5*std*slope -> ~32% below.
    "crevasse_takeoff_std": 0.0,       # cells; per-crevasse-cap Gaussian takeoff-position jitter (0 = off)
    "wide_gap_land_grace": 0.35,       # far-lip landing grace: land within this cprog of the solid
                                       # cell and still clear. Calibrated (sweep_grace.py + calib_landing.py):
                                       # ~20% device clip rate, and leaves the GOOD jumper v36 almost
                                       # untouched (grace-insensitive plateau) while punishing jump-spam.
                                       # Domain-rand (jump jitter, dr velocity) supplies the real span
                                       # variance (~0.29) so the policy can't exploit the hard boundary.
    # Powerup spawn probabilities are all 0 in the baseline read (sProbabilityBoost/Invincibility/
    # Vacuum/CoinBonus = 0) — why human runs saw 0 boost frames; the sim models no powerups (v1).
    "min_spaces_between_coin_runs": 12, # [READ] sMinSpacesBetweenCoinRuns
    "max_coins_per_run": 5,            # [READ] sMaxCoinsPerRun
    # obstacle-type mix among the "real" obstacles (must sum to ~1; calibrated later).
    # Keys are ObstacleType enum values (SPEC §2): 1 Hole, 2 Narrow, 3 TreeStumble,
    # 4 TreeJump, 5 TreeSlide, 6 FlameTower. (0 None, 7 LevelChange handled separately.)
    # [DEVICE-MEASURED 2026-07-27] pooled over 28,600 device cells / 4112 hazard structures across 6
    # capture sets, structure-collapsed (a 2-cell HoleLong or a multi-cell narrow counts ONCE), and
    # independently reproduced by the 41 km human set to within ~0.02 on every entry. Crevasses are
    # EXCLUDED — they come from the zone/level-change path, not this table (13-14% of all structures).
    # ⚠️ The OLD weights were badly off in composition even after total density was right: Narrow 0.08
    # vs 0.173 measured (2.2x UNDER) while TreeJump 0.22 vs 0.103 and TreeSlide 0.22 vs 0.110 were each
    # ~2x OVER. So the policy got half the strip practice it needed and double the tree practice --
    # USER 2026-07-27: "not as great on the narrow"; v78 died on 2.47% of narrow encounters vs
    # v90/v92 7.78% (p=0.028) and the human 0.95%. Total hazards/km was already correct (16.4 sim vs
    # 15.2 device), which is exactly why a density-only audit missed this: the MIX is a separate axis.
    # ⚠️ 2026-07-27 — REVERTED to the long-standing values after a MEASUREMENT ERROR of mine. I read the
    # sim's narrow share as 8% vs the device's 17% and "corrected" Narrow 0.08 -> 0.186. BOTH halves were
    # wrong:
    #   (a) A REAL narrow strip in this sim is encoded as path_type 17..30 with obstacles LEFT AT 0
    #       (see the narrow branch below, which sets ptype 19/22 and narrow_lane). Only the GENERIC
    #       weight-table path emits obstacles == 2 — and that path cannot set narrow_lane, so
    #       `_resolve_obstacle_on_enter` skips the band check (`if nl is not None`) and those cells are
    #       HARMLESS DECORATION. Counting obstacles == 2 therefore counted ONLY the fake narrows and
    #       missed every real strip. Counting path_type 17..30 as well: the sim emits ~41% narrows vs
    #       the device's 14% — it OVER-serves them ~3x, the opposite of what I concluded.
    #   (b) The narrow branch fires BEFORE this table is consulted, so raising the Narrow weight did not
    #       change narrow frequency at all. It only replaced REAL hazards (TreeJump 0.22 -> 0.078) with
    #       no-op cells — which is what regressed v94 (device 2193/2309/2305/1565, USER: "it strafes
    #       randomly to the left"; L/R strafe ratio 0.75 -> 1.78 -> 2.74 across v90/v92b/v94).
    # LESSON: verify that a generator knob actually MOVES the thing you measured before trusting the
    # measurement — and check how a mechanic is ENCODED on both sides before comparing shares.
    # The real narrow/hole/flame composition gap is still open; fix it via `narrow_prob` (the branch
    # that makes LANED strips), not via this table, and re-measure with narrows counted correctly.
    "obstacle_type_weights": {
        1: 0.22,   # Hole         -> jump
        4: 0.22,   # TreeJump     -> jump
        5: 0.22,   # TreeSlide    -> slide
        6: 0.14,   # FlameTower   -> slide
        3: 0.12,   # TreeStumble  -> stumble (survivable)
        2: 0.08,   # Narrow       -> generic no-lane cell (NOT a real strip; see the note above)
    },
    # piece PathType to emit for straight vs turn cells (obs path_type /37). [KNOWN enum]
    "path_type_straight": 1,           # StraightVert
    "path_type_turn": 3,               # TurnA (any of 3..6 are turns; 3 is representative)
    "path_type_t_junction": 7,         # T-junction (both sides turnable)
    # --- C. Scoring ----------------------------------------------------------
    # Calibrated so a start-velocity step (vel 100, dt 0.1 -> distance 10) yields ~+27 score,
    # matching CLAUDE.md §9 (~+27 score / 100ms-step at run start, growing with velocity).
    "score_per_distance": 2.7,         # [INFERRED] score units per distance unit (dist = vel*dt)
    "start_score": 162,                # [KNOWN-ish] reset frame score ~162 (CLAUDE.md §9)
    "coin_score": 50,                  # [INFERRED] score bump when a coin/bonus cell is consumed
    # The reward keys on DISTANCE (DistanceTraveled), which advances ~0.107 per (velocity*dt) of
    # model distance in real units (real dist-rate/velocity ~0.10-0.116; vel 100 -> ~1.16/10Hz-step).
    # The sim emits a real-scale "distance" so its distance-delta reward matches the phone's units.
    "dist_per_velsec": 0.107,          # [CALIBRATED] real DistanceTraveled per (velocity*dt) model unit
    # [DEVICE-MEASURED 2026-07-27] ...but the real rate is NOT constant: it rises linearly with speed.
    # Pooled over 154 capture files / 770k frame pairs, dist_per_velsec/velocity is 8.56, 8.46, 8.47,
    # 8.48 e-4 across v105..v215 -- one constant, no fitting freedom -- i.e. DistanceTraveled accrues
    # as ~8.47e-4 * v^2 per second. Device and human captures agree to within 5% in every band.
    # The flat 0.107 is right only near v=136 and understates distance ~1.4-1.5x at running speed, so
    # the sim needs ~1.39x MORE CELLS to report the same distance as the phone. That single error made
    # (a) every sim-vs-device distance comparison in this project off by ~1.5x, (b) the distance-keyed
    # difficulty ramp fire ~1.4x too late in cell terms, and (c) hazards-per-km look 1.44x over-packed
    # when hazards per CELL are actually correct (0.152 sim vs 0.146 device = 1.04). Set the slope to
    # enable the measured law; 0.0 keeps the old flat scalar so historical runs stay reproducible.
    "dist_per_velsec_slope": 0.0,      # [DEVICE-MEASURED] 8.47e-4 => dist rate = slope * velocity^2
    "start_distance": 5.3,             # [CALIBRATED] DistanceTraveled at run start (reset-frame offset)
    # HIGH-SPEED CURRICULUM: the runner accelerates (velocity 100->210 over ~4k distance), so the turn
    # window gets HARDER late (cprog steps faster per frame). To train the fast sections without first
    # surviving 4k, start a fraction of episodes at a random high distance (velocity set from the curve
    # there + the difficulty ramp already keyed to distance). 0 = always start slow.
    "curriculum_max_start_dist": 0.0,  # if >0, sample episode start distance in [0, this]
    # TAMING (2026-07-03): the raw 50%-uniform curriculum over-weighted MAX speed (velocity saturates
    # ~4000m, so uniform[0,8000] puts ~half the deep-starts at flat top speed) and CORRUPTED the
    # mid-speed crevasse jump-timing the device actually reaches first (ppo_speed_v8 regressed 1173->830,
    # 4/8 crevasse falls at v165-180). Two knobs: FRACTION of episodes that jump-start, and a distance
    # POWER that biases the sample toward the low/mid-speed ramp (power>1 -> d = cmax*u**power skews to 0).
    "curriculum_deep_start_frac": 0.35,  # P(an episode jump-starts) — was a hardcoded 0.5
    "curriculum_dist_power": 1.5,        # >1 biases start-distance toward 0 (keep mid-speed well-sampled)
    # --- D. Action timings (seconds; durations of the transient flags) -------
    # [DEVICE-MEASURED 2026-06-26] is_jumping run-length median 8 frames @10Hz = 0.80s (tight: p90=0.8);
    # is_sliding median 7 = 0.70s. The old 0.6 made the runner airborne ~2x too little vs real (14% vs
    # 22-32%) AND made the 2-cell crevasse too harsh — a real 0.8s jump spans ~2.8 cells at top speed.
    "jump_duration": 0.8,             # [MEASURED] airtime — IsJumping true ~0.80s
    # ★ AIRTIME SCALES WITH THE GROUND-HEIGHT CHANGE THE ARC CROSSES [DEVICE-MEASURED 2026-07-27,
    # experiments/stepdown_arc.py]. Landing on LOWER ground means falling further, so the runner stays
    # airborne LONGER and the arc spans 3 cells instead of 2; landing on RAISED ground cuts the fall
    # short. Two INDEPENDENT capture pipelines agree on direction AND magnitude:
    #                        flat            step-DOWN            step-UP
    #     HUMAN  30 Hz   0.767 s / 2 cells  0.900 (1.17x) / 3   0.567 (0.74x) / 2
    #     v92b   10 Hz   0.700 s / 2 cells  0.800 (1.14x) / 3   0.500 (0.71x) / 2
    # The sim used ONE fixed jump_duration for all three. Consequence, measured: a step-down jump with
    # a hazard within 4 cells kills 0.0% in-sim (n=41) but 50.0% on device (v102, n=4) and 16.7%
    # (v92b, n=6) -- the sim's 2-cell arc lands short of the follower, the device's 3-cell arc lands
    # ON it. That missing COST is why v103 (world fix) and v104 (world fix + reward gate) both left
    # P(jump | step-down to low) at 98.5%: with no death and no bonus the choice became reward-NEUTRAL,
    # and a habit already at ~100% has no gradient to move it.
    # The step-UP scale matters for the OTHER long-standing complaint: if real step-up arcs are ~25%
    # SHORTER than the sim's, the sim has been teaching a takeoff timing that does not land where the
    # phone lands (the mid->high graze).
    # Values are the HUMAN column (30 Hz = 0.033 s resolution; the 10 Hz policy sets quantise to 0.1 s
    # and agree to within one frame). 1.0/1.0 = old fixed-duration behaviour, no rng draw.
    "jump_dur_stepdown_scale": 1.0,
    "jump_dur_stepup_scale": 1.0,
    # JUMP EXECUTION LATENCY (2026-07-05): the device fires a jump ~1-2 frames after the command (TCP +
    # obs-staleness + frame lag), the SAME control latency that kills turns — but the sim fired jumps
    # INSTANTLY, giving a too-forgiving ~6-frame survival window at ALL speeds (measured). So v14 learned a
    # lazy "jump anywhere in the window" that works in-sim but the device latency pushes off the real
    # (narrower, speed-shifted) window -> Ledge deaths at high speed. Model it: buffer the jump firing by
    # jump_exec_lat_frames (+/- jump_exec_lat_jitter frames of noise so the landing isn't perfectly
    # predictable -> narrows the RELIABLE window). This forces the policy to PRE-AIM using predicted_jump
    # (the velocity-invariant takeoff cue), which is exactly what should transfer to the phone. TRAINING
    # only; the fidelity harness/replay run at 0 (instant, for one-step device-transition tests).
    "jump_drop_prob": 0.0,            # [2026-07-05] P(a JUMP byte is silently eaten) — device-observed (game
                                      # refuses Jump() in some zones, e.g. right after a turn cell). Train
                                      # ~0.10 to teach the closed-loop retry; keep 0 for harness/replay.
    "jump_refuse_strafing_prob": 0.6, # [2026-07-06 DEVICE-MEASURED, 3 fresh captures, deconfounded] the
                                      # game REFUSES Jump() during a lateral glide: lane-moving refusal
                                      # 5/9 (56%) vs 2/54 (4%) baseline once landing-phase and gap-cell
                                      # cases are excluded. Faithful DEFAULT ON -> level (NoOp) first,
                                      # then jump. (The apparent "landing lock" was the airborne rule +
                                      # 10Hz phase — already modeled by jump_exec_lat; no extra knob.)
    "gap_jump_lip_cp": 0.2,           # [2026-07-06 DEVICE-MEASURED] on a wide-gap/crevasse cell the game
                                      # refuses Jump() past the solid lip (fired takeoffs on-gap only at
                                      # cp<=0.18; 8 refusals at cp 0.15-0.95 incl ALL grounded-at-gap
                                      # crevasse deaths). Matches gap_entry_grace (same lip). Jump from
                                      # the APPROACH cell — predicted_jump already cues that takeoff.
    "strafe_exec_lat_frames": 0,      # [2026-07-06] CONTROL LATENCY on the STRAFE/level tilt command (the
                                      # commanded lat_target reaches the lane-lag N frames late). Device
                                      # control latency is ~1 frame uniform (turn+jump both measured 1);
                                      # strafe was UNMODELED (instant) -> the sim never reproduced the
                                      # narrows-AT-SPEED death (v36 device ep0@7040/ep4: reached lane only
                                      # +-1.8-2.8, short of the [3,7] band, fell off the inner edge — at
                                      # v210 the 1-cell narrow lookahead + strafe lag = tilt starts too
                                      # late to reach the band). Default 0 (harness/replay unchanged);
                                      # train at 1. Absorbed constant (policy strafes earlier) — the
                                      # POINT is to teach the earlier strafe.
    "jump_exec_lat_frames": 0,        # mean frames the JUMP action is buffered before it fires (0 = instant)
    "jump_exec_lat_jitter": 0,        # +/- uniform frame jitter on the above (models latency variance)
    "slide_duration": 0.7,            # [MEASURED] SlideDuration ~0.70s
    "strafe_duration": 0.25,          # [INFERRED] strafe lane-shift dur@0x12C
    # [DEVICE-MEASURED 2026-06-26] a strafe is a GRADUAL lateral glide, not an instant lane jump: one
    # [DEVICE-CALIBRATED 2026-07-01, controlled strafe probe scripts/probes/probe_strafe.py, 8 runs] ONE strafe
    # byte drives the lane a LONG way and HOLDS: from ~-2.5, StrafeLeft glided to the -7 EDGE over ~6
    # frames; StrafeRight to ~+4.8 over ~5 frames; then held (no recenter). => a strafe reaches ~7 units
    # (`strafe_reach`, clamped to the edge), NOT one 3.7 lane-step, over ~0.6 s. The OLD model (3.7 over
    # 0.35 s + auto-recenter) was a gentle self-correcting nudge -> the policy learned soft strafes that
    # SLAM to the edge and STICK on device -> off-center -> fatal turns (device median 226/201/295 < 736).
    "strafe_glide_time": 0.6,         # [MEASURED] seconds for one strafe's lane glide to complete (~6 frames)
    "strafe_reach": 7.0,              # [MEASURED] lane units one strafe covers (clamped to lane_edge) — big + held
    "stumble_duration": 0.5,          # [INFERRED] StumbleDuration; unrecovered -> death window
    # --- E. Geometry ---------------------------------------------------------
    "cell_length": 60.0,              # PROCEDURAL mode: cells crossed per (velocity*dt) of model distance
    # REPLAY mode crosses cells off the RECORDED hidden.distance (decoupled from the velocity
    # model so path_type tests cell-length alone). Calibrated from real traces: the cell-to-cell
    # gap in hidden.distance units is ~6.7 (median over 24 crossings, 3 episodes); the Temple
    # start-piece is partial, so the player begins ~3.2 into the first cell (first crossing after
    # ~3.5 distance, then every cell_length_real).
    "cell_length_real": 6.4,          # [CALIBRATED] hidden.distance units between adjacent PathElements
                                      # (8 body cells span distance 8.64->59.69 in ep0 => 6.38/cell)
    "replay_start_into_cell": 3.2,    # [CALIBRATED] dist_into_cell at run start (Temple is partial)
    "lookahead": 12,                  # cells of track kept ahead of the player (>= SIM_CONTRACT K>=8)
    # --- F. Domain randomization (sim-to-real robustness) --------------------
    # The policy ACES a clean sim (0 deaths / 20k steps) but the phone has timing variance — the
    # ~95% turn reliability, sub-step lag, velocity jitter. DR samples these per EPISODE so the
    # policy learns ONE robust behaviour (turn on the first cue, jump early) instead of overfitting
    # one exact timing. Ranges chosen so the oracle still survives the WHOLE range (fairness-checked).
    # OFF by default (fidelity harness/replay need determinism); train_sim flips it on.
    "domain_rand": False,
    # [DEVICE-REMODEL 2026-07-01] is_turn is a ~1-frame flash at cell start (cprog p90 0.38). Jitter the
    # presentation window around 0.40 so the policy is robust to a slightly wider/narrower flash; the
    # off-center early/late death is DELETED (early swipes are how humans turn + the game buffers them),
    # so those DR ranges are pinned to their no-op extremes.
    "dr_turn_window_frac": [0.40, 0.52],   # is_turn/direct-exec upper bound (flash width jitter)
    "dr_turn_early_frac":  [0.0, 0.0],     # [DELETED] no early-turn death
    "dr_turn_late_frac":   [1.0, 1.0],     # [DELETED] no late-turn death
    "dr_jump_duration":    [0.72, 0.88],   # [MEASURED] real airtime ~0.80s (run-len 7-9 frames)
    "dr_slide_duration":   [0.62, 0.80],   # [MEASURED] real ~0.70s
    "dr_velocity_scale":   [0.94, 1.08],   # multiply the curve velocity per episode
    "dr_obstacle_prob":    [0.35, 0.55],   # obstacle density jitter
    "dr_turn_prob":        [0.12, 0.28],   # early-game turn density jitter
    "dr_cell_length":      [56.0, 64.0],   # procedural cell length (cross timing jitter)
}

# ObstacleType enum (SPEC §2) -> which flag must be active when the cell is passed.
OBST_NONE = 0
OBST_HOLE = 1
OBST_NARROW = 2
OBST_TREE_STUMBLE = 3
OBST_TREE_JUMP = 4
OBST_TREE_SLIDE = 5
OBST_FLAME_TOWER = 6
OBST_LEVEL_CHANGE = 7

# TreeStumble (the lane-dependent "tree-root": left-lane=jump-or-die, middle=stumble-2nd-kills,
# right=free) is treated as MUST_JUMP — jumping clears it safely on every lane, so the policy
# learns one robust response instead of needing lane awareness. Observed killing the agent on the
# phone (must_jump=0 -> it NoOp'd -> died).
_MUST_JUMP = frozenset({OBST_HOLE, OBST_TREE_JUMP, OBST_TREE_STUMBLE, OBST_LEVEL_CHANGE})  # jump-or-die
# (OBST_LEVEL_CHANGE = the zone/altitude-transition crevasse; jump clears it — matches hook geo-latch-4)
_MUST_SLIDE = frozenset({OBST_TREE_SLIDE, OBST_FLAME_TOWER})  # must_slide = Obstacles ∈ {5,6}
_SURVIVABLE = frozenset({OBST_NARROW})                    # don't kill in v0 (strafe)

# ★ THE UNFILTERED SECOND PASS [READ FROM THE GAME 2026-07-28]. Trees are `ObstacleType`, placed
# onto an already-chosen straight by `AddObstacle`, which runs AFTER (and independently of) the
# PathType admissibility / fit / spacing filters. Everything else in the weight table is `PathType`
# = the cell geometry, and must pass those filters. See `type_reject_model`: Stage A (geometry) is
# subject to the pairwise rejection test, Stage B (these) never is. That asymmetry is the whole
# model -- it is what lets one mechanism push Hole->Hole BELOW chance while leaving the repeated
# slides (the device's two most common consecutive orderings) ABOVE it.
_TREE_TYPES = frozenset({OBST_TREE_STUMBLE, OBST_TREE_JUMP, OBST_TREE_SLIDE})

# PathDirection enum (SPEC §0): None0 North1 East2 South3 West4.
DIR_NONE, DIR_N, DIR_E, DIR_S, DIR_W = 0, 1, 2, 3, 4
# left/right of a heading (turning relative to current facing).
_LEFT_OF = {DIR_N: DIR_W, DIR_W: DIR_S, DIR_S: DIR_E, DIR_E: DIR_N}
_RIGHT_OF = {DIR_N: DIR_E, DIR_E: DIR_S, DIR_S: DIR_W, DIR_W: DIR_N}


_EMPTY_TYPE_REJECT: dict = {}


def parse_type_reject(raw) -> dict:
    """Normalise the ``type_reject`` config into ``{(prev, cur): p_reject}``.

    Accepts the JSON-friendly ``{"prev,cur": p}`` form (what a ``--sim-config`` file carries, where
    dict keys are always strings) and the Python ``{(prev, cur): p}`` tuple form.

    Raises rather than ignoring anything it cannot honour — an inert knob that looks applied is the
    single most expensive failure mode in this project (§5 of the spec; four measurements in one
    session silently tested nothing). Specifically:
      * a key that is not exactly two integers;
      * a type outside 0..7 (the C++ port indexes a fixed 8x8 matrix);
      * an entry whose CURRENT type is a TREE — trees are placed by the unfiltered second pass
        (`_TREE_TYPES`) and are never rejected, so the entry could never fire.
    Entries with p == 0 are dropped (that is the no-rejection default, and it keeps the parsed
    table identical to the C++ side's zero-initialised matrix).
    """
    tbl: dict = {}
    for k, v in (raw or {}).items():
        if isinstance(k, (tuple, list)):
            parts = list(k)
        else:
            parts = str(k).split(",")
        if len(parts) != 2:
            raise ValueError(f"type_reject key {k!r}: expected 'prev,cur' (two obstacle types)")
        try:
            a, b = int(parts[0]), int(parts[1])
        except (TypeError, ValueError):
            raise ValueError(f"type_reject key {k!r}: both sides must be integer obstacle types")
        if not (0 <= a <= 7 and 0 <= b <= 7):
            raise ValueError(f"type_reject key {k!r}: obstacle types must be in 0..7")
        if b in _TREE_TYPES:
            raise ValueError(
                f"type_reject key {k!r}: the CURRENT type {b} is a TREE. Trees are placed by the "
                f"game's separate AddObstacle pass, which the pairwise filter never touches, so "
                f"this entry would be silently inert. Reject a geometry type instead (1 Hole, "
                f"2 Narrow, 6 FlameTower).")
        p = float(v)
        if p != 0.0:
            tbl[(a, b)] = p
    return tbl


def _element(type: int = 1, level: int = 2, obstacles: int = 0,
             is_turn: bool = False, can_turn_left: bool = False,
             can_turn_right: bool = False, is_coin: bool = False) -> dict:
    """A single track cell (the sim's internal PathElement)."""
    return {
        "type": int(type),
        "level": int(level),
        "obstacles": int(obstacles),
        "is_turn": bool(is_turn),
        "can_turn_left": bool(can_turn_left),
        "can_turn_right": bool(can_turn_right),
        "is_coin": bool(is_coin),
    }


class TempleRunSimEnv(gym.Env):
    """Headless Temple Run simulator (see module docstring). Drop-in for ``TempleRunEnv``."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        track_source: "str | list[dict]" = "procedural",
        config: Optional[dict] = None,
        seed: Optional[int] = None,
        step_dt: float = C.DEFAULT_STEP_DT,
        max_episode_steps: int = C.DEFAULT_MAX_EPISODE_STEPS,
        reward_scale: float = C.DEFAULT_REWARD_SCALE,
        reward_mode: str = C.DEFAULT_REWARD_MODE,
        action_delay: int = 0,
        death_penalty: float = C.DEFAULT_DEATH_PENALTY,
        # same shaping knobs as the real env so training transfers (see reward.py):
        action_cost: float = C.DEFAULT_ACTION_COST,
        reaction_bonus: float = C.DEFAULT_REACTION_BONUS,
        turn_spam_penalty: float = C.DEFAULT_TURN_SPAM_PENALTY,
        idle_action_penalty: float = C.DEFAULT_IDLE_ACTION_PENALTY,
        redundant_action_penalty: float = C.DEFAULT_REDUNDANT_ACTION_PENALTY,
        airborne_action_penalty: float = 0.0,
        jump_hold_penalty: float = 0.0,
        strafe_penalty: float = 0.0,
        narrow_hold_bonus: float = 0.0,
        aim_center_weight: float = 0.0,
        velocity_drop_penalty: float = 0.0,
        off_center_penalty: float = 0.0,
        slide_jump_penalty: float = 0.0,
        turn_buffer_bonus: float = 0.0,
        landing_hazard_penalty: float = 0.0,
        turn_takeoff_penalty: float = 0.0,
        action_history_len: int = C.DEFAULT_ACTION_HISTORY_LEN,
        frame_stack: int = C.DEFAULT_FRAME_STACK,
        **_ignored: Any,
    ):
        super().__init__()
        # IDENTICAL spaces to TempleRunEnv -> drop-in. obs = frame_stack*OBS_LEN (+ K*7 action history).
        self.action_history_len = int(action_history_len)
        self.frame_stack = max(1, int(frame_stack))
        self.observation_space = O.make_observation_space(self.frame_stack, self.action_history_len)
        self.action_space = gym.spaces.Discrete(C.N_ACTIONS)
        self._act_hist: list[int] = []
        self._frames: list = []

        # Merge config over defaults (shallow + one level for the obstacle weights).
        self.cfg = dict(DEFAULT_SIM_CONFIG)
        if config:
            for k, v in config.items():
                if k == "obstacle_type_weights" and isinstance(v, dict):
                    w = dict(DEFAULT_SIM_CONFIG["obstacle_type_weights"])
                    # keys arrive as STRINGS from a JSON --sim-config; mixing them with the int
                    # defaults yields a dict whose keys cannot be sorted (str vs int) and the torch
                    # env dies building obst_types. Coerce so both spellings work.
                    w.update({int(kk): vv for kk, vv in v.items()})
                    self.cfg[k] = w
                else:
                    self.cfg[k] = v
        # Immutable base; `self.cfg` is re-derived from it each reset (domain randomization mutates
        # the per-episode copy, never the base).
        self._base_cfg = dict(self.cfg)
        # Validate + cache the type_reject matrix HERE rather than at the first hazard draw, so a
        # malformed table fails at construction (like the C++ backend does) instead of thousands of
        # steps into a run -- or, worse, never, if that branch happens not to be reached.
        self._type_reject_table()
        self._dr = bool(self.cfg.get("domain_rand", False))
        self._vel_scale = 1.0

        self.step_dt = float(step_dt)
        self.max_episode_steps = int(max_episode_steps)
        self.reward_scale = float(reward_scale)
        self.reward_mode = str(reward_mode)
        self.action_delay = int(action_delay)   # steps the chosen action is buffered before applying
        self._action_buffer: list[int] = []
        self.death_penalty = float(death_penalty)
        self.action_cost = float(action_cost)
        self.reaction_bonus = float(reaction_bonus)
        self.turn_spam_penalty = float(turn_spam_penalty)
        self.idle_action_penalty = float(idle_action_penalty)
        self.redundant_action_penalty = float(redundant_action_penalty)
        self.airborne_action_penalty = float(airborne_action_penalty)
        self.jump_hold_penalty = float(jump_hold_penalty)
        self.strafe_penalty = float(strafe_penalty)
        self.narrow_hold_bonus = float(narrow_hold_bonus)
        self.aim_center_weight = float(aim_center_weight)
        self.velocity_drop_penalty = float(velocity_drop_penalty)
        self.off_center_penalty = float(off_center_penalty)
        self.slide_jump_penalty = float(slide_jump_penalty)
        self.turn_buffer_bonus = float(turn_buffer_bonus)
        self.landing_hazard_penalty = float(landing_hazard_penalty)
        self.turn_takeoff_penalty = float(turn_takeoff_penalty)

        # Velocity curve (distance -> velocity) for fast interpolation in _advance.
        vc = self.cfg.get("velocity_curve") or []
        self._vel_curve_d = np.array([p[0] for p in vc], dtype=float) if vc else None
        self._vel_curve_v = np.array([p[1] for p in vc], dtype=float) if vc else None

        # Track source: "procedural" (RNG) or an injected list of element dicts (replay).
        if isinstance(track_source, str):
            self.track_source = "procedural"
            self._injected_track: Optional[list[dict]] = None
        else:
            self.track_source = "replay"
            self._injected_track = [self._coerce_element(e) for e in track_source]

        self._rng = np.random.default_rng(seed)

        # episode/bookkeeping state (set up in reset())
        self._prev_score = 0
        self._prev_distance = 0.0
        self._prev_state: Optional[dict] = None
        self._last_action: Optional[int] = None
        self._elapsed_steps = 0
        self._reset_internal_state()

    # ------------------------------------------------------------------ config
    @staticmethod
    def _coerce_element(e: dict) -> dict:
        """Normalize an injected/replay element dict to the internal schema."""
        return _element(
            type=int(e.get("type", e.get("path_type", 1))),
            level=int(e.get("level", e.get("path_level", 2))),
            obstacles=int(e.get("obstacles", 0)),
            is_turn=bool(e.get("is_turn", False)),
            can_turn_left=bool(e.get("can_turn_left", False)),
            can_turn_right=bool(e.get("can_turn_right", False)),
            is_coin=bool(e.get("is_coin", False)),
        )

    # ------------------------------------------------------- internal physics
    def _reset_internal_state(self) -> None:
        self._velocity = float(self.cfg["start_velocity"])
        self._facing = DIR_N
        self._lane = 0.0                     # CONTINUOUS lateral tilt offset, center 0 (recenters when idle)
        self._tilt_dir = 0                   # current tilt direction while strafing (-1 left / +1 right)
        self._lat_target = 0.0               # ACTIVE tilt target driving the lane lag (post-latency)
        self._lat_cmd = 0.0                  # COMMANDED tilt target (this frame's strafe/level byte)
        self._lat_target_buf: list[float] = []   # strafe_exec_lat_frames delay buffer (cmd -> active)
        self._distance = 0.0
        self._distance_real = float(self.cfg["start_distance"])  # real-scale DistanceTraveled (reward signal)
        # high-speed curriculum: jump-start a fraction of episodes deep into the run (fast + dense).
        _cmax = float(self.cfg.get("curriculum_max_start_dist", 0.0))
        _frac = float(self.cfg.get("curriculum_deep_start_frac", 0.5))
        if _cmax > 0.0 and self.track_source != "replay" and self._rng.random() < _frac:
            _pw = max(0.1, float(self.cfg.get("curriculum_dist_power", 1.0)))
            _u = self._rng.random() ** _pw     # power>1 skews toward 0 -> more mid-speed, less flat-top
            self._distance_real = float(self.cfg["start_distance"]) + _u * _cmax
            _vc = getattr(self, "_vel_curve_d", None)
            if _vc is not None:
                self._velocity = float(np.interp(self._distance_real, _vc, self._vel_curve_v))
        self._last_ext_distance = 0.0        # replay: last recorded hidden.distance (cell-crossing anchor)
        self._dist_into_cell = 0.0           # how far the player is into the current cell
        self._cell_index = 0                 # index of the cell the player is on (in self._track)
        self._is_dead = False
        self._death_type: Optional[int] = None
        # jump/slide ACTION recency (frames since the last JUMP/SLIDE action byte) for the windowed
        # jump_hold_penalty (v61): a re-jump within jump_hold_window frames of a prior jump is docked.
        self._jump_recency = 99
        self._slide_recency = 99
        # transient action flags as time-remaining counters (seconds)
        self._jump_t = 0.0
        # STEP-UP timing: longitudinal position (cell_index + cell_progress) where the CURRENT jump took
        # off, so a level-UP crevasse can check the runner jumped late enough (near the gap edge). None
        # until the first jump; refreshed each fresh takeoff (a new jump only starts once airborne ends).
        self._jump_takeoff_pos: Optional[float] = None
        self._pending_jump_frames = 0     # >0 = a JUMP command buffered by jump_exec_lat_frames, fires at 0
        self._cliff_edge_frames = 0       # consecutive frames riding |lane|>=cliff_edge_lane on level>=3
        self._slide_t = 0.0
        self._turn_debounce_t = 0.0          # hook anti-double-turn debounce (turn_drop_debounce_s)
        self._frames_since_land = 99         # frames since the last jump touchdown (post_land_gap_rejump)
        self._slide_air_rolled = -1          # cell index whose airborne-slide kill roll was made
        self._jump_held_airborne = False     # JUMP pressed while airborne -> fires on touchdown (WAVE 16)
        self._strafe_t = 0.0
        self._stumble_t = 0.0
        # STUMBLE kill-window: time since the last stumble (counts UP ~dt/frame; reset to 0 on a stumble).
        # Start "safe" (= window) so the FIRST stumble is always survivable. A 2nd stumble while this is
        # still below stumble_kill_window (or while still stumbling) -> death (matches device measurement).
        self._stumble_kill_t = float(self.cfg["stumble_kill_window"])
        self._stumble_count = 0
        self._stumble_vel = None          # persistent post-stumble velocity (None = on the curve)
        self._turn_landed = False         # a turn landed this step -> skip the recenter (device timing)
        self._gap_grace_cell = None       # wide-gap cell entered grounded (lip grace pending)
        self._post_turn_linger = 0      # frames is_turn stays True after a turn (sim-to-real trap)
        self._linger_can_l = False
        self._linger_can_r = False
        self._pending_turn = None       # a buffered early swipe (byte), fires on entering the turn area
        self._pending_turn_ttl = 0
        self._ground_height = 0.0
        self._score = float(self.cfg["start_score"])
        # Start the spacing counters at 0 (NOT the minimum): the real run has a straight intro
        # (first turn @ ~12 cells = sMinSpaceBetweenTurns, first obstacle @ ~6-29 cells), so a cell
        # only becomes turn/obstacle-eligible after that many cells from the start.
        self._cells_since_obstacle = 0
        self._cells_since_turn = 0
        self._cells_since_bonus = 0
        self._cells_since_zone = 0
        self._zone_level = int(self.cfg["default_path_level"])   # current altitude zone (mid at start)
        self._b2b_obstacles = 0
        # wide-gap (2-cell long hole / zone change) continuation
        self._pending_gap = 0
        self._pending_double = False
        self._pending_gap_stepup = False   # the pending crevasse's 2nd cell is a raised ledge (step-UP)
        self._pending_gap_stepdown_low = False  # the pending crevasse drops to LOW ground (foot-crossable)
        self._last_obst_type = 0     # type of the previous hazard (pair_same_type_prob)
        self._cells_since_hazard = 99  # cells since that hazard was emitted
        self._pending_gap_obst = 0
        self._pending_gap_type = 1
        self._pending_gap_level = 2
        # NARROW strip continuation: device strips span 1-7 cells (commonly 2-3), so the runner must
        # HOLD the safe lane across N cells, not just match it for one frame (the old 1-cell model let a
        # policy strafe back too early and fall off the 2nd/3rd cell on the phone).
        self._pending_narrow = 0
        self._pending_narrow_lane = 0
        self._pending_sequence: list[dict] = []   # compound stacked-obstacle queue (user blindspot)
        # sDoubleObstaclePercent (device-read, ramps 0 -> 0.20): the NEXT cell after an emitted
        # obstacle also carries one, bypassing min_space. Capped by max_back_to_back_obstacles(=2 on
        # the real device). This is the mechanism behind the device's gap-1 hazard pairs, previously
        # absent from the sim entirely.
        self._pending_double = False
        self._pending_post_gap: list[dict] = []    # jump->slide tail (slide obstacle after a wide-gap)
        self._pending_narrow_type = 17
        # replay cursor (which injected element feeds the next generated cell)
        self._replay_cursor = 0
        # the live track window: a list of cells, [_cell_index] = current.
        self._track: list[dict] = []
        self._build_initial_track()

    def _build_initial_track(self) -> None:
        # cell 0 is always a plain straight start cell (the reset/settle cell).
        self._track = [_element(type=self.cfg["path_type_straight"],
                                level=self.cfg["default_path_level"])]
        self._cell_index = 0
        self._extend_track()

    def _extend_track(self) -> None:
        """Ensure at least ``lookahead`` cells exist ahead of the player."""
        want = self._cell_index + int(self.cfg["lookahead"]) + 1
        while len(self._track) < want:
            self._track.append(self._next_cell())

    def _next_cell(self) -> dict:
        if self.track_source == "replay":
            return self._next_cell_replay()
        e = self._next_cell_procedural()
        # track the previous HAZARD for pair_same_type_prob. Done here, at the single point every
        # generated cell passes through, so the crevasse / pending-gap / compound early returns are
        # all covered. Multi-cell structures keep resetting the counter, which is correct: the gap is
        # measured from the END of the structure, matching how the device pairs were counted.
        _o = int(e.get("obstacles", 0))
        if _o in _MUST_JUMP or _o in _MUST_SLIDE:
            self._last_obst_type = _o
            self._cells_since_hazard = 0
        else:
            self._cells_since_hazard += 1
        return e

    def _peek_next(self) -> dict:
        """The already-generated next cell (what the obs's ``next_*`` fields reflect). NOT ``_next_cell()``
        — that GENERATES a fresh cell (advancing RNG + spacing counters); peeking with it corrupts the
        track. ``_extend_track`` keeps ``lookahead+1`` cells ahead so index+1 normally exists."""
        ni = self._cell_index + 1
        return self._track[ni] if ni < len(self._track) else {}

    def _next_cell_replay(self) -> dict:
        track = self._injected_track or []
        if self._replay_cursor < len(track):
            cell = dict(track[self._replay_cursor])
            self._replay_cursor += 1
            return cell
        # ran out of recorded track: feed neutral straight cells (the episode usually ended).
        return _element(type=self.cfg["path_type_straight"], level=self.cfg["default_path_level"])

    def _difficulty(self) -> float:
        """0 at run start, 1 by ``ramp_distance`` — drives the turn-density ramp (real game
        densifies turns ~8.5/1k -> ~21/1k over distance 0 -> 6k)."""
        d = self._distance_real - float(self.cfg["start_distance"])
        return float(np.clip(d / max(1.0, float(self.cfg["ramp_distance"])), 0.0, 1.0))

    def _queue_compound(self, lane_a: int, level: int, strip_a: int) -> int:
        """Build a STACKED compound sequence (user-observed blindspot) into _pending_sequence: the
        REST of narrow A (same lane) -> a jump-barrier -> a clear landing -> narrow B on the OPPOSITE
        side. Survivable only by strafing A->B MID-AIR over the barrier. Returns lane_a (the current
        cell = narrow A cell 1). The barrier + narrow B bypass min_space (that's the point)."""
        cfg = self.cfg
        seq: list[dict] = []
        # rest of narrow A
        for _ in range(max(0, strip_a - 1)):
            e = _element(type=self._narrow_ptype(lane_a), level=level, is_turn=False)
            e["narrow_lane"] = lane_a
            seq.append(e)
        # the jump-barrier (TreeJump: must-jump, has solid ground so you LAND after) — centered
        seq.append(_element(type=2, level=level, obstacles=OBST_TREE_JUMP, is_turn=False))
        # clear landing cell(s) after the barrier (where the mid-air strafe touches down toward B)
        for _ in range(max(1, int(cfg.get("compound_land_cells", 1)))):
            seq.append(_element(type=2, level=level, is_turn=False))
        # narrow B on the OPPOSITE side
        lane_b = -lane_a
        lw = cfg["narrow_len_weights"]
        lens = list(lw.keys())
        probs = np.array([lw[k] for k in lens], dtype=float)
        strip_b = int(self._rng.choice(lens, p=probs / probs.sum()))
        for _ in range(max(1, strip_b)):
            e = _element(type=self._narrow_ptype(lane_b), level=level, is_turn=False)
            e["narrow_lane"] = lane_b
            seq.append(e)
        self._pending_sequence.extend(seq)
        return lane_a

    def _queue_jumpslide(self, level: int) -> None:
        """Build a JUMP->SLIDE tail into _pending_post_gap: a few clear cells then a SLIDE obstacle,
        placed after the current wide-gap so it lands WITHIN the jump's airtime -> the runner must jump
        the gap EARLY (land before the slide) or die airborne (dt=3/5). Device death @4991/3916.
        Bypasses min_space (that's the point). Only ever called when jumpslide_prob>0 (rng-gated)."""
        cfg = self.cfg
        seq: list[dict] = []
        for _ in range(max(0, int(cfg.get("jumpslide_gap_cells", 1)))):
            seq.append(_element(type=int(cfg["path_type_straight"]), level=level, is_turn=False))
        is_flame = self._rng.random() < float(cfg.get("jumpslide_flame_frac", 0.5))
        sob = OBST_FLAME_TOWER if is_flame else OBST_TREE_SLIDE
        spt = 31 if is_flame else int(cfg["path_type_straight"])  # flame = pt31; tree-slide sits on a straight
        seq.append(_element(type=spt, level=level, obstacles=sob, is_turn=False))
        self._pending_post_gap.extend(seq)

    def _queue_jumpnarrow(self, level: int) -> None:
        """Build a JUMP->STRAFE tail into _pending_post_gap: a few clear cells then a SIDE-NARROW strip
        (lane +/-1) -> the runner must strafe into the narrow's lane during/right after the jump, or fall
        off (dt=1). Teaches EARLY strafe-after-jump. Device death @5148/3281/2793. Only when prob>0."""
        cfg = self.cfg
        seq: list[dict] = []
        for _ in range(max(0, int(cfg.get("jumpnarrow_gap_cells", 1)))):
            seq.append(_element(type=int(cfg["path_type_straight"]), level=level, is_turn=False))
        lane = int(self._rng.choice([-1, 1]))
        strip = max(1, int(cfg.get("jumpnarrow_strip_cells", 2)))
        for _ in range(strip):
            e = _element(type=self._narrow_ptype(lane), level=level, is_turn=False)
            e["narrow_lane"] = lane
            seq.append(e)
        self._pending_post_gap.extend(seq)

    def _queue_jumpgap(self, level: int) -> None:
        """Build a GAP->GAP tail into _pending_post_gap: a couple clear cells then a 2-cell HoleLong
        (pt12, wide_gap) — the high-ground DOUBLE-JUMP death scenario (device 2026-07-08: 35% of eps end
        within 15 frames of a landing-frame re-jump; holes packed 2-3 cells apart on high ground, while the
        sim's min_space 6 made the blind re-jump STRUCTURALLY harmless -> 0/47 sim lethality, no learning
        signal). A blind re-jump right after landing from gap-1 lands INSIDE gap-2 (wide-gap landing model
        -> dt4 Ledge, the exact device signature) or arrives grounded past the lip (refusal -> Fall);
        land -> run the clear cells -> deliberate jump at the lip CLEARS. Only when jumpgap_prob>0."""
        cfg = self.cfg
        # RHYTHM CHAIN [DEVICE 2026-07-08, ab_v62_r1 ep0 @794 / ab_v60_r2 ep4 @1481]: high-ground zones
        # chain long-holes every ~4 cells (hole x2 -> clear x2 -> hole x2 ...) — the documented multi-hole
        # RHYTHM wall. One tail trains ONE repetition; the device demands a land->run->jump METRONOME over
        # 3+. chain_max=1 (default) = single tail, NO extra rng draw (v62 stream untouched); >1 draws the
        # repeat count uniform 1..chain_max.
        n_rep = 1
        chain_max = int(cfg.get("jumpgap_chain_max", 1))
        if chain_max > 1:
            n_rep = 1 + int(self._rng.integers(0, chain_max))
        seq: list[dict] = []
        for _ in range(n_rep):
            for _ in range(max(0, int(cfg.get("jumpgap_gap_cells", 2)))):
                seq.append(_element(type=int(cfg["path_type_straight"]), level=level, is_turn=False))
            n_hole = max(1, int(cfg.get("jumpgap_hole_cells", 2)))
            for _ in range(n_hole):
                # hole_cells==1 -> mirror NORMAL generation (OBST_HOLE on a straight cell, NON-wide) =
                # the device-true form of the tight LevelChange->Hole pair (WAVE 14, n=21, the most
                # common inside-the-arc pair); >1 -> pt12 HoleLong + wide_gap, the v62 gap->gap form.
                # Same rule slidegap already uses. Default 2 -> unchanged (equivalence-safe).
                e = _element(type=(12 if n_hole > 1 else int(cfg["path_type_straight"])),
                             level=level, obstacles=OBST_HOLE, is_turn=False)
                e["wide_gap"] = n_hole > 1
                seq.append(e)
        self._pending_post_gap.extend(seq)

    def _queue_slidegap(self, level: int) -> None:
        """Build a SLIDE->HOLE tail into _pending_post_gap: a couple clear cells then a HOLE, placed
        after a SLIDE obstacle. The ~0.7s slide commit (jump refused while sliding) eats the jump window
        unless the policy slides EARLY and jumps the INSTANT the slide ends — the human's survival recipe
        (12/14 events, takeoff -0.33..+0.07s rel. hole entry; 2 deaths = the swipe came mid-fall).
        Bypasses min_space (that's the point). Only ever called when slidegap_prob>0 (rng-gated)."""
        cfg = self.cfg
        seq: list[dict] = []
        for _ in range(max(0, int(cfg.get("slidegap_gap_cells", 2)))):
            seq.append(_element(type=int(cfg["path_type_straight"]), level=level, is_turn=False))
        n_hole = max(1, int(cfg.get("slidegap_hole_cells", 1)))
        for _ in range(n_hole):
            # single hole mirrors normal generation exactly (OBST_HOLE on a straight cell, non-wide);
            # hole_cells>1 -> a HoleLong (pt12 + wide_gap, landing model applies) like jumpgap.
            e = _element(type=(12 if n_hole > 1 else int(cfg["path_type_straight"])),
                         level=level, obstacles=OBST_HOLE, is_turn=False)
            e["wide_gap"] = n_hole > 1
            seq.append(e)
        self._pending_post_gap.extend(seq)

    @staticmethod
    def _narrow_ptype(lane: int) -> int:
        return 19 if lane == -1 else 22

    def _ramp(self):
        """Current row of the GAME'S OWN difficulty ramp, or None when the ramp is off.

        Returns (min_space_obstacles, min_space_turns, double_obstacle_pct, narrows_on, flame_slide_on).
        Device-read 2026-07-26 (hook 17j): the real game tightens min_space 6->2 and turn spacing
        12->2 with distance while the sim held them flat, which is why the sim was ~1.5x too easy and
        could not produce the tight hazard pairs that dominate device deaths past ~1.5 km.
        """
        if not self.cfg.get("device_difficulty_ramp"):
            return None
        row = None
        for r in self.cfg["difficulty_ramp_table"]:
            if self._distance_real >= float(r[0]):
                row = r
            else:
                break
        return row[1:] if row else None

    def _next_cell_procedural(self) -> dict:
        cfg = self.cfg
        level = int(self._zone_level)
        self._cells_since_obstacle += 1
        self._cells_since_turn += 1
        self._cells_since_bonus += 1
        self._cells_since_zone += 1

        # COMPOUND-SEQUENCE emit (user-observed stacked blindspot): a pre-built list of cells
        # (narrow A -> barrier -> narrow B-opposite) that bypasses the normal spacing so the policy must
        # MID-AIR STRAFE across the barrier. Takes priority over all normal generation while non-empty.
        if self._pending_sequence:
            self._cells_since_obstacle = 0
            return self._pending_sequence.pop(0)

        # WIDE-GAP continuation: the 2nd+ cell of a 2-cell gap (long hole / zone change). A single
        # well-timed jump must span both -> jumping a cell early lands in this far half and falls.
        if self._pending_gap > 0:
            self._pending_gap -= 1
            self._cells_since_obstacle = 0
            e = _element(type=self._pending_gap_type, level=self._pending_gap_level,
                         obstacles=self._pending_gap_obst, is_turn=False)
            e["wide_gap"] = True
            e["stepdown_low"] = bool(getattr(self, "_pending_gap_stepdown_low", False))
            # the LAST continuation cell of a zone crevasse is the RAISED LEDGE (apex check fires here)
            if self._pending_gap == 0 and getattr(self, "_pending_gap_stepup", False):
                e["stepup"] = True
                self._pending_gap_stepup = False
            return e
        # JUMP->SLIDE tail: a slide obstacle a few clear cells after a wide-gap (emits AFTER the gap
        # continuation so the crevasse-jump airtime overlaps the slide -> airborne-at-slide death).
        if self._pending_post_gap:
            self._cells_since_obstacle = 0
            return self._pending_post_gap.pop(0)
        # NARROW-strip continuation: the 2nd+ cell of a multi-cell narrow (same safe lane). Each cell
        # re-applies the off-lane Fall on entry, so the policy must HOLD the safe lane across the strip.
        if self._pending_narrow > 0:
            self._pending_narrow -= 1
            self._cells_since_obstacle = 0
            e = _element(type=self._pending_narrow_type, level=int(self._zone_level), is_turn=False)
            e["narrow_lane"] = self._pending_narrow_lane
            return e
        wide = False         # this cell STARTS a 2-cell gap
        stepup = False       # this cell is a level-UP crevasse (raised far wall -> jump must be LATE)
        stepdown_low = False  # this cell is a step-DOWN to low ground -> crossable on foot (device-true)
        narrow_lane = None   # if set (-1/0/+1), this cell is a narrow needing that lane (else Fall)

        # ★ THE GAME'S OWN RAMP (device-read 2026-07-26). When on, the measured schedule REPLACES the
        # flat constants + the lerp below: spacing, turn spacing, the double-obstacle rate and the
        # narrows/flame-slide gates all come straight from the table. See DEFAULT_CONFIG.
        ramp = self._ramp()
        # Difficulty-ramped turn spacing/probability (lerp baseline -> late by distance).
        diff = self._difficulty()
        min_turns = (float(ramp[1]) if ramp else
                     (float(cfg["min_space_between_turns"]) * (1.0 - diff)
                      + float(cfg["min_space_between_turns_late"]) * diff))
        turn_p = (float(cfg["turn_prob"]) * (1.0 - diff)
                  + float(cfg["turn_prob_late"]) * diff)

        # --- junction / turn decision (gated by min_space_between_turns) -----
        is_turn = False
        can_l = False
        can_r = False
        ptype = int(cfg["path_type_straight"])
        if self._cells_since_turn >= min_turns:
            if self._rng.random() < turn_p:
                is_turn = True
                self._cells_since_turn = 0
                roll = self._rng.random()
                if roll < 0.4:
                    can_l, can_r = True, False
                    ptype = int(cfg["path_type_turn"])
                elif roll < 0.8:
                    can_l, can_r = False, True
                    ptype = int(cfg["path_type_turn"])
                else:
                    can_l, can_r = True, True       # T-junction: both sides
                    ptype = int(cfg["path_type_t_junction"])

        # ZONE TRANSITION (altitude change): every ~zone_change_every cells the track steps to an
        # adjacent altitude zone; that cell is a LevelChange CREVASSE (must-jump). Takes priority over
        # a normal obstacle. Kept off turn cells + out of the post-junction clear window (unclearable).
        obstacle = OBST_NONE
        if (not is_turn
                and self._cells_since_turn >= int(cfg["min_clear_after_turn"])
                and self._cells_since_obstacle >= (ramp[0] if ramp
                                                   else int(cfg["min_space_between_obstacles"]))
                and self._cells_since_zone >= int(cfg["zone_change_every_cells"])
                and self._distance_real >= float(cfg.get("zone_after_distance", 0.0))):
            obstacle = OBST_LEVEL_CHANGE
            self._cells_since_zone = 0
            self._cells_since_obstacle = 0
            self._b2b_obstacles = 0
            lo, hi = int(cfg["zone_min_level"]), int(cfg["zone_max_level"])
            prev_zone = self._zone_level
            if self._zone_level <= lo:
                self._zone_level += 1
            elif self._zone_level >= hi:
                self._zone_level -= 1
            else:
                # [AUDIT 2026-06-26] real spends ~32% low / 13% high (biased LOW); a symmetric walk gave
                # 22/23. Bias the mid-step DOWN so the stationary time matches (down 0.65 / up 0.35).
                self._zone_level += -1 if self._rng.random() < 0.65 else 1
            level = int(self._zone_level)   # the crevasse cell already sits in the NEW zone (matches live capture)
            # ZONE crevasse [DEVICE-MEASURED 2026-07-03, v13_16i death frames]: a 2-CELL structure —
            # cell 1 = the GAP, cell 2 = the RAISED LEDGE (ground_height jumps to 15 there; the runner
            # dies dt=4 kDeathTypeLedge AT the 2nd cell while airborne but not high enough). So the APEX
            # check must fire on the 2nd cell (the ledge), and the runner must reach it near the jump
            # apex. Model it as a 2-cell gap: cell1 = plain gap (airborne needed), cell2 = the ledge
            # (apex check, flagged `stepup`).
            # ONE-CELL CREVASSES [DEVICE-MEASURED WAVE 14, 2026-07-25, n=409 LevelChange structures
            # over the v70/v74/v78 captures]: **67.5% of real crevasses are ONE cell**, only 32.5%
            # are two — the sim generated 100% two-cell. That extra cell pushes everything downstream
            # one cell further out, BEYOND the ~2.5-3.3-cell jump arc, so the AIRBORNE TRAP (reach the
            # next obstacle still descending from the crevasse jump) was geometrically UNREACHABLE
            # in-sim while it is ~80% of the device hole deaths. A 1-cell crevasse is the cap cell
            # itself and carries the apex check directly. Gated prob>0-first (0 = no rng draw ->
            # the 2-cell-only stream is untouched -> equivalence baseline safe).
            _p1 = float(cfg.get("crevasse_single_cell_prob", 0.0))
            self._pending_gap_obst = OBST_LEVEL_CHANGE
            self._pending_gap_type = 35    # ~ real SingleLevelStepDownCap path_type (obs match)
            self._pending_gap_level = int(self._zone_level)
            ptype = 35
            if _p1 > 0.0 and self._rng.random() < _p1:
                # ONE-cell crevasse. DEVICE-MEASURED OUTCOME (experiments/crevasse_cells_outcome.py,
                # n=827 structures): the 1-cell form is ~6x SAFER than the 2-cell one — 1.1% deaths
                # and ZERO grazes vs 7.0% / 33.7%. So it is a plain short gap: airborne-at-entry is
                # the whole requirement, with NO raised-ledge apex check and no cap graze. Modelled
                # by leaving it NON-wide: `wide` gates the apex/graze branch, the wide-gap landing
                # model and the gap-lip jump refusal, all of which are 2-cell-ledge physics. It stays
                # must-jump (OBST_LEVEL_CHANGE is in _MUST_JUMP) so the runner must still clear it.
                self._pending_gap = 0
                stepup = False
                wide = False
            else:
                self._pending_gap = 1      # -> 1 continuation cell = the raised ledge
                self._pending_gap_stepup = bool(self._zone_level > prev_zone)  # ledge cell = step-UP
                stepup = False             # cell 1 is the GAP (airborne), not the ledge; ledge is cell 2
                wide = True
            # STEP-DOWN TO LOW GROUND: device-measured survivable on foot (see stepdown_grounded_ok).
            # Marked on every cell of the structure so both the gap and the ledge accept a grounded
            # crossing. Direction is decided here, where prev_zone is still in scope.
            stepdown_low = bool(
                self.cfg.get("stepdown_grounded_ok")
                and self._zone_level < prev_zone
                and self._zone_level <= int(self.cfg.get("stepdown_grounded_max_level", 1)))
            self._pending_gap_stepdown_low = stepdown_low

        # POST-TURN OBSTACLE [DEVICE-MEASURED 2026-07-08, 72 must-jump/slide onsets ~1 cell after a
        # junction across 6 captures]: the real game DOES place obstacles on the first post-junction cell;
        # min_clear_after_turn(2) meant the sim NEVER did -> the policy exits turns blind (obs null through
        # junctions) and untrained -> 7/15 of the pt(-1/1/2) deaths follow a turn by <12 frames (ends the
        # 4-7k runs). With prob, place a standard obstacle here (a HOLE stays 1-cell; a wide long-hole one
        # cell after a corner would be unclearable), bypassing min_clear/min_space. b2b deliberately NOT
        # bumped (the elif below resets it — mirrors the torch nonturn_reset). Gated prob>0 FIRST (no rng
        # draw at default 0 -> stream/equivalence safe).
        if (obstacle == OBST_NONE and not is_turn and self._cells_since_turn == 1
                and float(cfg.get("turnobstacle_prob", 0.0)) > 0.0
                and self._rng.random() < float(cfg.get("turnobstacle_prob", 0.0))):
            obstacle = self._sample_obstacle_type_paired()
            self._cells_since_obstacle = 0
        # ★ THE DEVICE CONCENTRATES POST-TURN HAZARDS AT +3, NOT +1 [DEVICE-MEASURED 2026-07-27 night,
        # experiments/postturn_reconcile.py over 1446 junctions, hazard = must-jump OR must-slide,
        # any-frame cell attribution]:
        #     offset   +1     +2     +3     +4     +5
        #     device  3.2%   9.5%  36.6%  18.9%   9.4%
        #     sim     7.4%   0.0%  11.2%   8.1%   7.6%   (v107: min_clear_after_turn 3, turnobst .08)
        # More than one junction in THREE is followed by a hazard exactly 3 cells later -- that is the
        # user's v107 death ("it took a turn with an obstacle... you have to jump before the obstacle
        # or during the turn"), and turnobstacle_prob cannot produce it because it only ever fills +1.
        # A separate offset-3 dose is therefore its own knob, not a re-dose of the existing one.
        # ⚠️ The +2 hole was NOT a device fact: wave-15 measured device +2 at 0.7% and set
        # min_clear_after_turn 2 -> 3 on that basis, but that 0.7% is an artifact of first-frame cell
        # attribution (`obst.setdefault`) against WINDOW-PRESENTED device flags. Re-measured any-frame
        # it is 5.3-9.5% on the SAME captures. Both capture eras agree at every definition level, so
        # this is methodology, not a change in the game.
        elif (obstacle == OBST_NONE and not is_turn and self._cells_since_turn == 3
                and float(cfg.get("turnobstacle3_prob", 0.0)) > 0.0
                and self._rng.random() < float(cfg.get("turnobstacle3_prob", 0.0))):
            obstacle = self._sample_obstacle_type_paired()
            self._cells_since_obstacle = 0

        # A turn cell carries no obstacle (the player must turn here); the cell right after a junction
        # is also kept clear (min_clear_after_turn) — a must-jump obstacle there is unclearable (you
        # can't turn AND jump on one crossing).
        # TIGHT PAIRS [DEVICE-MEASURED WAVE 14, n=2684 consecutive must-jump pairs over the v66-v78
        # captures]: the real game places 19% of them CLOSER than the sim's HARD min_space floor of 6
        # (gap 1:23, 2:49, 3:95, 4:74, 5:272) and 6.2% within the ~2.5-3.3-cell jump arc (<=3 cells).
        # The sim's floor made that impossible outside the bespoke compound tails, which only ever
        # produce gap 3-4 -- one cell too generous to form the airborne trap. With prob, allow a
        # placement at a SHORT spacing. Gated prob>0-first (default 0 -> no rng draw -> the track
        # stream and equivalence baseline are untouched).
        _tp = float(cfg.get("tight_pair_prob", 0.0))
        _tight = (_tp > 0.0 and obstacle == OBST_NONE and not is_turn
                  and self._cells_since_turn >= int(cfg["min_clear_after_turn"])
                  and int(cfg.get("tight_pair_min_cells", 1)) <= self._cells_since_obstacle
                  < int(cfg["min_space_between_obstacles"])
                  and self._rng.random() < _tp)
        if (obstacle == OBST_NONE and not is_turn
                and self._cells_since_turn >= int(cfg["min_clear_after_turn"])
                and (self._cells_since_obstacle >= (ramp[0] if ramp
                                                    else int(cfg["min_space_between_obstacles"]))
                     or _tight or self._pending_double)):
            place = self._rng.random() < float(cfg["obstacle_prob"])
            # honor max back-to-back: if we've hit the cap, force a gap.
            if self._b2b_obstacles >= int(cfg["max_back_to_back_obstacles"]):
                place = False
            if place:
                self._cells_since_obstacle = 0
                self._b2b_obstacles += 1
                # arm the DOUBLE: prob>0-first so the default (0) draws no rng and leaves the stream
                # untouched. The cap is checked when it is consumed, mirroring the device's b2b limit.
                _dbl = (ramp[2] if ramp else float(cfg.get("double_obstacle_percent", 0.0)))
                self._pending_double = bool(
                    _dbl > 0.0
                    and self._b2b_obstacles < int(cfg["max_back_to_back_obstacles"])
                    and self._rng.random() < _dbl)
                _narrows_ok = (ramp[3] if ramp
                               else self._distance_real >= float(cfg["narrows_after_distance"]))
                if (_narrows_ok and self._rng.random() < float(cfg["narrow_prob"])):
                    # NARROW: a multi-cell one-lane SIDE strip with a RANDOM safe lane; runner falls off
                    # elsewhere. The policy must read narrow_target_lane (not the subtype) and HOLD it
                    # across the strip. [DEVICE-MEASURED 2026-07-02, Region2D bands narrow_calib2]: EVERY
                    # device narrow band is a SIDE strip (center +/-4-5), NEVER center — so the sim only
                    # generates side lanes (-1/+1). (Was -1/0/+1; the 0/center-safe narrow does not exist
                    # on device -> the policy learned a center-safe narrow that always fell.)
                    narrow_lane = int(self._rng.choice([-1, 1]))    # -1 (left strip) / +1 (right strip)
                    ptype = {-1: 19, 1: 22}[narrow_lane]
                    # device-measured strip length -> queue the continuation cells (same safe lane).
                    lw = cfg["narrow_len_weights"]
                    lens = list(lw.keys())
                    probs = np.array([lw[k] for k in lens], dtype=float)
                    strip = int(self._rng.choice(lens, p=probs / probs.sum()))
                    # COMPOUND (user blindspot): with compound_prob turn this narrow into a stacked
                    # narrow(A) -> barrier -> narrow(B opposite) that needs a MID-AIR strafe. The current
                    # cell is narrow A cell 1 (narrow_lane set below); _queue_compound enqueues the REST
                    # of the whole sequence into _pending_sequence (bypasses _pending_narrow entirely).
                    # short-circuit on compound_prob>0 FIRST so the default (prob 0) path draws NO rng
                    # here — else the extra random() shifts the whole track stream (scorecard velocity).
                    if (float(cfg.get("compound_prob", 0.0)) > 0.0
                            and self._distance_real >= float(cfg.get("compound_after_distance", 0.0))
                            and self._rng.random() < float(cfg.get("compound_prob", 0.0))):
                        narrow_lane = self._queue_compound(narrow_lane, level, strip)
                        ptype = {-1: 19, 1: 22}[narrow_lane]
                    else:
                        self._pending_narrow = strip - 1
                        self._pending_narrow_lane = narrow_lane
                        self._pending_narrow_type = ptype
                else:
                    obstacle = self._sample_obstacle_type_paired()
                    # a HOLE is sometimes a 2-cell LONG hole (path_type 12/13/15/16 on device).
                    if obstacle == OBST_HOLE and self._rng.random() < float(cfg["wide_gap_prob"]):
                        self._pending_gap = int(cfg["wide_gap_extra_cells"])
                        self._pending_gap_obst = OBST_HOLE
                        self._pending_gap_type = 12   # ~ real HoleLong path_type (obs match)
                        self._pending_gap_level = level
                        # this pending slot is now a HOLE, not a crevasse: clear the crevasse-only
                        # step-down mark so it cannot leak onto the hole's continuation cell.
                        self._pending_gap_stepdown_low = False
                        ptype = 12
                        wide = True
            else:
                self._b2b_obstacles = 0
        elif not is_turn:
            self._b2b_obstacles = 0

        # JUMP->SLIDE / JUMP->STRAFE compounds: if this cell STARTED a wide-gap (crevasse/long-hole),
        # sometimes place a SLIDE obstacle (jumpslide) OR a side-NARROW (jumpnarrow) a couple clear cells
        # later -> airborne-at-slide death (@4991/3916) / strafe-too-late narrow fall (@5148/3281/2793). Each
        # rng gated on its prob>0 FIRST so the default (0) draws NO rng -> stream/equivalence safe. A wide-gap
        # gets AT MOST one tail (slide rolled first; narrow only if slide didn't fire).
        queued_tail = False
        if (wide and float(cfg.get("jumpslide_prob", 0.0)) > 0.0
                and self._distance_real >= float(cfg.get("jumpslide_after_distance", 0.0))
                and self._rng.random() < float(cfg.get("jumpslide_prob", 0.0))):
            self._queue_jumpslide(level)
            queued_tail = True
        if (wide and not queued_tail and float(cfg.get("jumpnarrow_prob", 0.0)) > 0.0
                and self._distance_real >= float(cfg.get("jumpnarrow_after_distance", 0.0))
                and self._rng.random() < float(cfg.get("jumpnarrow_prob", 0.0))):
            self._queue_jumpnarrow(level)
            queued_tail = True
        # GAP->GAP compound (the high-ground double-jump death): another 2-cell hole a couple clear cells
        # after this wide-gap. Same prob>0-first rng gating; at most one tail per wide-gap.
        if (wide and not queued_tail and float(cfg.get("jumpgap_prob", 0.0)) > 0.0
                and self._distance_real >= float(cfg.get("jumpgap_after_distance", 0.0))
                and self._rng.random() < float(cfg.get("jumpgap_prob", 0.0))):
            self._queue_jumpgap(level)
            queued_tail = True
        # SLIDE->HOLE compound [WAVE 9 HUMAN-DEMO]: a hole a couple clear cells after a SLIDE obstacle —
        # the slide commit eats the jump window (killed the human twice; never generated -> free in-sim).
        # Same prob>0-first rng gating; only fires on a MUST-SLIDE obstacle cell (not wide-gaps).
        if (obstacle in _MUST_SLIDE and not queued_tail
                and float(cfg.get("slidegap_prob", 0.0)) > 0.0
                and self._distance_real >= float(cfg.get("slidegap_after_distance", 0.0))
                and self._rng.random() < float(cfg.get("slidegap_prob", 0.0))):
            self._queue_slidegap(level)

        # --- coin / bonus cell (gated) --------------------------------------
        is_coin = False
        if (not is_turn and obstacle == OBST_NONE
                and self._cells_since_bonus >= int(cfg["min_spaces_between_bonus"])):
            if self._rng.random() < 0.5:
                is_coin = True
                self._cells_since_bonus = 0

        e = _element(type=ptype, level=level, obstacles=obstacle, is_turn=is_turn,
                     can_turn_left=can_l, can_turn_right=can_r, is_coin=is_coin)
        e["wide_gap"] = wide
        e["stepup"] = stepup
        e["stepdown_low"] = stepdown_low
        e["narrow_lane"] = narrow_lane
        return e

    def _type_reject_table(self) -> dict:
        """Parsed ``{(prev, cur): p_reject}``, cached on the identity of the raw config value.

        ``self.cfg`` is re-derived from ``_base_cfg`` on every reset but ``type_reject`` is not a
        domain-randomised key, so the raw object is stable and the parse happens once per env.
        """
        raw = self.cfg.get("type_reject") or _EMPTY_TYPE_REJECT
        if getattr(self, "_tr_raw", None) is not raw:
            self._tr_raw = raw
            self._tr_tbl = parse_type_reject(raw)
        return self._tr_tbl

    def _sample_obstacle_type_reject(self) -> int:
        """DRAW-THEN-SUBSTITUTE, the game's own shape (see ``type_reject_model``).

        Stage A — a GEOMETRY type (Hole / Narrow / FlameTower) is tested against the previously
        emitted hazard type by the pairwise admissibility rule; on rejection it is re-drawn
        (`OriginalPick` -> `FinalPick`). ONE re-draw by default: that single substitution is what
        the game models, and it is what makes the suppression FLOOR reachable rather than zero.

        Stage B — a TREE is returned untouched, because the game places trees in a later pass that
        the filter never sees. This is the half that produces the elevated slide clustering for
        free, and it is why the rule is not a blanket anti-repeat.

        RNG DRAW ORDER (the whole risk; `cpp_sim/sim.cpp:sample_obstacle_type_reject` must match
        value-for-value): one draw for the pick, then per re-draw round at most one rejection roll
        and at most one further pick. The rejection roll is skipped entirely when the pair carries
        no rejection probability, mirroring the house "check prob > 0 FIRST" rule, so a table that
        does not mention the drawn pair costs exactly the same draws as the i.i.d. sampler.
        """
        t = self._sample_obstacle_type()          # OriginalPick
        prev = int(self._last_obst_type)
        if not prev:
            return t                              # nothing emitted yet -> no pair to judge
        tbl = self._type_reject_table()
        for _ in range(max(0, int(self.cfg.get("type_reject_redraws", 1)))):
            if t in _TREE_TYPES:
                break                             # Stage B: the second pass is unfiltered
            p = tbl.get((prev, t), 0.0)
            if p <= 0.0:
                break                             # admissible pair -> no roll, no draw
            if self._rng.random() >= p:
                break                             # accepted
            t = self._sample_obstacle_type()      # FinalPick (the substitute)
        return t

    def _sample_obstacle_type_paired(self) -> int:
        """Draw an obstacle type, sometimes REPEATING the previous one (see pair_same_type_prob).

        Device-measured clustering: repeated slides are the game's two most common consecutive
        orderings. Gated prob>0-FIRST so the default draws no rng and the stream is untouched.
        """
        # ★ MODEL SELECT, and it is checked BEFORE any random(): with type_reject_model False this
        # costs zero draws, so every pre-2026-07-28 track stream and calibration is reproduced
        # bit-for-bit.
        #
        # ⚠️ CORRECTED 2026-07-28: the first spec said this model REPLACES pair_same_type_prob. That
        # was WRONG, and measurement caught it -- turning the model on DROPPED slide clustering from
        # 7.9% to 3.8% (device 17.9%). The two are STAGE A and STAGE B of the same generator, not
        # competing models:
        #   Stage A  geometry (Hole/Narrow/Flame) -- drawn, then pairwise-REJECTED and substituted.
        #            Rejection can only ever SUPPRESS, which is why it reproduces the device's
        #            Hole->Hole 6.1% (below its 10.3% independent rate) and nothing else.
        #   Stage B  trees (Stumble/Jump/Slide) -- placed by the game's separate AddObstacle pass,
        #            which the pairwise filter never sees. The device's TreeSlide->TreeSlide is 9.8%
        #            against a 1.5% independent rate: 6.5x ELEVATED. No rejection rule can raise a
        #            rate, so that elevation must come from repetition inside the tree pass, which
        #            is exactly what pair_same_type_prob models.
        # So Stage A runs first, then Stage B may override with a repeat. Draw order is fixed and
        # mirrored value-for-value by cpp_sim/sim.cpp.
        if self.cfg.get("type_reject_model"):
            t = self._sample_obstacle_type_reject()
            p = float(self.cfg.get("pair_same_type_prob", 0.0))
            if (p > 0.0 and self._last_obst_type
                    and int(self._last_obst_type) in tuple(self.cfg.get("pair_same_type_types", (5, 6)))
                    and self._cells_since_hazard <= int(self.cfg.get("pair_same_type_cells", 4))
                    and self._rng.random() < p):
                return int(self._last_obst_type)
            return t
        p = float(self.cfg.get("pair_same_type_prob", 0.0))
        # SLIDES ONLY by default. Type-agnostic repetition is the WRONG model: on device 73% of all
        # same-type pairs are repeated slides (TreeSlide->TreeSlide 9.8%, Flame->Flame 8.1%), which is
        # 6.5x and 2.3x their independent rates -- while Hole->Hole is 6.1% against an independent
        # 10.3%, i.e. holes repeat LESS than chance. Repeating every type overshoots total same-type
        # (28.3% vs the device's 24.6%) long before the slide rhythm gets anywhere near 17.9%.
        _types = self.cfg.get("pair_same_type_types", (5, 6))
        if (p > 0.0 and self._last_obst_type and int(self._last_obst_type) in tuple(_types)
                and self._cells_since_hazard <= int(self.cfg.get("pair_same_type_cells", 4))
                and self._rng.random() < p):
            return int(self._last_obst_type)
        t = self._sample_obstacle_type()
        # ★ ANTI-REPETITION FOR JUMP OBSTACLES [2026-07-28]. The comment above records the other half
        # of the same device measurement and nothing ever implemented it: Hole->Hole is 6.1% on
        # device against an INDEPENDENT 10.3%, i.e. the game actively AVOIDS repeating a hole, while
        # the sim draws types i.i.d. and repeats them at chance. Consequence, measured on 148 sim
        # chains: `Hole->Hole->Hole` is the sim's most common 3-chain (n=15) and does not appear in
        # the device's top five at all -- so the policy over-trains jump-land-jump and under-trains
        # the slide-led chains the device actually serves (50.7% vs the sim's 39.5%).
        # Suppressing 41% of hole-after-hole draws takes 10.3% -> 6.1%, so the dose is READ OFF THE
        # DEVICE, not fitted. Gated prob>0-FIRST so the default draws no rng and the stream is
        # bit-identical when off.
        q = float(self.cfg.get("pair_anti_repeat_prob", 0.0))
        if (q > 0.0 and self._last_obst_type and int(t) == int(self._last_obst_type)
                and int(t) in tuple(self.cfg.get("pair_anti_repeat_types", (1, 4)))
                and self._cells_since_hazard <= int(self.cfg.get("pair_same_type_cells", 4))
                and self._rng.random() < q):
            return self._sample_obstacle_type()      # one re-draw; may legitimately land on t again
        return t

    def _sample_obstacle_type(self) -> int:
        weights = dict(self.cfg["obstacle_type_weights"])
        # Slides (TreeSlide=5, FlameTower=6) only appear past slides_after_distance — the baseline
        # statics have sAllowTreeSlide/FlameTowers = False, so early runs are jump/stumble only.
        # With the device ramp on, the game's OWN sAllowTreeSlide/sAllowFlameTowers gate decides
        # (device-read: both switch on at ~507 m), instead of the hand-calibrated distance constant.
        _ramp = self._ramp()
        _slides_ok = _ramp[4] if _ramp else (self._distance_real >= float(self.cfg["slides_after_distance"]))
        if not _slides_ok:
            weights = {t: w for t, w in weights.items() if t not in (5, 6)}
        types = list(weights.keys())
        probs = np.array([weights[t] for t in types], dtype=float)
        probs = probs / probs.sum()
        return int(self._rng.choice(types, p=probs))

    # ------------------------------------------------------------- gym: reset
    def _sample_domain(self) -> None:
        """Per-episode domain randomization: re-derive ``self.cfg`` from the base and jitter the
        sim-to-real-sensitive knobs (turn window, durations, velocity, density). No-op if off."""
        self.cfg = dict(self._base_cfg)
        self._vel_scale = 1.0
        if not self._dr:
            return
        r = self._rng
        c = self.cfg

        def U(key):
            lo, hi = self._base_cfg[key]
            return float(r.uniform(lo, hi))
        c["turn_window_frac"] = U("dr_turn_window_frac")
        c["turn_early_frac"] = U("dr_turn_early_frac")
        c["turn_late_frac"] = U("dr_turn_late_frac")
        c["jump_duration"] = U("dr_jump_duration")
        c["slide_duration"] = U("dr_slide_duration")
        c["obstacle_prob"] = U("dr_obstacle_prob")
        c["cell_length"] = U("dr_cell_length")
        tp = U("dr_turn_prob")
        c["turn_prob"] = tp
        c["turn_prob_late"] = max(tp, float(self._base_cfg["turn_prob_late"]))
        self._vel_scale = U("dr_velocity_scale")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sample_domain()
        self._reset_internal_state()
        self._elapsed_steps = 0
        self._last_action = None
        self._jump_recency = 99
        self._slide_recency = 99
        self._action_buffer = [int(C.Action.NOOP)] * self.action_delay
        self._act_hist = []
        self._frames = []
        state = self._snapshot()
        self._prev_score = int(state["score"])
        self._prev_distance = float(state["distance"])
        self._prev_state = state
        obs = self._build_obs(state)
        info = {"raw_state": state, "score": self._prev_score, "distance": self._prev_distance}
        return obs, info

    # -------------------------------------------------------------- gym: step
    def step(self, action: int, ext_distance: Optional[float] = None):
        a = int(action)
        if a >= C.N_ACTIONS:
            raise ValueError(
                f"action {a} is reserved for reset(); the policy must emit 0..{C.N_ACTIONS - 1}"
            )

        # Action-application delay: the phone applies the agent's action ~1 step late (TCP + frame
        # lag) while the sim is instant. Buffer it so the policy trains against the SAME latency it
        # meets on the phone and learns to anticipate (it has the lookahead obs to do so) — the cause
        # of marginal/unreliable turns. action_delay=0 (default) for the fidelity harness/replay.
        if self.action_delay > 0:
            self._action_buffer.append(a)
            a = self._action_buffer.pop(0)

        if not self._is_dead:
            self._apply_action(a)
            self._advance(self.step_dt, ext_distance=ext_distance)

        state = self._snapshot()
        score = int(state["score"])
        distance = float(state["distance"])

        # Reward on distance (true objective) or legacy score — mirror the real env (see constants).
        if self.reward_mode == "distance":
            reward = max(0.0, distance - self._prev_distance) * self.reward_scale
        else:
            reward = (score - self._prev_score) * self.reward_scale
        reward += R.shaping_reward(
            a, self._prev_state, self._last_action,
            action_cost=self.action_cost, reaction_bonus=self.reaction_bonus,
            turn_spam_penalty=self.turn_spam_penalty, idle_action_penalty=self.idle_action_penalty,
            redundant_action_penalty=self.redundant_action_penalty, strafe_penalty=self.strafe_penalty,
            airborne_action_penalty=self.airborne_action_penalty,
            jump_hold_penalty=self.jump_hold_penalty,
            jumped_recently=self._jump_recency < int(self.cfg.get("jump_hold_window", 6)),
            slid_recently=self._slide_recency < int(self.cfg.get("jump_hold_window", 6)),
            narrow_hold_bonus=self.narrow_hold_bonus,
            aim_center_weight=self.aim_center_weight,
            velocity_drop_penalty=self.velocity_drop_penalty,
            off_center_penalty=self.off_center_penalty,
            slide_jump_penalty=self.slide_jump_penalty,
            turn_buffer_bonus=self.turn_buffer_bonus,
            landing_hazard_penalty=self.landing_hazard_penalty,
            turn_takeoff_penalty=self.turn_takeoff_penalty,
            prev_velocity=float(self._prev_state.get("velocity", 0.0)),
            velocity=float(state.get("velocity", 0.0)),
            lane=float(self._prev_state.get("lane", 0.0)),
        )
        terminated = bool(state["is_dead"] or state["is_game_over"])
        if terminated:
            reward -= self.death_penalty

        self._prev_score = score
        self._prev_distance = distance
        self._prev_state = state
        self._last_action = a
        # advance jump/slide action recency (0 on a fresh press) for the windowed jump_hold_penalty
        self._jump_recency = 0 if a == int(C.Action.JUMP) else min(99, self._jump_recency + 1)
        self._slide_recency = 0 if a == int(C.Action.SLIDE) else min(99, self._slide_recency + 1)
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.max_episode_steps

        # record the action the policy just executed (for the action-history obs), keep last K.
        if self.action_history_len:
            self._act_hist.append(a)
            if len(self._act_hist) > self.action_history_len:
                self._act_hist = self._act_hist[-self.action_history_len:]
        obs = self._build_obs(state)
        info = {
            "raw_state": state,
            "score": score,
            "distance": distance,
            "elapsed_steps": self._elapsed_steps,
            "max_episode_steps": self.max_episode_steps,
            "death_type": self._death_type,
        }
        return obs, float(reward), terminated, truncated, info

    # ----------------------------------------------------------- action model
    def _apply_action(self, a: int) -> None:
        """Translate the action byte into a flag transition / lane shift / turn (SPEC §3)."""
        cfg = self.cfg
        if a != int(C.Action.JUMP):
            self._jump_held_airborne = False      # the button was released
        if a == int(C.Action.JUMP):
            # HELD-JUMP BOOKKEEPING (WAVE 16): a JUMP pressed while ALREADY AIRBORNE is dropped by the
            # guard below -- free in-sim, but on device the still-held press fires on touchdown. Record
            # it so the landing can honour it. See DEFAULT_CONFIG held_jump_rejump_*.
            if self._jump_t > 0.0:
                self._jump_held_airborne = True
            # only meaningful from the ground; sets IsJumping for jump_duration.
            if self._jump_t <= 0.0 and self._slide_t <= 0.0:
                # JUMP-BYTE DROP [2026-07-05, v14_freshphone ep6 @108m]: on device the game can silently
                # REFUSE a Jump() (observed right after a turn cell: byte sent, is_jumping never fired,
                # policy trusted its action history over the persisting must_jump -> died in the hole).
                # Train with a small drop prob so the policy learns the CLOSED-LOOP retry (re-jump while
                # must_jump and grounded). 0 by default (harness/replay fidelity unchanged).
                if float(cfg.get("jump_drop_prob", 0.0)) > 0.0 and \
                        self._rng.random() < float(cfg.get("jump_drop_prob", 0.0)):
                    return
                # JUMP-WHILE-STRAFING REFUSAL [2026-07-06, DEVICE-MEASURED 56% vs 4% — see config]:
                # a jump commanded mid-glide is usually eaten by the game. The device criterion is the
                # lane actually MOVING (>0.8/frame), i.e. |target-lane| > 0.8/alpha ~ 1.8 — a rider
                # SETTLED on a narrow strip (small residual) jumps normally; a fresh strafe glide does not.
                if abs(self._lat_target - self._lane) > 1.8 and \
                        self._rng.random() < float(cfg.get("jump_refuse_strafing_prob", 0.6)):
                    return
                # GAP-LIP REFUSAL [2026-07-06, DEVICE-MEASURED — see gap_jump_lip_cp]: past the solid lip
                # of a wide-gap/crevasse cell the runner is already over the void — Jump() is ignored.
                # (ALL grounded-at-gap crevasse deaths: act=2 spam on pt34/35 at cp 0.15+, takeoff never
                # fired.) The survivable takeoff is on the APPROACH cell (predicted_jump cues it).
                _cell_now = self._track[self._cell_index]
                if _cell_now.get("wide_gap") and \
                        self._cell_progress() > float(cfg.get("gap_jump_lip_cp", 0.2)):
                    # POST-LANDING GAP RE-JUMP [WAVE 12 DEVICE-MEASURED 2026-07-09, landing_lock_
                    # forensic context split]: a jump within ~1 frame of TOUCHDOWN on a wide-gap cell
                    # fires 46% on device (41/90) — the hole-chain RECOVERY affordance (the human's
                    # wave-9 ep10 pattern) — while the lip rule alone refuses 100% in-sim, making the
                    # recovery UNLEARNABLE. With the knob on, a fresh landing bypasses the lip refusal
                    # at the measured rate (prob>0-first rng gating; default 0 = unchanged).
                    plr = float(cfg.get("post_land_gap_rejump_prob", 0.0))
                    if not (plr > 0.0 and self._frames_since_land <= 1
                            and self._rng.random() < plr):
                        return
                lat = int(cfg.get("jump_exec_lat_frames", 0))
                if lat > 0 and self._pending_jump_frames <= 0:
                    # DEVICE JUMP LATENCY: don't fire now — schedule the takeoff `lat` (+/- jitter) frames
                    # later (see jump_exec_lat_frames). The policy must PRE-AIM (predicted_jump).
                    jit = int(cfg.get("jump_exec_lat_jitter", 0))
                    self._pending_jump_frames = max(1, lat + (self._rng.integers(-jit, jit + 1) if jit > 0 else 0))
                elif lat <= 0:
                    self._jump_t = self._jump_airtime()
                    # record WHERE this jump took off (longitudinal pos) — the step-up crevasse checks the
                    # runner jumped late enough (near the gap edge) so an early jump fails on the raised wall.
                    self._jump_takeoff_pos = self._long_pos()
        elif a == int(C.Action.SLIDE):
            if self._slide_t <= 0.0 and self._jump_t <= 0.0:
                # NOTE [WAVE 13 2026-07-14]: a slide_send_lat_frames knob was prototyped here (flame
                # late-slide fix) but MEASURED INERT — the sim already forces the pre-slide because the
                # must_slide death resolves during the cell crossing in _advance, before the next step's
                # must_slide obs updates (an on-cell reaction is already fatal in-sim; survivors pre-slide
                # on next_must_slide). Flame late-slide is NOT a sim gap. Reverted to keep this path clean.
                self._slide_t = float(cfg["slide_duration"])
        elif a == int(C.Action.STRAFE_LEFT):
            # ANALOG TILT (device-measured 2026-07-02, lateral-accel-15a): byte0 = HOLD LEFT — sets a
            # persistent lateral target -edge; the lane first-order-lags toward it in _advance. The
            # target STAYS until byte1/byte6 (level) — exactly the hook's tiltCmd semantics.
            self._lat_cmd = -float(cfg["lane_edge"])
            self._tilt_dir = -1
        elif a == int(C.Action.STRAFE_RIGHT):
            self._lat_cmd = +float(cfg["lane_edge"])
            self._tilt_dir = +1
        elif a in (int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT)):
            self._attempt_turn(a)
        elif a == int(C.Action.NOOP):
            # NoOp = LEVEL the phone: target 0, the game recenters (measured alpha == hold alpha).
            self._lat_cmd = 0.0
            self._tilt_dir = 0
        # jump/slide/turn leave the tilt target UNCHANGED (device: lateral drift continues mid-jump)

    def _attempt_turn(self, a: int) -> None:
        """Turn at a junction. RE-MODELLED 2026-07-01 from the device capture (human_calib_v2, 365 turns):
        turning is POSITIONAL and PROMPT — a swipe only turns you when you are ON the junction cell.

        Device facts (from human_calib_v2 + a DEVICE-EVAL of a buffered model, 2026-07-01):
          * is_turn is a ~1-frame flash at cell START (cprog med 0.23, p90 0.38) — the human swipes AS
            SOON as it appears (cprog med 0.25); the facing flips ~1 step later. There is NO "turn at
            cell-centre"; turning early WITHIN the junction cell is correct (off-centre death DELETED).
          * NO SAFE EARLY BUFFER. A swipe issued BEFORE the junction (on the approach cell / a straight)
            does NOT get latched-and-fired-later — on device it hits the wall (a survivable STUMBLE) and,
            if repeated, rotates you off the path into the void. [MEASURED: a buffered-turn model that
            pre-fired on next_is_turn survived 99.8% in-sim but DIED on device firing turns on straights
            -> off-path Falls (median 226 vs the un-buffered 736). The buffer was a SIM EXPLOIT.] So a
            turn off the junction cell is a STUMBLE, never a free pre-fire. The policy must turn ONLY on
            the is_turn flash — like the human.

        The ONLY turn death is MISSING a mandatory turn -> run off the edge (Fall, in
        _cross_into_next_cell) OR two stumbles too close (spamming pre-fires)."""
        # OVER-ROTATION: a 2nd turn during the post-turn linger hits the wall -> stumble (survivable once).
        if self._post_turn_linger > 0:
            self._stumble()
            return
        # HOOK DEBOUNCE [WAVE 10, hook-measured 2026-07-05 drop path 'debounce 0.3s']: the gadget drops
        # any turn byte within turn_drop_debounce_s of the previous one. Modeled so turn-byte SPAM keeps
        # its real device price when turn_offjunction_drop makes off-junction turns non-fatal: a junk
        # turn just before a junction EATS the real swipe -> missed turn -> Fall. Every processed turn
        # byte (incl. dropped ones) arms the timer. Default 0 = off (stream/replay unchanged).
        deb = float(self.cfg.get("turn_drop_debounce_s", 0.0))
        if deb > 0.0:
            if self._turn_debounce_t > 0.0:
                return                       # eaten by the hook's anti-double-turn debounce
            self._turn_debounce_t = deb
        want_left = a == int(C.Action.TURN_LEFT)
        cell = self._current_cell()
        cp = self._cell_progress()
        # DIRECT turn: on the junction cell, within the executable window, valid exit side. The turn
        # executes at ANY cprog in [early, late] but WHERE the swipe lands depends on cprog (see
        # _execute_turn) — swipe at cell center to land centered.
        # HOOK TURN-BUFFER (2026-07-04) — models temple_run_hook.js TURN_BUFFER=true (hook 16j): a turn
        # byte issued anywhere ON the junction cell fires the native turn CENTERED at cell-center (cp 0.40)
        # on the 60Hz Update, timing-free. Removes the 10Hz turn-timing latency (the v14 ceiling). The
        # policy must still be CENTERED (offset = lane*carry) and swipe near the junction; it no longer
        # needs precise cp timing. Sim + device now MATCH on turns. Approach-cell pre-swipes are handled by
        # turn_approach_buffer (enable both together).
        if self.cfg.get("hook_turn_buffer") and bool(cell.get("is_turn")) and (
                cell.get("can_turn_left") if want_left else cell.get("can_turn_right")):
            # LATE-QUEUE MISS [2026-07-05 v3, DEVICE-MEASURED — see turn_buffer_late_cp]: the hook's
            # valid-dir guard checks native CanTurn at write time; past the CanTurn window end (~cp 0.56,
            # velocity-INDEPENDENT: 205 first-swipes, fired max 0.55-0.56 in every velocity band incl
            # v207, missed min 0.59) the GI_ShouldTurn write is dropped -> the turn NEVER FIRES -> off
            # the edge. Punishes second-frame swipes (~0.6 at v>150, v27's device hesitation deaths)
            # while first-frame swipes stay safe at all speeds (v14: 0 missed turns, 5462m fresh-phone).
            # decision cp -> PROCESSING cp: the byte arrives turn_send_lat_frames later (velocity-scaled).
            # At high speed this pushes an on-junction decision past the deadline -> dead-letter.
            slat = float(self.cfg.get("turn_send_lat_frames", 0.0))
            proc_cp = cp + (self._velocity * slat * float(self.step_dt) / float(self.cfg["cell_length"])
                            if slat > 0.0 else 0.0)
            if proc_cp <= float(self.cfg.get("turn_buffer_late_cp", 0.56)):
                self._execute_turn(cell, want_left, float(self.cfg["turn_center_cprog"]), from_buffer=True)
            # else: queued too late (arrival past the deadline) -> silent miss / dead-letter off-track.
            return
        early = float(self.cfg["turn_exec_early"])
        late = float(self.cfg["turn_exec_late"])
        if bool(cell.get("is_turn")) and early <= cp <= late and (
                cell.get("can_turn_left") if want_left else cell.get("can_turn_right")):
            self._execute_turn(cell, want_left, cp)
            return
        # APPROACH-BUFFER (flag `turn_approach_buffer`): a swipe on the APPROACH cell (the NEXT cell is the
        # junction, valid exit side) is LATCHED and fired CENTERED on entering the junction (see
        # _cross_into_next_cell) — the game's HandleDelayedTurning. NOT the §14 exploit (which turned
        # immediately on the approach). A swipe FAR from any junction still falls through to FATAL below.
        if self.cfg.get("turn_approach_buffer") and not bool(cell.get("is_turn")):
            nxt = self._peek_next()
            if bool(nxt.get("is_turn")) and (
                    nxt.get("can_turn_left") if want_left else nxt.get("can_turn_right")):
                self._pending_turn = "L" if want_left else "R"
                self._pending_turn_ttl = int(self.cfg.get("turn_buffer_ttl", 3))
                return
        # TURN ON A STRAIGHT = FATAL [DEVICE-MEASURED 2026-07-02, single TurnLeft on a plain centered
        # straight: 8/8 DIED, lane thrown to +/-8-14 off the path]. The native turn rotates the runner
        # off the path -> water. This SUPERSEDES the old "swipe on a straight = survivable stumble"
        # (§13/§14, wrong under the accel-tilt hook). It is THE reason every policy spam-turned: the sim
        # gave free turns anywhere (stumble), the device kills each one. A turn with NO junction context
        # (not on a turn cell, not approaching one) -> Fall. Only a MISTIMED turn AT/NEAR a junction
        # (right place, wrong cprog/side) is a survivable stumble.
        # OFF-JUNCTION DROP [WAVE 10 2026-07-09]: the fatal rule below was measured 2026-07-02 under the
        # OLD direct fn.TurnLeft execution. Hook 16o (2026-07-04) reroutes turn bytes through the game's
        # HandleDelayedTurning, and the 2026-07-05 probe MEASURED the hook DROPPING off-junction bytes
        # (debounce / invalid-dir CanTurn=false / not-near-junction; pre-onset writes: 0 in all captures)
        # -> on TODAY'S device an off-junction turn byte is a harmless no-op. With the knob on (+
        # hook_turn_buffer) the sim matches that: silent drop instead of kill/stumble — which also removes
        # the exploration kill-barrier around the approach-latch (the turn dead-letter fix needs the
        # policy to DISCOVER the pre-swipe). Pair with turn_drop_debounce_s so spam still costs (a junk
        # turn eats the next real swipe). Default off = v70-era behavior unchanged.
        if self.cfg.get("turn_offjunction_drop") and self.cfg.get("hook_turn_buffer"):
            return
        near_junction = bool(cell.get("is_turn")) or bool(self._peek_next().get("is_turn"))
        if near_junction:
            self._stumble()      # mistimed at a junction -> survivable wobble (2nd too soon -> Fall)
        else:
            self._kill(1)        # turn on a plain straight -> rotates off the path -> water

    def _execute_turn(self, cell: dict, want_left: bool, swipe_cp: float, from_buffer: bool = False) -> None:
        """Round the corner and LAND at a lateral offset set by the swipe cprog (device-measured):
        offset = (swipe_cp - center) * scale + carry(pre-turn lane). Land past the edge -> off the exit
        path -> Fall (water). Otherwise land there; the analog tilt recenters. Swipe at ~0.44 = centered."""
        center = float(self.cfg["turn_center_cprog"])
        scale = float(self.cfg["turn_offset_scale"])
        kill = float(self.cfg["turn_land_kill"])
        carry = float(self.cfg["turn_lane_carry"])
        # BUFFERED (swipe-queue) TURNS LAND NATIVELY [2026-07-06 DEVICE-MEASURED]: the game's own
        # HandleDelayedTurning fires the turn at the corner with a forgiving native landing — v14 swiped
        # off-center (lane up to 4.6) on 212/212 fresh-phone junctions with ZERO landing deaths. The
        # carry=3.0 amplification was measured on the OLD immediate-turn path and, applied to buffered
        # turns, teaches "recenter BEFORE swiping" -> the 2nd-frame swipe -> late-queue MISS (v27/v32
        # device junction deaths at lane 2-3.4 onsets). Buffered turns use a mild carry instead.
        if from_buffer:
            carry = float(self.cfg.get("turn_lane_carry_buffered", 1.0))
        # CONTROL LATENCY [DEVICE-MEASURED 2026-07-04]: on the phone the Frida-over-TCP loop lags a
        # JUNCTION-CELL swipe by ~2 frames (obs staleness +~0.20cp + exec delay +~0.24cp), so it LANDS
        # LATE (execution cp ~ swipe_cp + velocity*lat_frames*dt/cell). At speed this pushes the landing
        # past the kill edge -> Fall. The APPROACH-BUFFER swipe is IMMUNE (from_buffer -> fires centered,
        # HandleDelayedTurning), so with lat>0 the ONLY latency-robust strategy is to PRE-SWIPE on the
        # approach cell. Default lat_frames=0 (fidelity harness / replay unchanged); v20 training turns it on.
        if not from_buffer:
            lat_frames = float(self.cfg.get("turn_exec_lat_frames", 0.0))
            if lat_frames > 0.0:
                swipe_cp = swipe_cp + self._velocity * lat_frames * float(self.step_dt) / float(self.cfg["cell_length"])
        # SIGN [DEVICE-MEASURED 2026-07-02, replay_diff turn-landing detail]: TurnLeft lands the runner
        # to the RIGHT of the new-path center when swiped late (cp>center), TurnRight to the LEFT — i.e.
        # left -> +offset, right -> -offset. (Was backwards, giving turn_land_err ~4.6; the magnitude
        # (cp-0.44)*45 and the recenter rate were already right.)
        sign = 1.0 if want_left else -1.0
        # LANDING OFFSET = cprog term + AMPLIFIED pre-turn lane. [DEVICE-MEASURED 2026-07-02, forced
        # turns-only track]: a policy that STRAFES into the turn (pre-turn lane ~5) lands at ~18 off
        # (the turn rotates + the held tilt keeps pushing) -> off-path water. So the pre-turn lane is
        # AMPLIFIED by turn_lane_carry (~3), not just added. Centered (lane 0) -> cprog term only ->
        # survives. This is the on-device NN death the controlled track exposed (leaning into turns);
        # it forces the policy to CENTER (NoOp) before swiping. The tilt target is NOT reset here (it
        # persists through the turn on device), so a still-held strafe keeps the drift going after.
        if from_buffer:
            # NATIVE delayed-turn buffer (hook 16k): the game fires the turn at the corner and lands the
            # runner CENTERED on the exit path — device-measured landing lane ~0, corr w/ approach lane
            # = -0.06 (NO cp timing, NO lane amplification). v14+16k: 0/8 turn overshoots. The HELD tilt
            # (lat_target) still drifts the lane AFTER, modelling the pt-1 post-turn Falls (strafe held
            # into the turn). So the ONLY residual is: don't hold a strafe into the junction.
            offset = 0.0
        else:
            offset = sign * (swipe_cp - center) * scale + self._lane * carry
        self._facing = _LEFT_OF[self._facing] if want_left else _RIGHT_OF[self._facing]
        if abs(offset) > kill:
            self._kill(1)   # landed off the exit path -> water
            return
        # MARGINAL-ZONE TAIL RISK [DEVICE-MEASURED 2026-07-04, 150 pooled turn-swipes]: a swipe in the
        # marginal cp window occasionally lands CATASTROPHICALLY off-path (lane ~22 -> Fall) even when the
        # deterministic offset would survive. Device death rate by swipe cp: 0.35-0.50 ~0%, 0.50-0.58 ~8%,
        # 0.58-0.64 ~18% (median landing only ~4-5 — it's a sharp cliff at cp~0.60 + 10Hz frame
        # quantization, NOT velocity/tilt driven). The deterministic sim can't see this, so retrains aim at
        # the sim-safe marginal zone (cp 0.5-0.55) and die on device. Model it: a probabilistic kill that
        # ramps with cp above turn_tail_cp_lo -> the sim PENALIZES aiming near the cliff -> the policy learns
        # to aim EARLY (reliable cp<0.50) like v14. replay_diff sets turn_tail_pmax=0 (deterministic replay).
        lo = float(self.cfg.get("turn_tail_cp_lo", 0.50))
        hi = float(self.cfg.get("turn_tail_cp_hi", 0.64))
        pmax = float(self.cfg.get("turn_tail_pmax", 0.20))
        if pmax > 0.0 and swipe_cp > lo:
            p = min(pmax, (swipe_cp - lo) / max(1e-6, hi - lo) * pmax)
            if self._rng.random() < p:
                self._kill(1)   # device marginal-zone tail: catastrophic off-path landing
                return
        self._lane = offset          # lands here; the HELD tilt keeps pushing (lat_target unchanged)
        self._tilt_dir = 0
        self._turn_landed = True      # device shows the FULL landing at f+1; skip the recenter THIS step
        cell["is_turn"] = False
        cell["can_turn_left"] = False
        cell["can_turn_right"] = False
        self._cells_since_turn = 0
        self._pending_turn = None
        # Start the linger: is_turn stays True for N frames (snapshot), tempting a fatal re-turn.
        n = int(self.cfg.get("turn_linger_frames", 0))
        if n > 0:
            self._post_turn_linger = n
            self._linger_can_l = bool(want_left)
            self._linger_can_r = not want_left

    # --------------------------------------------------------- physics tick
    def _advance(self, dt: float, ext_distance: Optional[float] = None) -> None:
        """Advance velocity, transient timers, distance, cell crossings + scoring for ``dt``.

        ``ext_distance`` (replay only) is the real ``hidden.distance`` for this step. When given,
        cell crossings are driven off the recorded distance delta (units = ``cell_length_real``),
        decoupling path_type from the velocity model — the velocity/score fields still evolve via
        the model so their own tiers stay meaningful. ``None`` = procedural: cross on velocity*dt.
        """
        cfg = self.cfg

        # DEVICE JUMP LATENCY: a buffered JUMP command fires `jump_exec_lat_frames` steps after it was
        # issued (see _apply_action). Count down and fire at the CURRENT position (so the takeoff is later
        # than the policy's command -> it must pre-aim). Only fires if still grounded.
        if self._pending_jump_frames > 0:
            self._pending_jump_frames -= 1
            if self._pending_jump_frames == 0 and self._jump_t <= 0.0 and self._slide_t <= 0.0:
                self._jump_t = self._jump_airtime()
                self._jump_takeoff_pos = self._long_pos()

        # tick down transient flags.
        _was_airborne = self._jump_t > 0.0
        self._jump_t = max(0.0, self._jump_t - dt)
        # post_land_gap_rejump support: track frames since the jump TOUCHDOWN (this decay crossing 0).
        if _was_airborne and self._jump_t <= 0.0:
            self._frames_since_land = 0
            # HELD-JUMP -> LANDING RE-JUMP (WAVE 16): the button was still down at touchdown, so the
            # device fires a fresh jump the policy never chose. Context-dependent rate (device-
            # measured: gap cells convert far more often than solid ground). prob>0-first gating.
            _hj_s = float(cfg.get("held_jump_rejump_solid", 0.0))
            _hj_g = float(cfg.get("held_jump_rejump_gap", 0.0))
            if self._jump_held_airborne and (_hj_s > 0.0 or _hj_g > 0.0) and not self._is_dead:
                _c = self._track[self._cell_index] if self._cell_index < len(self._track) else {}
                _p = _hj_g if _c.get("wide_gap") else _hj_s
                if _p > 0.0 and self._rng.random() < _p:
                    self._jump_t = self._jump_airtime()
                    self._jump_takeoff_pos = self._long_pos()
            self._jump_held_airborne = False
        else:
            self._frames_since_land = min(99, self._frames_since_land + 1)
        self._slide_t = max(0.0, self._slide_t - dt)
        self._turn_debounce_t = max(0.0, self._turn_debounce_t - dt)
        self._strafe_t = max(0.0, self._strafe_t - dt)

        # STRAFE-COMMAND LATENCY (strafe_exec_lat_frames): the commanded tilt (_lat_cmd) reaches the
        # ACTIVE target (_lat_target, driving the lag below) N frames late — the device Frida-TCP control
        # lag on the tilt byte. lat=0 -> immediate (harness/replay unchanged).
        slat = int(cfg.get("strafe_exec_lat_frames", 0))
        if slat > 0:
            self._lat_target_buf.append(self._lat_cmd)
            if len(self._lat_target_buf) > slat:
                self._lat_target = self._lat_target_buf.pop(0)
        else:
            self._lat_target = self._lat_cmd

        # LATERAL model (device-MEASURED 2026-07-02, strafe_probe_tilt2 / lateral-accel-15a): the lane
        # is a FIRST-ORDER LAG toward the commanded target (hold -> +/-edge, level/NoOp -> 0):
        #   lane += (target - lane) * (1 - exp(-dt/lat_tau))
        # alpha 0.446 per 0.1 s frame both ways (tau ~0.165 s); full 0->7 reach ~0.5 s; recenter is the
        # exact mirror. The OLD nudge-timer/near-hold model is DEAD (it modeled the persistent-px strafe).
        edge = float(cfg["lane_edge"])
        if getattr(self, "_turn_landed", False):
            self._turn_landed = False     # a turn just landed this step -> show the FULL landing (device
                                          # recenters starting NEXT frame, not the landing frame)
        else:
            alpha = 1.0 - math.exp(-dt / max(1e-3, float(cfg["lat_tau"])))
            self._lane += (self._lat_target - self._lane) * alpha
            self._lane = max(-edge, min(edge, self._lane))
        self._stumble_kill_t += dt          # time since the last stumble (drives the 2nd-stumble kill)
        if self._stumble_t > 0.0:
            self._stumble_t = max(0.0, self._stumble_t - dt)
        # WALL-HIT: over-tilted past the edge -> stumble (survivable once) and clamp back off the wall.
        # (Reachable only via turn-landing overshoot — the tilt glide itself clamps at the edge.)
        # HIGH GROUND (Cliff, zone level 3) [USER-OBSERVED on device 2026-07-05]: there is NO side wall up
        # there — over-strafing past the edge falls off the cliff -> DEATH, not a stumble.
        if abs(self._lane) > edge:
            if int(self._track[self._cell_index].get("level", 2)) >= 3:
                self._kill(1)   # landed/pushed past the high-ground cliff edge -> Fall
                return
            self._stumble()
            self._lane = max(-edge, min(edge, self._lane))
        # CLIFF EDGE-RIDING [2026-07-06 DEVICE-MEASURED — the check above was DEAD CODE for strafing (the
        # glide clamps at the edge, so |lane|>edge never fired) -> v31 rode |lane| 7.0 on plvl 3 and died
        # on device]. Captures: SUSTAINED |lane|>=6.3 on level>=3 for 8-10 frames = Fall (dt1); short
        # touches (1-4 frames) survive. Kill after cliff_edge_kill_frames consecutive frames -> policies
        # must not HOLD full tilt at altitude (transient dodges stay legal).
        if int(self._track[self._cell_index].get("level", 2)) >= 3 and \
                abs(self._lane) >= float(cfg.get("cliff_edge_lane", 6.3)):
            self._cliff_edge_frames += 1
            if self._cliff_edge_frames >= int(cfg.get("cliff_edge_kill_frames", 6)):
                self._kill(1)   # rode the high-ground cliff edge too long -> Fall
                return
        else:
            self._cliff_edge_frames = 0
        if self._post_turn_linger > 0:          # post-turn is_turn linger (sim-to-real trap) counts down
            self._post_turn_linger -= 1

        # Advance the real-scale DistanceTraveled FIRST (it indexes the velocity curve + drives cell
        # crossings): replay echoes the recorded value; procedural integrates the current velocity.
        if ext_distance is not None:
            moved_cells = max(0.0, float(ext_distance) - self._last_ext_distance)
            self._last_ext_distance = float(ext_distance)
            self._distance_real = float(ext_distance)
            # Same 2026-07-27 correction on the REPLAY side: the device's distance between adjacent
            # PathElements is not the constant 6.4 either -- measured 5.4 @v100 rising to 10.2 @v215,
            # i.e. slope*cell_length*v (0.0508*v, which reproduces the measured 9.13 at v180). The
            # flat cell_length_real only holds near v=136 and desynchronises a replay at speed.
            _dslope = float(cfg.get("dist_per_velsec_slope", 0.0))
            cell_len = (_dslope * float(cfg["cell_length"]) * self._velocity if _dslope > 0.0
                        else float(cfg["cell_length_real"]))
        else:
            _dslope = float(cfg.get("dist_per_velsec_slope", 0.0))
            _dpv = (_dslope * self._velocity) if _dslope > 0.0 else float(cfg["dist_per_velsec"])
            self._distance_real += self._velocity * dt * _dpv
            cell_len = float(cfg["cell_length"])

        # velocity = calibrated curve(distance) — the real AccCurve is distance-gated, not linear
        # (measured 100->210 over ~4k distance, then plateau). Linear accel is the fallback.
        if self._vel_curve_d is not None:
            curve_v = float(np.interp(self._distance_real, self._vel_curve_d, self._vel_curve_v))
        else:
            curve_v = min(float(cfg["max_velocity"]),
                          self._velocity + float(cfg["accel_per_sec"]) * dt)
        # post-stumble the slowdown PERSISTS: velocity re-climbs at stumble_recover_rate (device-
        # measured ~0.5-1.1 u/s) instead of teleporting back to curve(distance) next tick.
        if self._stumble_vel is not None:
            self._stumble_vel = min(curve_v,
                                    self._stumble_vel + float(cfg["stumble_recover_rate"]) * dt)
            self._velocity = self._stumble_vel
            if self._stumble_vel >= curve_v:
                self._stumble_vel = None
        else:
            self._velocity = curve_v
        self._velocity *= self._vel_scale   # domain-randomization per-episode velocity jitter (1.0 if off)

        # distance covered this tick, scored ~ velocity*dt (SPEC §5).
        moved = self._velocity * dt
        self._distance += moved
        self._score += moved * float(cfg["score_per_distance"])
        if ext_distance is None:
            moved_cells = moved
        self._dist_into_cell += moved_cells

        # cross as many cell boundaries as we covered (usually 0-1 at 10 Hz).
        guard = 0
        while self._dist_into_cell >= cell_len and not self._is_dead:
            guard += 1
            if guard > 64:
                break
            self._dist_into_cell -= cell_len
            self._cross_into_next_cell()

        # JUMP-INTO-SLIDE-OBSTACLE death [DEVICE-MEASURED 2026-07-03, ppo_speed_v8 eval, 4/8 deaths]: a
        # slide-obstacle (TreeSlide / FlameTower) is a LOW bar the runner passes UNDER — being AIRBORNE
        # in that cell (jumped INTO it on entry, or fired a JUMP *mid-slide*) collides -> death. The old
        # model only checked "not sliding" ON ENTRY, so a jump issued DURING the slide cell was
        # survivable in-sim but FATAL on device (the "slid then jumped and died" the user reported).
        # Check every tick while airborne on a must-slide cell.
        # ★ DEVICE-CALIBRATED 2026-07-28: this killed 100% of the time, the DEVICE kills 24%
        # (23 of 94 events over 45 captures). The gap matters more than its size: these takeoffs are
        # 100% FORCED (a must_jump demanded them -- 0 discretionary in BOTH sim and device), and in
        # sim NO takeoff timing survives (15/15 fatal across cp 0.11..0.76). So the world presented a
        # situation with no survivable action, punished the one correct jump, and taught nothing --
        # while accounting for ~33% of the champion's deaths (flame 88% airborne, TreeSlide 50%).
        # Probabilistic rather than a threshold, following hole_trap_kill_prob; a sharp cutoff was
        # the v55 mistake. prob >= 1.0 short-circuits BEFORE the draw so the old stream is exact.
        if not self._is_dead and self._jump_t > 0.0:
            _oc = int(self._current_cell().get("obstacles", 0))
            if _oc in _MUST_SLIDE:
                _pk = float(self.cfg.get("airborne_slide_kill_prob", 1.0))
                if _pk >= 1.0:
                    self._kill(5 if _oc == OBST_FLAME_TOWER else 3)
                elif self._slide_air_rolled != self._cell_index:
                    # ONE roll per cell. This check runs every tick, so re-rolling here turned a
                    # 24% device rate into ~75% (1 - 0.76^n over n ticks on the cell).
                    self._slide_air_rolled = self._cell_index
                    if self._rng.random() < _pk:
                        self._kill(5 if _oc == OBST_FLAME_TOWER else 3)

        # wide-gap LIP deadline: entered a wide-gap cell grounded (survivable lip) — still grounded
        # past gap_entry_grace of the cell -> fell in. A takeoff (or leaving the cell) clears it.
        if self._gap_grace_cell is not None and not self._is_dead:
            if self._cell_index != self._gap_grace_cell or self._jump_t > 0.0:
                self._gap_grace_cell = None
            elif (self._dist_into_cell / max(1e-6, cell_len)) >= float(cfg["gap_entry_grace"]):
                cell = self._track[self._cell_index]
                self._kill(1 if int(cell.get("obstacles", 0)) == OBST_HOLE else 2)
                self._gap_grace_cell = None

    def _cross_into_next_cell(self) -> None:
        """The player leaves the current cell -> resolve junction (exit), step forward, then resolve
        the entered cell's obstacle (ENTRY)."""
        leaving = self._current_cell()
        # Missed-turn death: must turn while ON the junction; checked on exit.
        if leaving.get("is_turn") and (leaving.get("can_turn_left") or leaving.get("can_turn_right")):
            self._kill(1)   # Fall
            return
        # move onto the next cell.
        self._cell_index += 1
        self._extend_track()
        entered = self._current_cell()
        # APPROACH-BUFFER fire (flag `turn_approach_buffer`): a swipe latched on the approach cell fires
        # CENTERED (swipe_cp = turn_center_cprog) on entering the junction — the game's HandleDelayedTurning.
        # Pre-turn lane still carries (so only a centered approach lands clean). Expires (no-op = the game's
        # buffer timeout) if the latch runs out before a junction is reached.
        if self._pending_turn is not None:
            _wl = self._pending_turn == "L"
            if bool(entered.get("is_turn")) and (
                    entered.get("can_turn_left") if _wl else entered.get("can_turn_right")):
                self._pending_turn = None
                self._execute_turn(entered, _wl, float(self.cfg["turn_center_cprog"]), from_buffer=True)
                if self._is_dead:
                    return
            else:
                self._pending_turn_ttl -= 1
                if self._pending_turn_ttl <= 0:
                    self._pending_turn = None
        # Obstacle death checked on ENTRY: the jump/slide must already be ACTIVE when crossing into
        # the obstacle cell. So the policy must act on the LOOKAHEAD (next_must_jump/slide) one cell
        # early — matching the real phone, where reacting only when the obstacle is the nearest cell
        # lands the action a step too late (observed: agent jumps after it's already falling).
        self._resolve_obstacle_on_enter(entered)
        if self._is_dead:
            return
        if entered.get("is_coin"):
            self._score += float(self.cfg["coin_score"])
            entered["is_coin"] = False

    def _resolve_obstacle_on_enter(self, cell: dict) -> None:
        # NARROW strip: must be inside the safe band on every strip-cell entry, else fall off.
        # Band center = narrow_lane * narrow_lane_center (+/-4.5) +/- narrow_lane_tol (2.5) -- device-
        # measured from the game's Region2D polygons (narrow_calib2 2026-07-02: pt20/21 [-7,-2], pt22-27
        # [3,7]). narrow_lane is ALWAYS a SIDE (-1/+1) now -- the device has NO center-safe narrow.
        nl = cell.get("narrow_lane")
        if nl is not None:
            target = int(nl) * float(self.cfg["narrow_lane_center"])
            if abs(self._lane - target) > float(self.cfg["narrow_lane_tol"]):
                self._kill(1)   # outside the narrow's safe band -> fell off the strip
                return
        obst = int(cell.get("obstacles", 0))
        # apex check fires on the crevasse LEDGE cell: step-UP always; ALL crevasse last-cells when
        # crevasse_apex_all is on (step-DOWN/high-ground also demand near-apex timing — device-measured).
        apex_cell = bool(cell.get("stepup"))
        # crevasse_apex_all OR the world-Y graze model self-extend apex detection to ALL last-crevasse cells.
        if not apex_cell and (self.cfg.get("crevasse_apex_all") or self.cfg.get("crevasse_graze_worldy")) \
                and obst == OBST_LEVEL_CHANGE and cell.get("wide_gap"):
            nxt = self._track[self._cell_index + 1] if self._cell_index + 1 < len(self._track) else {}
            apex_cell = not nxt.get("wide_gap")   # the LAST crevasse cell (next cell solid)
        # A step-DOWN TO LOW GROUND has NO raised cap to clear, so it is not an apex cell at all: the
        # apex branch would otherwise kill a grounded runner (jump_t <= 0) before the soft-flag check
        # below ever runs, and crevasse_graze_worldy promotes every last-crevasse cell into it. The
        # graze model itself is a no-op here anyway (cap height is 0 below level 3).
        if cell.get("stepdown_low"):
            apex_cell = False
        if obst in _MUST_JUMP and apex_cell:
            # STEP-UP = VELOCITY-DEPENDENT APEX model [device-MEASURED, human_calib_v2 v210 probe
            # 2026-07-03]: the raised far ledge must be reached NEAR the jump APEX (still high). The jump
            # height is a parabola over the span the airtime covers = jump_duration*velocity/cell_len
            # cells; at HIGH speed the span is BIGGER, so the runner must take off EARLIER to be at apex
            # at the ledge (human: ~2 cells early @v210, on-gap @v150 — the takeoff LEAD scales with
            # velocity). Reach the ledge too EARLY (just launched, low) OR too LATE (descending, low) ->
            # clip the raised wall -> Fall. Replaces the fixed-cprog gate (wrong at speed: it demanded a
            # LATE takeoff; the human jumps EARLIER the faster they go). No lip-grace (a raised ledge, not
            # a hole edge). step-down / holes are UNAFFECTED (handled below).
            if self._jump_t <= 0.0 or self._jump_takeoff_pos is None:
                self._kill(4)       # not airborne at the step-up ledge -> clip the raised wall -> Ledge
                return
            # AIRTIME-CONSISTENT SPAN [2026-07-27 night]: this is a step-UP ledge, and a step-up jump
            # is device-measured ~26% SHORTER (jump_dur_stepup_scale) because landing on raised ground
            # cuts the fall short. The apex parabola must be evaluated over the arc the runner is
            # ACTUALLY flying, not over the flat nominal duration -- otherwise the model says "you are
            # mid-arc" when the runner is already coming down, which is exactly the mid->high graze
            # the user has reported for weeks. Scale 1.0 (default) reproduces the old span exactly.
            span = (float(self.cfg["jump_duration"]) * float(self.cfg.get("jump_dur_stepup_scale", 1.0))
                    * self._velocity / float(self.cfg["cell_length"]))
            # evaluate at the ledge cell's ENTRY (not the current mid-cell frame pos): at 10Hz the first
            # frame on the cell is up to ~0.3 cells past the lip, which skewed frac by up to +0.13 and
            # made the device-fit mh (entry-referenced, 2026-07-05) effectively stricter than measured.
            d = float(self._cell_index) - self._jump_takeoff_pos   # cells from takeoff to the ledge lip
            # DEVICE geometry: the raised cap sits slightly LATER in the arc than the cell index implies
            # (the runner is descending at it) -> shift d so a cp~0.9 takeoff lands the cap in the
            # descending arc (clips) instead of at apex (clears). 0 = the old apex-at-ledge model.
            d += float(self.cfg.get("crevasse_cap_arc_offset", 0.0))
            # takeoff-timing VARIANCE (device std ~0.06-0.1 cp): per-crossing Gaussian shift of d so the
            # descending-arc clearance spreads around the cap top -> partial graze (~32% @lvl3), not the
            # sharp all-or-nothing v55 threshold. Gated (0 = no draw -> equivalence baseline untouched).
            _tk_std = float(self.cfg.get("crevasse_takeoff_std", 0.0))
            if _tk_std > 0.0:
                d += float(self._rng.normal(0.0, _tk_std))
            frac = d / max(1e-6, span)                      # 0=takeoff, 0.5=apex, 1=landing
            height = 1.0 - (2.0 * frac - 1.0) ** 2          # parabola: 0 at ends, 1 at apex
            # WORLD-Y GRAZE model (device-decoded): arc height in WORLD units = jump_peak * parabola; the
            # cap top by LEVEL (15 @lvl>=3, 0 @lvl<3). Graze (chronic stumble) iff arc < cap top; deep clip
            # -> Ledge kill. Supersedes the normalised mh-proxy; reproduces device 32% graze @lvl3 / 0% @lvl2.
            if self.cfg.get("crevasse_graze_worldy"):
                arc_y = float(self.cfg["crevasse_jump_peak"]) * height   # world-units height at the cap
                cap_h = float(self.cfg["crevasse_cap_height_hi"]) if int(cell.get("level", 2)) >= 3 else 0.0
                band = float(self.cfg["crevasse_graze_band"])
                if arc_y < cap_h - band:
                    self._kill(4)    # arc well below the raised cap top -> deep clip -> Ledge
                elif arc_y < cap_h:
                    self._stumble()  # arc just below the cap top -> chronic graze slowdown (x0.85)
                return
            mh = float(self.cfg["stepup_min_height"])
            if height < mh:
                # marginal graze -> non-fatal STUMBLE slowdown (device-measured 2026-07-06); deep
                # clip -> Fall. band 0 (default) = the old binary clear-or-die.
                band = float(self.cfg.get("stepup_clip_band", 0.0))
                if band > 0.0 and height >= mh - band:
                    self._stumble()
                else:
                    self._kill(4)   # not high enough at the raised ledge -> clip it -> Ledge
            return
        if obst in _MUST_JUMP:
            # Non-step-up wide gap (long hole / step-DOWN) or 1-cell hole: must be AIRBORNE crossing in.
            # wide gaps get a lip-grace (the hole starts a bit into the cell; device jumps at cprog
            # 0.98-1.02 still cleared). 1-cell holes keep the strict on-entry check.
            need = float(self.cfg["wide_gap_min_airtime"]) if cell.get("wide_gap") else 0.0
            if self._jump_t <= need:
                # STEP-DOWN TO LOW GROUND: crossing it ON FOOT is device-measured FREE (17 human
                # grounded crossings of a 2->1 / 3->1 drop, ZERO deaths), so the must_jump flag is SOFT
                # here. Without this the sim kills every grounded crossing, "always jump" is the only
                # correct sim policy, and the phone inherits a superstition that then collides with
                # whatever follows the cap. See DEFAULT_SIM_CONFIG stepdown_grounded_ok.
                if cell.get("stepdown_low"):
                    return
                if cell.get("wide_gap") and float(self.cfg["gap_entry_grace"]) > 0.0:
                    self._gap_grace_cell = self._cell_index
                else:
                    self._kill(1 if obst == OBST_HOLE else
                               (4 if obst == OBST_LEVEL_CHANGE else 2))  # Fall / Ledge / Tree
                return
            # WIDE-GAP LANDING MODEL: airborne here, but does the arc actually LAND past the gap?
            # Fires only on the LAST gap cell (next cell solid). A short arc lands in the far gap cell
            # -> clips the far wall (device dt4 Ledge). span(velocity) = a*v + b (device fit). To clear,
            # landing (takeoff_pos + span) must reach the solid cell after this one (cell_index + 1),
            # within a far-lip grace. See DEFAULT_CONFIG wide_gap_landing_model.
            # HOLES ONLY: the span fit was calibrated on flat holes (pt12-16). LevelChange crevasses
            # (obst=OBST_LEVEL_CHANGE) are ALSO wide_gap+must-jump but have a raised-ledge apex physics
            # (handled by the stepup model, or the airborne check for step-DOWN); applying the hole span
            # here rewarded EARLY crevasse jumps -> v47 device-regressed on high-ground crevasses.
            if self.cfg.get("wide_gap_landing_model") and cell.get("wide_gap") \
                    and obst == OBST_HOLE and self._jump_takeoff_pos is not None:
                nxt = self._track[self._cell_index + 1] if self._cell_index + 1 < len(self._track) else {}
                if not nxt.get("wide_gap"):                     # this is the LAST gap cell
                    span = float(self.cfg["wide_gap_span_a"]) * self._velocity \
                        + float(self.cfg["wide_gap_span_b"])
                    d = float(self._cell_index) - self._jump_takeoff_pos   # cells from takeoff to here
                    if span < d + 1.0 - float(self.cfg["wide_gap_land_grace"]):
                        self._kill(4)   # arc lands inside the gap -> clip the far wall (Ledge)
                        return
            # SINGLE-HOLE AIRBORNE TRAP [WAVE 14]: the wide-gap branch above skips a 1-cell hole (not
            # wide_gap), leaving the binary airborne test as its ONLY gate. Device: reaching the hole
            # PAST THE ARC APEX (descending) kills ~30%; reaching it rising/at apex is free. See
            # DEFAULT_CONFIG hole_landing_model. prob>0-first gating -> model off draws NO rng.
            _hp = float(self.cfg.get("hole_trap_kill_prob", 0.0))
            if _hp > 0.0 and self.cfg.get("hole_landing_model") and obst == OBST_HOLE \
                    and not cell.get("wide_gap") and self._jump_takeoff_pos is not None:
                span = float(self.cfg["wide_gap_span_a"]) * self._velocity \
                    + float(self.cfg["wide_gap_span_b"])
                frac = (float(self._cell_index) - self._jump_takeoff_pos) / max(1e-6, span)
                if frac > float(self.cfg["hole_trap_apex_frac"]) \
                        and (_hp >= 1.0 or self._rng.random() < _hp):
                    self._kill(1)   # descending into a 1-cell hole -> Fall (device dt1)
                    return
        elif obst in _MUST_SLIDE:
            # Two DIFFERENT failures were conflated here, and only one is a skill failure:
            #   GROUNDED and not sliding = the policy simply missed the slide -> always fatal.
            #   AIRBORNE                 = it was mid-jump, and could not have slid. On DEVICE this
            #     kills only 24% (23 of 94 over 45 captures); in sim it killed 100%, and those
            #     takeoffs are 100% FORCED by a must_jump (0 discretionary, measured both sides).
            #     15/15 were fatal across takeoff cp 0.11..0.76 -- no action survived, so the world
            #     taught nothing and punished the one correct jump. ~33% of the champion's deaths.
            if self._jump_t > 0.0:
                _pk = float(self.cfg.get("airborne_slide_kill_prob", 1.0))
                if _pk >= 1.0:
                    self._kill(5 if obst == OBST_FLAME_TOWER else 3)  # Burnt / Slide
                elif self._slide_air_rolled != self._cell_index:
                    self._slide_air_rolled = self._cell_index      # one roll per cell
                    if self._rng.random() < _pk:
                        self._kill(5 if obst == OBST_FLAME_TOWER else 3)
            elif self._slide_t <= 0.0:
                self._kill(5 if obst == OBST_FLAME_TOWER else 3)
        # OBST_NARROW / OBST_NONE: no death here (narrow handled above; LevelChange is in _MUST_JUMP).

    def _dist_next_hazard(self, kinds) -> float:
        """Cells from the current cell to the next hazard in `kinds`, capped at OBS horizon.

        0 = the CURRENT cell carries one. Returns the cap when nothing is visible, which the obs
        encodes as 1.0 ("clear as far as I can see"), NOT as certainty of clear.
        """
        from .obs import HORIZON_CELLS
        i0 = self._cell_index
        for d in range(0, HORIZON_CELLS + 1):
            j = i0 + d
            if j >= len(self._track):
                break
            c = self._track[j]
            if d > 0 and c.get("can_turn_left") and c.get("can_turn_right"):
                break          # true T-junction: the continuation is undecided -> blind past it
            if int(c.get("obstacles", 0)) in kinds:
                return float(d)
        return float(HORIZON_CELLS)

    def _jump_airtime(self) -> float:
        """Airtime for a jump starting NOW, scaled by the ground-height change its arc will cross.

        DEVICE-MEASURED (see DEFAULT_SIM_CONFIG jump_dur_stepdown_scale): landing on LOWER ground
        means falling further, so the runner stays airborne longer and the arc spans 3 cells instead
        of 2; landing on RAISED ground cuts the fall short (~0.74x). The sim used one fixed duration
        for all three, so a step-down jump cleared its follower in-sim while sailing into it on the
        phone.

        The landing level is read at the NOMINAL span (the unscaled arc), which is what decides where
        the runner is heading at the moment of takeoff. Both scales at 1.0 -> the old constant, and no
        rng is drawn either way, so the OFF stream stays bit-identical.
        """
        base = float(self.cfg["jump_duration"])
        ds = float(self.cfg.get("jump_dur_stepdown_scale", 1.0))
        us = float(self.cfg.get("jump_dur_stepup_scale", 1.0))
        if ds == 1.0 and us == 1.0:
            return base
        span = base * self._velocity / float(self.cfg["cell_length"])
        i0 = self._cell_index
        j = min(len(self._track) - 1, i0 + max(1, int(span + 0.5)))
        lvl0 = int(self._track[i0].get("level", 2))
        lvl1 = int(self._track[j].get("level", 2))
        if lvl1 < lvl0:
            return base * ds
        if lvl1 > lvl0:
            return base * us
        return base

    def _stumble(self) -> None:
        """Hit a side wall (mistimed swipe/turn OR over-tilt past the edge). Device-measured: the FIRST
        stumble is survivable — velocity drops ~15%, the runner recenters — but a 2nd stumble while
        still stumbling OR within stumble_kill_window of the last one KILLS (Fall)."""
        if self._is_dead:
            return
        if self._stumble_t > 0.0 or self._stumble_kill_t < float(self.cfg["stumble_kill_window"]):
            self._kill(1)   # 2nd stumble too soon after the last -> Fall
            return
        self._velocity *= float(self.cfg["stumble_slowdown"])
        # the slowdown PERSISTS: device velocity re-climbs at ~1 u/s (measured, eval_stumble captures),
        # it is NOT snapped back to curve(distance) next frame — _advance honors _stumble_vel until
        # it re-reaches the curve.
        self._stumble_vel = self._velocity
        self._lane = 0.0                  # the stumble recenters the runner
        self._tilt_dir = 0
        self._strafe_t = 0.0
        self._stumble_t = float(self.cfg["stumble_duration"])
        self._stumble_kill_t = 0.0
        self._stumble_count += 1

    def _kill(self, death_type: int) -> None:
        self._is_dead = True
        self._death_type = int(death_type)

    # -------------------------------------------------------------- snapshot
    def _current_cell(self) -> dict:
        if self._cell_index >= len(self._track):
            self._extend_track()
        return self._track[self._cell_index]

    def _nearest_cell(self) -> dict:
        """The element the hook would report as NearestElement (the current/just-ahead cell)."""
        return self._current_cell()

    def _cell_progress(self) -> float:
        """Longitudinal position within the nearest cell, 0..1 — the obs the policy uses to TIME a turn
        (turn at cell-center, not on the first is_turn frame) and to anticipate. Matches the hook's
        cell_progress (CharacterPlayer pos vs PathElement.Rectangle)."""
        cl = float(self.cfg["cell_length_real"] if self.track_source == "replay" else self.cfg["cell_length"])
        return float(min(1.0, max(0.0, self._dist_into_cell / max(1e-6, cl))))

    def _long_pos(self) -> float:
        """Monotonic longitudinal position = cell_index + cell_progress (in cell units). Used by the
        step-up crevasse to know how many cells before the gap the current jump took off."""
        return float(self._cell_index) + self._cell_progress()

    def _turn_window_open(self) -> bool:
        """is_turn is PRESENTED across the executable window [turn_exec_early, turn_exec_late] (device-
        measured: the swipe executes from the flash ~0.15 to ~0.60). The policy must aim cell CENTER
        (~0.44) within it — swiping early (the flash) lands off-center -> water (see _execute_turn).
        Matches the hook's is_turn latch. Replay keeps the recorded is_turn."""
        if self.track_source == "replay":
            return True
        cell_len = float(self.cfg["cell_length"])
        cp = self._dist_into_cell / max(1e-6, cell_len)
        return float(self.cfg["turn_exec_early"]) <= cp <= float(self.cfg["turn_exec_late"])

    def _snapshot(self) -> dict:
        """Produce the raw-state dict (constants.STATE_FIELDS order) incl. 1-cell lookahead."""
        cell = self._nearest_cell()
        obst = int(cell.get("obstacles", 0))
        must_jump = obst in _MUST_JUMP
        must_slide = obst in _MUST_SLIDE
        win = self._turn_window_open()
        is_turn = bool(cell.get("is_turn")) and win
        # CanTurnLeft/Right are heading-relative in the game, but the recorded/cell fields are
        # already the post-heading turnability the hook reports -> expose them directly (windowed).
        can_l = bool(cell.get("can_turn_left")) and win
        can_r = bool(cell.get("can_turn_right")) and win
        # Post-turn LINGER: after a turn, force is_turn True for N frames (like the phone) so the
        # policy faces the double-turn temptation and must learn (via action history) to resist it.
        if self._post_turn_linger > 0:
            is_turn = True
            can_l = self._linger_can_l
            can_r = self._linger_can_r

        # 1-cell lookahead: the cell ahead of the nearest (raw, not window-gated -> anticipation).
        self._extend_track()
        nxt = self._track[self._cell_index + 1] if self._cell_index + 1 < len(self._track) else {}
        nxt2 = self._track[self._cell_index + 2] if self._cell_index + 2 < len(self._track) else {}
        nxt3 = self._track[self._cell_index + 3] if self._cell_index + 3 < len(self._track) else {}
        # ---- LOOKAHEAD BLOCKING (WAVE 15 PERCEPTION FIX, device-probed 2026-07-26) ----------------
        # The hook walks the PathElement neighbour chain with ONE fixed heading offset per step, so the
        # walk dies the moment the track turns: 72% of device frames on/approaching a junction stream
        # next/next2_path_level = -1, i.e. NOBODY can see past a corner. The sim faithfully reproduced
        # that blindness. Device probe (hook 17g, `s.wp`): a HEADING-AWARE walk resolves the full
        # 6-cell horizon on 100% of the frames where the plain walk falls short (56.8% of all frames,
        # mean +3.46 cells) — the game exposes the data, the hook simply was not following it.
        # WHY THIS MATTERS: 61-70% of every policy's WASTED jumps sit within 3 cells of a junction, and
        # they are ~7% lethal on device while the human (who can SEE past the corner) hedges 2.4x MORE
        # and never dies from it. The blindness is the cause; it does not exist in-sim, which is why no
        # amount of sim training ever fixed the user-reported "random jump after a turn".
        # THE ONE CASE THAT MUST STAY BLIND: a real T-junction where BOTH directions exist (32.6% of
        # device junction frames). There the continuation depends on a decision not yet made, so the
        # walk must REFUSE rather than feed a confidently WRONG lookahead. Gated (default off = the
        # old always-blind behaviour, so equivalence baselines and obs62/63 policies are untouched).
        if self.cfg.get("turn_aware_lookahead"):
            def _blocked(c: dict) -> bool:
                return bool(c.get("is_turn")) and bool(c.get("can_turn_left")) and bool(c.get("can_turn_right"))
        else:
            def _blocked(c: dict) -> bool:
                return bool(c.get("is_turn"))
        _blk0, _blk1, _blk2 = _blocked(cell), _blocked(nxt), _blocked(nxt2)
        nobst = int(nxt.get("obstacles", 0))

        on_ground = self._jump_t <= 0.0 and not self._is_dead
        state = {
            "is_dead": bool(self._is_dead),
            "is_jumping": self._jump_t > 0.0,
            "is_sliding": self._slide_t > 0.0,
            "is_on_ground": bool(on_ground),
            "is_stumbling": self._stumble_t > 0.0,
            "has_invincibility": False,
            "has_angel_wings": False,
            "has_vacuum": False,
            "has_boost": False,
            "is_mega_boost": False,
            "is_strafing": abs(self._lat_target - self._lane) > 0.5,   # tilted + still gliding
            "score": int(self._score),
            "velocity": float(self._velocity),
            "path_type": int(cell.get("type", 1)),
            "path_level": int(cell.get("level", 2)),
            "facing": int(self._facing),
            "must_jump": bool(must_jump),
            "must_slide": bool(must_slide),
            "is_game_over": bool(self._is_dead),   # game-over follows death in this model
            "can_turn_left": bool(can_l),
            "can_turn_right": bool(can_r),
            "is_turn": bool(is_turn),
            "distance": float(self._distance_real),
            "next_is_turn": bool(nxt.get("is_turn")),
            "next_can_turn_left": bool(nxt.get("can_turn_left")),
            "next_can_turn_right": bool(nxt.get("can_turn_right")),
            "next_must_jump": nobst in _MUST_JUMP,
            "next_must_slide": nobst in _MUST_SLIDE,
            "next_wide_gap": bool(nxt.get("wide_gap")),   # a wide gap ahead (long hole OR zone crevasse)
            # SOFT must-jump = a step-DOWN to low ground, crossable ON FOOT (stepdown_grounded_ok).
            # NOT part of the observation (obs.py builds from explicit field lists, so OBS_LEN is
            # unchanged and no model needs re-surgery) -- it exists so the REWARD can tell a required
            # jump from an optional one. The device streams must_jump=True here too, so the policy's
            # INPUT stays faithful; only the shaping learns the difference.
            # HORIZON (WAVE 17): cells to the next must-JUMP / must-SLIDE within HORIZON_CELLS.
            # The scan stops at a TRUE T-junction (both directions available) because the
            # continuation is genuinely undecided there -- that is exactly what the hook's
            # turn-aware neighbour walk refuses to resolve (17h), so the sim must be blind in the
            # same place or the policy would train on information the phone cannot give it.
            "dist_next_must_jump": self._dist_next_hazard(_MUST_JUMP),
            "dist_next_must_slide": self._dist_next_hazard(_MUST_SLIDE),
            "soft_stepdown": bool(cell.get("stepdown_low")),
            "next_soft_stepdown": bool(nxt.get("stepdown_low")),
            # ZONE-CREVASSE ahead: the next cell is an altitude transition (LevelChange) -> jump LATE (a
            # step-UP needs it, a step-down tolerates it); a 2-cell LONG HOLE (OBST_HOLE) needs an EARLIER
            # takeoff to span it. The policy distinguishes the two jump timings via this. DEPLOYABLE: the
            # device hook emits it from next_path_type ∈ 33-36 (the future PE_Level is unreadable=0, but
            # the SingleLevelStepDownCap path_type is reliable). Was `stepup` (step-up only, undeployable).
            "next_ground_change_higher": int(nxt.get("obstacles", 0)) == OBST_LEVEL_CHANGE,
            # upcoming altitude 2-cell lookahead (device-verified readable, probe 16q): the cap/ledge cells
            # carry the NEW zone level, so sign(next2_path_level - path_level) = the crevasse direction.
            # JUNCTION DROPOUT [2026-07-05, DEVICE-MEASURED v27_deviceeval]: the hook walks the PE
            # neighbor chain, which BREAKS through a turn — on a junction cell both reads return -1
            # (178/202 on-junction frames), and on the approach cell next2 (past the junction) is -1.
            # The sim MUST reproduce this: obs30 policies fine-tuned on always-valid sim levels got a
            # junction-frame input spike on device -> 1-frame swipe hesitation (cp~0.6 > CanTurn end
            # 0.56) -> the v26/v27/v30 missed-turn deaths v14 (zero-init cols) was immune to.
            "next_path_level": -1 if _blk0 else int(nxt.get("level", -1)),
            "next2_path_level": -1 if (_blk0 or _blk1)
                                else int(nxt2.get("level", -1)),
            # 2-cell OBSTACLE lookahead (chained-obstacle fix): the obstacle class of cell+2 (the cell a
            # jump over cell+1 LANDS the runner airborne into). SAME junction dropout as next2_path_level —
            # the hook's PE neighbor walk breaks through turns, so both read False on/through a junction.
            "next2_must_jump": False if (_blk0 or _blk1)
                               else int(nxt2.get("obstacles", 0)) in _MUST_JUMP,
            "next2_must_slide": False if (_blk0 or _blk1)
                                else int(nxt2.get("obstacles", 0)) in _MUST_SLIDE,
            # 3-cell OBSTACLE lookahead (hedge-jump fix 2026-07-08): a jump spans ~2.8 cells, so 2-cell
            # lookahead is SHORTER than one jump — the policy can't know whether a jump launched now lands
            # clear. next3 extends the horizon past the span. SAME junction-dropout CHAIN (any turn cell
            # between here and cell+3 nulls the read — the hook's PE walk breaks through turns).
            "next3_must_jump": False if (_blk0 or _blk1 or _blk2)
                               else int(nxt3.get("obstacles", 0)) in _MUST_JUMP,
            "next3_must_slide": False if (_blk0 or _blk1 or _blk2)
                                else int(nxt3.get("obstacles", 0)) in _MUST_SLIDE,
            "narrow_target_lane": nxt.get("narrow_lane"),  # safe lane for the upcoming narrow (else None)
            # 2-ahead NARROW SIDE (WAVE 9, obs 62->63): the jump->narrow mid-air tilt fix. The human tilts
            # DURING the jump toward the coming strip (sees it 3+ cells out); narrow_target_lane fires only
            # 1 cell ahead -> at a jump takeoff the policy was structurally BLIND to the side and strafed
            # last-second (USER-observed). SAME junction dropout as next2_path_level (the hook's PE walk
            # breaks through turns -> None on/through a junction).
            "next2_narrow_lane": None if (_blk0 or _blk1)
                                 else nxt2.get("narrow_lane"),
            # NN perception (replaces the host-side assists): cell_progress to TIME turns, the CONTINUOUS
            # lane (real device units) so obs.lane_to_obs maps sim/real identically (incl. near-edge).
            "cell_progress": self._cell_progress(),
            "lane": float(self._lane),
            # STUMBLE state (device fields): time since the last stumble (counts up; ~2s = "safe" again)
            # and the running stumble count. is_stumbling (above) = _stumble_t > 0.
            "stumble_kill_timer": float(self._stumble_kill_t),
            "stumble_count": int(self._stumble_count),
        }
        return state

    def _build_obs(self, state: dict):
        """Push this frame's base obs and return the stacked (frame_stack) + action-history vector."""
        self._frames.append(O.state_to_obs(state))
        if len(self._frames) > self.frame_stack:
            self._frames = self._frames[-self.frame_stack:]
        return O.stacked_obs(self._frames, self._act_hist, self.frame_stack, self.action_history_len)

    # ------------------------------------------- fidelity harness: raw/set state
    def raw_state(self) -> dict:
        """Current 22-field raw-state dict (for the diff harness)."""
        return self._snapshot()

    def set_state(self, raw_state: dict, hidden: Optional[dict] = None) -> None:
        """Force the sim to a given real (raw_state, hidden) — for one-step fidelity tests.

        ``raw_state`` is the 22-field dict; ``hidden`` carries the extra replay state
        (distance, lane, the active-timer remainders, death_type) per SIM_CONTRACT §1.
        Missing/``None`` hidden fields fall back to sensible derivations from raw_state.
        """
        hidden = hidden or {}
        s = raw_state

        self._is_dead = bool(s.get("is_dead", False))
        self._death_type = hidden.get("death_type")
        self._velocity = float(s.get("velocity", self.cfg["start_velocity"]))
        self._facing = int(s.get("facing", DIR_N)) if int(s.get("facing", DIR_N)) > 0 else DIR_N
        self._score = float(s.get("score", self.cfg["start_score"]))

        lane = hidden.get("lane")
        self._lane = float(lane) if lane is not None else 0.0
        self._tilt_dir = 0
        self._stumble_kill_t = float(hidden.get("stumble_kill_timer") or self.cfg["stumble_kill_window"])
        self._stumble_count = int(hidden.get("stumble_count") or 0)
        self._stumble_vel = None
        self._distance = float(hidden.get("distance") or 0.0)
        self._distance_real = float(hidden.get("distance") or self.cfg["start_distance"])
        self._ground_height = float(hidden.get("ground_height") or 0.0)

        # Reconstruct transient timers: prefer the recorded remainders, else derive a full
        # duration from the boolean flag (so the flag stays set for one step at least).
        self._jump_t = self._timer_from(hidden.get("slide_t") if False else None,
                                        s.get("is_jumping"), self.cfg["jump_duration"],
                                        explicit=hidden.get("jump_t"))
        self._slide_t = self._timer_from(None, s.get("is_sliding"), self.cfg["slide_duration"],
                                        explicit=hidden.get("slide_t"))
        self._strafe_t = self._timer_from(None, s.get("is_strafing"), self.cfg["strafe_duration"],
                                         explicit=hidden.get("strafe_t"))
        self._lat_target = float(hidden.get("lat_target") or 0.0)
        self._lat_cmd = self._lat_target
        self._lat_target_buf = []
        self._gap_grace_cell = None
        self._stumble_t = self._timer_from(None, s.get("is_stumbling"), self.cfg["stumble_duration"],
                                          explicit=hidden.get("stumble_t"))

        # Seed the current cell from the raw fields so raw_state() echoes it back exactly.
        cell = _element(
            type=int(s.get("path_type", 1)) if int(s.get("path_type", -1)) >= 0 else 1,
            level=int(s.get("path_level", 2)) if int(s.get("path_level", -1)) >= 0 else 2,
            obstacles=self._obstacle_from_flags(s),
            is_turn=bool(s.get("is_turn")),
            can_turn_left=bool(s.get("can_turn_left")),
            can_turn_right=bool(s.get("can_turn_right")),
        )
        # rebuild the track window with this cell as the current one.
        if self.track_source == "replay":
            # The seeded current cell IS injected element 0 (the recorded nearest cell == the
            # front of the recorded track window). So the NEXT injected cell to consume is
            # index 1 -- starting the cursor at 0 duplicates element 0 (observed: the Temple
            # start-piece (type 37) appeared twice, holding path_type=37 for ~11 steps instead
            # of the real ~4).
            self._replay_cursor = 1
            self._track = [cell]
            self._cell_index = 0
            # Anchor cell-crossing to the recorded distance: the player starts partway into the
            # (partial) Temple start-piece, so seed dist_into_cell from the calibrated offset.
            self._last_ext_distance = float(self._distance)
            self._dist_into_cell = float(self.cfg["replay_start_into_cell"])
            self._extend_track()
        else:
            self._track = [cell]
            self._cell_index = 0
            self._dist_into_cell = 0.0
            self._extend_track()

        self._prev_state = self._snapshot()
        self._prev_score = int(self._prev_state["score"])

    @staticmethod
    def _timer_from(_unused, flag, full_duration, explicit=None) -> float:
        if explicit is not None:
            return max(0.0, float(explicit))
        return float(full_duration) if bool(flag) else 0.0

    @staticmethod
    def _obstacle_from_flags(s: dict) -> int:
        """Pick a representative obstacle enum consistent with must_jump/must_slide."""
        if s.get("must_jump"):
            return OBST_HOLE
        if s.get("must_slide"):
            return OBST_TREE_SLIDE
        return OBST_NONE

    def render(self):
        return None

    def close(self):
        return None


# Register on import (SIM_CONTRACT §2). The env truncates itself via max_episode_steps, so
# register without a TimeLimit wrapper (mirrors TempleRun-v0). Guard against double-register
# (this module is imported by __init__ AND runnable as -m) to avoid gymnasium's override warning.
try:
    from gymnasium.envs.registration import register as _register
    from gymnasium.envs.registration import registry as _registry

    if "TempleRunSim-v0" not in _registry:
        _register(
            id="TempleRunSim-v0",
            entry_point="templerun_rl.sim_env:TempleRunSimEnv",
            max_episode_steps=None,
            disable_env_checker=False,
        )
except Exception:  # noqa: BLE001  (re-import / already-registered is fine)
    pass


# =============================================================================
# Smoke test: 1 procedural episode with a reactive policy. Asserts survival > 50
# steps and score > 0, then prints a summary. Run: python -m templerun_rl.sim_env
# =============================================================================
def _reactive_policy(state: dict, last_action: int) -> int:
    """A simple hand-coded reactive controller. Acts on the 1-cell LOOKAHEAD (next_must_jump/slide)
    because the sim resolves obstacles on cell ENTRY — reacting only when the obstacle is the nearest
    cell lands the jump/slide a step too late. Turns at ~cell-center (once, not mid-strafe); strafes to
    a narrow's safe lane when signalled (otherwise the lane auto-recenters, so it never strafes on a
    straight -> never wall-stumbles)."""
    cp = float(state.get("cell_progress", 1.0))
    if state.get("is_turn"):
        if cp < 0.40 or state.get("is_strafing"):
            return int(C.Action.NOOP)
        if state.get("can_turn_left"):
            return int(C.Action.TURN_LEFT)
        if state.get("can_turn_right"):
            return int(C.Action.TURN_RIGHT)
        return int(C.Action.NOOP)
    if state.get("next_wide_gap"):   # 2-cell gap: jump LATE (at the gap edge) so airtime spans both
        return int(C.Action.JUMP) if cp >= 0.62 else int(C.Action.NOOP)
    if state.get("must_jump") or state.get("next_must_jump"):
        return int(C.Action.JUMP)
    if state.get("must_slide") or state.get("next_must_slide"):
        return int(C.Action.SLIDE)
    ntl = state.get("narrow_target_lane")
    if ntl is not None:
        target = int(ntl) * LANE_STEP
        lane = float(state.get("lane", 0.0))
        if lane < target - 0.4 * LANE_STEP:
            return int(C.Action.STRAFE_RIGHT)
        if lane > target + 0.4 * LANE_STEP:
            return int(C.Action.STRAFE_LEFT)
    return int(C.Action.NOOP)


def _smoke_test() -> None:
    env = TempleRunSimEnv(track_source="procedural", seed=7)
    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs), "obs not in observation_space"
    assert obs.shape == (O.OBS_LEN,), f"obs shape {obs.shape} != ({O.OBS_LEN},)"

    state = info["raw_state"]
    last_action = int(C.Action.NOOP)
    total_reward = 0.0
    steps = 0
    start_score = state["score"]
    max_score = start_score
    deaths = 0

    for _ in range(400):
        action = _reactive_policy(state, last_action)
        obs, reward, terminated, truncated, info = env.step(action)
        last_action = action
        state = info["raw_state"]
        total_reward += reward
        steps += 1
        max_score = max(max_score, state["score"])
        if terminated or truncated:
            deaths += int(terminated)
            break

    score_gain = max_score - start_score
    print("=" * 60)
    print("TempleRunSim-v0 smoke test (procedural, reactive policy)")
    print("=" * 60)
    print(f"  steps survived   : {steps}")
    print(f"  terminated/death : {bool(deaths)}  (death_type={info.get('death_type')})")
    print(f"  start score      : {start_score}")
    print(f"  max score        : {max_score}   (gain {score_gain})")
    print(f"  total reward     : {total_reward:+.3f}")
    print(f"  final velocity   : {state['velocity']:.1f}")
    print(f"  obs space        : {env.observation_space}")
    print(f"  action space     : {env.action_space}")

    # also verify replay mode + set_state round-trips (used by the fidelity harness).
    sample_track = [
        _element(type=1, level=2, obstacles=OBST_HOLE),
        _element(type=1, level=2, obstacles=OBST_TREE_SLIDE),
        _element(type=3, level=2, is_turn=True, can_turn_left=True),
    ]
    renv = TempleRunSimEnv(track_source=sample_track, seed=1)
    robs, rinfo = renv.reset()
    renv.set_state(rinfo["raw_state"], {"distance": 0.0, "lane": 0})
    rs = renv.raw_state()
    # the snapshot is a SUPERSET of the device STATE_FIELDS contract (it adds sim-perception fields
    # cell_progress/lane/narrow_target_lane/next_wide_gap + stumble_kill_timer/stumble_count).
    assert set(C.STATE_FIELDS).issubset(rs.keys()), "raw_state missing STATE_FIELDS contract keys"
    print(f"  replay/set_state : OK (raw_state has all {len(C.STATE_FIELDS)} contract fields)")

    assert steps > 50, f"reactive policy died too early ({steps} steps)"
    assert score_gain > 0, f"no score accrued (gain={score_gain})"
    print("\nSMOKE TEST PASSED  (survived >50 steps, scored >0)")


if __name__ == "__main__":
    _smoke_test()
