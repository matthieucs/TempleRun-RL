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
    # POST-TURN OBSTACLE [DEVICE-MEASURED 2026-07-08]: the game places obstacles on the FIRST cell after a
    # junction (72 onsets ~1 cell post-turn across 6 captures; bimodal with the normal bulk at 2.5-5 cells);
    # min_clear_after_turn(2) meant the sim NEVER did -> post-turn deaths end the 4-7k runs (7/15 of the
    # pt(-1/1/2) deaths follow a turn by <12 frames). Standard obstacle sampler (holes stay 1-cell). 0=off.
    "turnobstacle_prob": 0.0,           # P(a standard obstacle on the first post-junction cell)
    "min_space_between_obstacles": 6,   # [READ] sMinSpaceBetweenObstacles (cells)
    "min_space_between_turns": 12,      # [READ] sMinSpaceBetweenTurns (cells)
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
    # TURN EXECUTION [DEVICE-MEASURED 2026-07-02, scripts/probe_turn_exec.py — one swipe at controlled
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
    # jump must LAND past the far edge. Jump span (cells) fits span = a*velocity + b (NEGATIVE
    # intercept — NOT proportional; the sim's jump_dur*v/cell_len=v/75 has the wrong slope 0.0133 and
    # b=0, underestimating high-v span). Only the LAST gap cell is checked (next cell solid). OFF by
    # default (no-op -> equivalence stays green); enabled in the device-true / training config.
    "wide_gap_landing_model": False,
    "wide_gap_span_a": 0.0204,         # cells of jump span per unit velocity (device fit)
    "wide_gap_span_b": -0.94,          # span intercept (cells) — the arc's fixed shortfall
    # CREVASSE APEX for ALL level-changes [DEVICE-MEASURED 2026-07-06, vel_y arcs + crevasse_clips.py].
    # The step-UP apex model only fired on level-UP crevasses; step-DOWN / high-ground crevasses used
    # the lenient airborne check -> the sim never penalised an EARLY crevasse jump, so v47 drifted to
    # takeoff cp 0.33 (device: 0.63) and took repeated hits (slowdowns) on high-ground crevasses. The
    # vel_y arcs show step-down ALSO needs near-apex at the far cell (pt35 cp0.96 vel_y~4 = apex). When
    # True, the apex check (+ stepup_clip_band graze->stumble) fires on EVERY crevasse ledge cell. OFF
    # by default (no-op -> equivalence green); enabled in the device-true / training config.
    "crevasse_apex_all": False,
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
    "obstacle_type_weights": {
        1: 0.22,   # Hole         -> jump
        4: 0.22,   # TreeJump     -> jump
        5: 0.22,   # TreeSlide    -> slide
        6: 0.14,   # FlameTower   -> slide
        3: 0.12,   # TreeStumble  -> stumble (survivable)
        2: 0.08,   # Narrow       -> strafe to center (survivable in v0)
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
    # [DEVICE-CALIBRATED 2026-07-01, controlled strafe probe scripts/probe_strafe.py, 8 runs] ONE strafe
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

# PathDirection enum (SPEC §0): None0 North1 East2 South3 West4.
DIR_NONE, DIR_N, DIR_E, DIR_S, DIR_W = 0, 1, 2, 3, 4
# left/right of a heading (turning relative to current facing).
_LEFT_OF = {DIR_N: DIR_W, DIR_W: DIR_S, DIR_S: DIR_E, DIR_E: DIR_N}
_RIGHT_OF = {DIR_N: DIR_E, DIR_E: DIR_S, DIR_S: DIR_W, DIR_W: DIR_N}


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
                    w.update(v)
                    self.cfg[k] = w
                else:
                    self.cfg[k] = v
        # Immutable base; `self.cfg` is re-derived from it each reset (domain randomization mutates
        # the per-episode copy, never the base).
        self._base_cfg = dict(self.cfg)
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
        self._pending_gap_stepup = False   # the pending crevasse's 2nd cell is a raised ledge (step-UP)
        self._pending_gap_obst = 0
        self._pending_gap_type = 1
        self._pending_gap_level = 2
        # NARROW strip continuation: device strips span 1-7 cells (commonly 2-3), so the runner must
        # HOLD the safe lane across N cells, not just match it for one frame (the old 1-cell model let a
        # policy strafe back too early and fall off the 2nd/3rd cell on the phone).
        self._pending_narrow = 0
        self._pending_narrow_lane = 0
        self._pending_sequence: list[dict] = []   # compound stacked-obstacle queue (user blindspot)
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
        return self._next_cell_procedural()

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
            for _ in range(max(1, int(cfg.get("jumpgap_hole_cells", 2)))):
                e = _element(type=12, level=level, obstacles=OBST_HOLE, is_turn=False)
                e["wide_gap"] = True
                seq.append(e)
        self._pending_post_gap.extend(seq)

    @staticmethod
    def _narrow_ptype(lane: int) -> int:
        return 19 if lane == -1 else 22

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
        narrow_lane = None   # if set (-1/0/+1), this cell is a narrow needing that lane (else Fall)

        # Difficulty-ramped turn spacing/probability (lerp baseline -> late by distance).
        diff = self._difficulty()
        min_turns = (float(cfg["min_space_between_turns"]) * (1.0 - diff)
                     + float(cfg["min_space_between_turns_late"]) * diff)
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
                and self._cells_since_obstacle >= int(cfg["min_space_between_obstacles"])
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
            self._pending_gap = 1          # -> 1 continuation cell = the raised ledge
            self._pending_gap_obst = OBST_LEVEL_CHANGE
            self._pending_gap_type = 35    # ~ real SingleLevelStepDownCap path_type (obs match)
            self._pending_gap_level = int(self._zone_level)
            self._pending_gap_stepup = bool(self._zone_level > prev_zone)  # ledge cell = step-UP
            ptype = 35
            wide = True
            stepup = False                 # cell 1 is the GAP (airborne), not the ledge; ledge is cell 2

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
            obstacle = self._sample_obstacle_type()
            self._cells_since_obstacle = 0

        # A turn cell carries no obstacle (the player must turn here); the cell right after a junction
        # is also kept clear (min_clear_after_turn) — a must-jump obstacle there is unclearable (you
        # can't turn AND jump on one crossing).
        if (obstacle == OBST_NONE and not is_turn
                and self._cells_since_turn >= int(cfg["min_clear_after_turn"])
                and self._cells_since_obstacle >= int(cfg["min_space_between_obstacles"])):
            place = self._rng.random() < float(cfg["obstacle_prob"])
            # honor max back-to-back: if we've hit the cap, force a gap.
            if self._b2b_obstacles >= int(cfg["max_back_to_back_obstacles"]):
                place = False
            if place:
                self._cells_since_obstacle = 0
                self._b2b_obstacles += 1
                if (self._distance_real >= float(cfg["narrows_after_distance"])
                        and self._rng.random() < float(cfg["narrow_prob"])):
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
                    obstacle = self._sample_obstacle_type()
                    # a HOLE is sometimes a 2-cell LONG hole (path_type 12/13/15/16 on device).
                    if obstacle == OBST_HOLE and self._rng.random() < float(cfg["wide_gap_prob"]):
                        self._pending_gap = int(cfg["wide_gap_extra_cells"])
                        self._pending_gap_obst = OBST_HOLE
                        self._pending_gap_type = 12   # ~ real HoleLong path_type (obs match)
                        self._pending_gap_level = level
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
        e["narrow_lane"] = narrow_lane
        return e

    def _sample_obstacle_type(self) -> int:
        weights = dict(self.cfg["obstacle_type_weights"])
        # Slides (TreeSlide=5, FlameTower=6) only appear past slides_after_distance — the baseline
        # statics have sAllowTreeSlide/FlameTowers = False, so early runs are jump/stumble only.
        if self._distance_real < float(self.cfg["slides_after_distance"]):
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
        if a == int(C.Action.JUMP):
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
                    return
                lat = int(cfg.get("jump_exec_lat_frames", 0))
                if lat > 0 and self._pending_jump_frames <= 0:
                    # DEVICE JUMP LATENCY: don't fire now — schedule the takeoff `lat` (+/- jitter) frames
                    # later (see jump_exec_lat_frames). The policy must PRE-AIM (predicted_jump).
                    jit = int(cfg.get("jump_exec_lat_jitter", 0))
                    self._pending_jump_frames = max(1, lat + (self._rng.integers(-jit, jit + 1) if jit > 0 else 0))
                elif lat <= 0:
                    self._jump_t = float(cfg["jump_duration"])
                    # record WHERE this jump took off (longitudinal pos) — the step-up crevasse checks the
                    # runner jumped late enough (near the gap edge) so an early jump fails on the raised wall.
                    self._jump_takeoff_pos = self._long_pos()
        elif a == int(C.Action.SLIDE):
            if self._slide_t <= 0.0 and self._jump_t <= 0.0:
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
                self._jump_t = float(cfg["jump_duration"])
                self._jump_takeoff_pos = self._long_pos()

        # tick down transient flags.
        self._jump_t = max(0.0, self._jump_t - dt)
        self._slide_t = max(0.0, self._slide_t - dt)
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
            cell_len = float(cfg["cell_length_real"])
        else:
            self._distance_real += self._velocity * dt * float(cfg["dist_per_velsec"])
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
        if not self._is_dead and self._jump_t > 0.0:
            _oc = int(self._current_cell().get("obstacles", 0))
            if _oc in _MUST_SLIDE:
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
                self._kill(2)       # not airborne at the step-up ledge -> fell in the gap
                return
            span = float(self.cfg["jump_duration"]) * self._velocity / float(self.cfg["cell_length"])
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
                    self._kill(2)    # arc well below the raised cap top -> deep clip -> Ledge
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
                    self._kill(2)   # not high enough at the raised ledge -> clip it -> Fall
            return
        if obst in _MUST_JUMP:
            # Non-step-up wide gap (long hole / step-DOWN) or 1-cell hole: must be AIRBORNE crossing in.
            # wide gaps get a lip-grace (the hole starts a bit into the cell; device jumps at cprog
            # 0.98-1.02 still cleared). 1-cell holes keep the strict on-entry check.
            need = float(self.cfg["wide_gap_min_airtime"]) if cell.get("wide_gap") else 0.0
            if self._jump_t <= need:
                if cell.get("wide_gap") and float(self.cfg["gap_entry_grace"]) > 0.0:
                    self._gap_grace_cell = self._cell_index
                else:
                    self._kill(1 if obst == OBST_HOLE else 2)   # Fall(hole) / LevelChange/Tree
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
                        self._kill(2)   # arc lands inside the gap -> clip the far wall (Ledge)
                        return
        elif obst in _MUST_SLIDE:
            # die if NOT sliding OR AIRBORNE (jumped into the low bar — device-fatal, 2026-07-03).
            if self._slide_t <= 0.0 or self._jump_t > 0.0:
                self._kill(5 if obst == OBST_FLAME_TOWER else 3)  # Burnt / Slide
        # OBST_NARROW / OBST_NONE: no death here (narrow handled above; LevelChange is in _MUST_JUMP).

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
            "next_path_level": -1 if cell.get("is_turn") else int(nxt.get("level", -1)),
            "next2_path_level": -1 if (cell.get("is_turn") or nxt.get("is_turn"))
                                else int(nxt2.get("level", -1)),
            # 2-cell OBSTACLE lookahead (chained-obstacle fix): the obstacle class of cell+2 (the cell a
            # jump over cell+1 LANDS the runner airborne into). SAME junction dropout as next2_path_level —
            # the hook's PE neighbor walk breaks through turns, so both read False on/through a junction.
            "next2_must_jump": False if (cell.get("is_turn") or nxt.get("is_turn"))
                               else int(nxt2.get("obstacles", 0)) in _MUST_JUMP,
            "next2_must_slide": False if (cell.get("is_turn") or nxt.get("is_turn"))
                                else int(nxt2.get("obstacles", 0)) in _MUST_SLIDE,
            # 3-cell OBSTACLE lookahead (hedge-jump fix 2026-07-08): a jump spans ~2.8 cells, so 2-cell
            # lookahead is SHORTER than one jump — the policy can't know whether a jump launched now lands
            # clear. next3 extends the horizon past the span. SAME junction-dropout CHAIN (any turn cell
            # between here and cell+3 nulls the read — the hook's PE walk breaks through turns).
            "next3_must_jump": False if (cell.get("is_turn") or nxt.get("is_turn") or nxt2.get("is_turn"))
                               else int(nxt3.get("obstacles", 0)) in _MUST_JUMP,
            "next3_must_slide": False if (cell.get("is_turn") or nxt.get("is_turn") or nxt2.get("is_turn"))
                                else int(nxt3.get("obstacles", 0)) in _MUST_SLIDE,
            "narrow_target_lane": nxt.get("narrow_lane"),  # safe lane for the upcoming narrow (else None)
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
