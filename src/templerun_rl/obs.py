"""Observation encoding: raw game-state dict -> normalized float32 vector.

The policy sees an OBS_LEN-element ``Box(0, 1)`` vector (see OBS_FIELDS; 28 as of 2026-07-04). We deliberately exclude the raw
``score`` (unbounded / accumulating -> would break MlpPolicy normalization); it feeds
the reward instead and is always available in ``info["raw_state"]``.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from . import constants as C

# Normalization constants (see docs/OFFSETS.md for value semantics).
VEL_MIN = 100.0   # run velocity starts ~100
VEL_MAX = 220.0   # ramps to ~210; 220 leaves headroom
PATH_TYPE_MAX = 37.0
PATH_LEVEL_MAX = 4.0
FACING_MAX = 4.0   # PathDirection: None=0, N=1, E=2, S=3, W=4

# Observation vector layout: the bool flags (incl. 1-cell lookahead), then normalized scalars.
OBS_BOOL_FIELDS = (
    "is_dead", "is_jumping", "is_sliding", "is_on_ground", "is_stumbling",
    # power-ups REMOVED 2026-07-04 (user): never spawn on device (rate 0) nor in the sim -> always-0 dead
    # inputs. has_invincibility/has_angel_wings/has_vacuum/has_boost/is_mega_boost dropped. OBS_LEN 32->27.
    "is_strafing", "must_jump", "must_slide", "is_game_over",
    "can_turn_left", "can_turn_right", "is_turn",
    # 1-cell lookahead — anticipation (see a junction/obstacle before it's the nearest cell).
    "next_is_turn", "next_can_turn_left", "next_can_turn_right", "next_must_jump", "next_must_slide",
    # the upcoming must-jump is a wide gap -> jump LATE. next_ground_change_higher distinguishes a
    # 1-cell step-UP crevasse (raised ledge, needs a LATE takeoff, device probe 2026-07-03) from a
    # 2-cell long hole (needs an EARLIER takeoff to span both cells) — different jump timings.
    "next_wide_gap", "next_ground_change_higher",
    # 2-cell OBSTACLE lookahead (2026-07-07) — the chained-obstacle death fix. A jump spans ~2.75 cells,
    # so a jump over cell+1 lands the runner airborne INTO cell+2; if cell+2 is a slide-obstacle
    # (flame/tree-slide) the runner can't slide while airborne -> Burnt (device death @6216m: hole-jump ->
    # flame tower right behind). Giving the policy the 2-ahead must-jump/slide lets it defer/re-time the
    # jump. SAME junction-dropout as next2_path_level (False through a turn). BREAKING: obs 58->60.
    "next2_must_jump", "next2_must_slide",
    # 3-cell OBSTACLE lookahead (2026-07-08) — the HEDGE-JUMP fix. A jump spans ~2.8 cells at speed, so a
    # 2-cell lookahead is SHORTER than one jump: the policy cannot know whether a jump launched NOW lands
    # clear -> it hedge-jumps (held% ~83 on device) and dies landing on unseen hole lips / re-jumping into
    # chained holes (41% of deaths = holes). next3 extends the horizon past the jump span (the same
    # mechanism as next2 = the +108% device win). SAME junction-dropout chain (False if any of the 3 cells
    # to it is a turn). BREAKING: obs 60->62 (hook 17a reads the 3rd neighbor).
    "next3_must_jump", "next3_must_slide",
)
OBS_SCALAR_FIELDS = ("velocity", "path_type", "path_level", "facing")
# Perception added 2026-06-19 so the NN can time turns + control its lane natively (replacing the
# host-side assists): cell_progress = longitudinal position in the nearest cell (0..1; lets the policy
# fire a turn at cell-center instead of too early), lane = which of the 3 lanes (L/C/R) the runner is in
# (for narrow strips). Both come from the hook (CharacterPlayer.GetPosition2D + PathElement.Rectangle)
# and from the sim's own state — they are OBSERVATIONS, not actions.
# next_path_level / next2_path_level: the altitude Level (PathLevel 1/2/3 = low/mid/high) of the cell 1 and
# 2 ahead. DEVICE-VERIFIED READABLE (probe 16q, 10/10): sign(next2_path_level - path_level) = the true
# crevasse direction (up/down), which `next_ground_change_higher` (non-directional) could NOT convey. Gives
# the NN the SIGNED transition (low->mid vs high->mid etc.) + magnitude, so it stops blindly jumping every
# cap. The old "future Level reads 0" assumption was wrong for this build.
OBS_EXTRA_FIELDS = ("cell_progress", "lane", "narrow_target_lane", "predicted_landing", "predicted_jump",
                    "next_path_level", "next2_path_level",
                    # next2_narrow_lane (WAVE 9 2026-07-09, obs 62->63): the 2-ahead narrow's SIDE — the
                    # jump->narrow MID-AIR TILT fix. The human tilts DURING a jump toward the coming strip
                    # (sees it 3+ cells out); narrow_target_lane fires only 1 cell ahead, so at a jump
                    # takeoff the policy was structurally BLIND to the side -> the USER-observed last-
                    # second strafe (v54's jumpnarrow trained the scenario but the late cp~0.9 commit
                    # persisted = this obs gap, not a training gap). Same encoding as narrow_target_lane;
                    # junction dropout (None -> center) through turns, mirroring next2_path_level.
                    "next2_narrow_lane",
                    # predicted_turn (WAVE 11, obs 63->64): the approach-cell swipe cue — margin to the
                    # junction consume deadline as one velocity-invariant scalar (see predicted_turn_obs).
                    # The dead-letter fix as an OBSERVATION (wave-10: pressure alone plateaued at +40%
                    # and eroded crevasse skill in a modified mix). Host-derived; hook 17f unchanged.
                    "predicted_turn",
                    # landing_hazard (WAVE 16, obs 64->65): WHERE MY JUMP ARC PUTS ME DOWN, and whether
                    # something is there. THE measured death mode: a behavioural audit of 44 device
                    # deaths (scripts/death_behaviour.py) found 55% "COMPOUND TRAP" (airborne into the
                    # fatal hazard from a jump that WAS needed) + 5% "RANDOM-JUMP TRAP" = 60% of all
                    # deaths are "my arc landed me somewhere fatal", and the takeoff->fatal distance
                    # clusters at 2.0-3.5 cells = exactly the jump span. 84% of deaths had another
                    # hazard within 5 cells behind the fatal one, 33/37 of those at gap 1-2.
                    # The policy HAS next1/next2/next3_must_jump, but relating them to its own arc is
                    # velocity-dependent arithmetic it demonstrably does not do. This collapses that
                    # into ONE velocity-invariant scalar -- the same trick that worked twice before
                    # (predicted_landing broke the turn wall; predicted_jump did crevasse takeoff).
                    # Host-derived from already-streamed fields: NO hook change, NO sim change.
                    "landing_hazard",
                    # slide_landing_hazard (WAVE 18, obs 65->66): THE SAME QUESTION FOR THE HAZARD
                    # YOU CANNOT ANSWER ONCE AIRBORNE. landing_hazard asks "does my arc come down on
                    # a must-JUMP cell" — but a must-JUMP hazard is at least survivable in principle
                    # (land, jump again). A must-SLIDE hazard is not: you cannot slide in mid-air, so
                    # by the time it is in front of you the only action that saves you is already
                    # unavailable. The counterfactual lives entirely at TAKEOFF.
                    # MEASURED over 100 device events (every takeoff whose predicted arc spans a
                    # must-slide hazard, pooled over 45 captures):
                    #                    n    takeoff cp   span   hazard at   ARC SLACK   pressed slide
                    #     survived      73       0.53      2.61      +2.0       +1.54          34%
                    #     died          27       0.62      2.69      +3.0       +0.21          89%
                    #   arc_slack = (cell_progress + span) - hazard_offset = how far PAST the hazard
                    #   the arc reaches. Survivors fly over it and land beyond; victims' arcs END ON
                    #   IT. And 89% of the victims DID press slide — they read the situation right and
                    #   died anyway, because they were still in the air. The jump itself was forced in
                    #   100% of these (a must_jump demanded it, 0 discretionary): the policy is not
                    #   choosing badly, it is timing badly, and 0.53-vs-0.62 takeoff cp is the whole
                    #   difference. ~33% of all deaths are this shape (88% of Flame deaths airborne).
                    # int(landing) == hazard_offset is exactly that discriminator (slack <1 vs >=1),
                    # so this is the same one-scalar collapse as landing_hazard, on the must_slide
                    # ladder instead of the must_jump one. Host-derived: NO hook change (17j/17k ok).
                    "slide_landing_hazard",
                    # landing_margin (WAVE 19, obs 66->67): HOW MUCH ROOM my arc leaves, not just
                    # whether it hits. `landing_hazard` is BINARY, so a takeoff that lands 1.05 cells
                    # past a hole (one perturbation from fatal) and one that lands 1.65 past (safe)
                    # both read 0.0. DEVICE-MEASURED over 2034 takeoffs whose arc spans a must-jump
                    # hazard: survivors cluster tightly at margin +1.65 (p25 +1.28, only 2% within
                    # +-0.5 of the hazard) while victims are scattered (p25 +0.35, 38% within +-0.5)
                    # = a 19x enrichment. The champion also takes off with a median lead of ONE frame
                    # vs the HUMAN's four, i.e. it has no cue for where the safe window ENDS, and it
                    # is squeezed between two priced failures: take off early -> the arc lands ON the
                    # hazard (mode A, 25% of deaths); take off late -> no time at all (mode B, 44%).
                    # margin = (cell_progress + height-scaled span) - nearest must-jump offset;
                    # <0 lands short of it, >0 lands past it. 1.0 when the ladder is clear (maximum
                    # room). Host-derived from already-streamed fields: NO hook change.
                    "landing_margin",
                    # ★ HORIZON SCALARS (WAVE 17, obs 65->67) — THE 3-OBSTACLE CHAIN FIX.
                    # USER (2026-07-27, watching v104): "The struggle is really when you have 3
                    # obstacles in a row. It really handles it poorly."  MEASURED on that same device
                    # capture: 3-obstacle chains kill 14.3% per encounter (2 of 14, BOTH on the third
                    # obstacle) while the human dies on 0 of 45. And the reason is structural, not a
                    # training gap: 100% of chains -- device AND human -- have their THIRD obstacle
                    # OUTSIDE the +3-cell boolean horizon at the moment the runner must commit to the
                    # first jump. At v210 a jump lasts 0.8 s and +3 cells is 0.86 s, so the policy can
                    # see barely one jump ahead while the human sees the whole screen.
                    # Two velocity-invariant scalars over a 6-cell horizon: normalised distance to the
                    # next must-JUMP and next must-SLIDE cell (0 = on it, 1.0 = nothing within 6).
                    # Scalars, not 6 more booleans: they carry the RHYTHM (how much room after this
                    # one) in a form that generalises across speed, which is the same trick that made
                    # predicted_landing and landing_hazard work.
                    # ⚠️ NEEDS HOOK 17k: unlike landing_hazard this is NOT host-derivable -- the
                    # stream only carries obstacles out to +3. The hook already walks the PathElement
                    # neighbour chain 6 deep (turn_aware_lookahead, 17h+), so it is a read, not a new
                    # traversal.
                    )
# OPT-IN until the whole pipeline is ready. Turning this on changes OBS_LEN 38 -> 40 (full width
# 66 -> 68), which every obs-66 model cannot load without surgery, and the device needs
# hook 17k to stream the 4..6-cell obstacles. Flip to True only together with: the C++ base_obs
# mirror, hook 17k, and a surgery-widened base checkpoint.
# REVERTED to False 2026-07-28 on the criterion set before training: a learned distance cue must
# make P(JUMP) higher at d=4 than d=6 among states where the +3 booleans are all False (the only
# place this cue carries information). Measured after 40M steps of consolidation: d4-d6 = -0.00005,
# and +0.00006 at 10M, against a +0.0030 bar (where v90's landing_hazard cue sat at 2M). Three arms
# (v109 LR5e-5, v109b LR2e-4, v110 40M) all flat; n_epochs 20 flat too. The policy learned "be
# twitchier in the dark" (30x the base's blind-state jump rate) but never "the hazard is 4 cells
# away". The infrastructure stays -- obs encoding, both sims, hook 17k, surgery -- so flipping this
# back to True is a one-line experiment when there is a reason to believe the training regime can
# actually develop the zero-init columns.
HORIZON_OBS = False
HORIZON_CELLS = 6
if HORIZON_OBS:
    OBS_EXTRA_FIELDS = OBS_EXTRA_FIELDS + ("dist_next_must_jump", "dist_next_must_slide")
OBS_FIELDS = OBS_BOOL_FIELDS + OBS_SCALAR_FIELDS + OBS_EXTRA_FIELDS
OBS_LEN = len(OBS_FIELDS)  # 23 bool/lookahead + 4 scalar + 8 extra = 35 (keep derived from OBS_FIELDS)

# Lane geometry (real device units, 2026-07-01 tilt-controlled model): the lateral position is
# CONTINUOUS and centered on 0 — center 0, one strafe ~ 3.7 -> lanes at -3.7 / 0 / +3.7, side walls
# at ~+-7. Normalizing over a SYMMETRIC [LANE_OBS_MIN, LANE_OBS_MAX] centered on 0 maps center -> 0.5
# and the edges +-7 -> ~0.06/0.94, so "near which wall" is a distinct, recoverable obs (the old
# asymmetric -7..+4.5 range was for the retired home=-1.3 lock-lane model).
LANE_HOME = 0.0
LANE_STEP = 3.7
NARROW_CENTER = 4.5   # [MEASURED 2026-07-02] long-narrow strip center (Region2D bands [2,7]/[3,7])
LANE_OBS_MIN = -8.0   # symmetric, just past the left wall
LANE_OBS_MAX = 8.0    # symmetric, just past the right wall
# kept for the capture/analysis scripts that bucket lane for display (center-0 lane boundaries).
LANE_LEFT_MAX = -1.85
LANE_RIGHT_MIN = 1.85

# PREDICTED-LANDING geometry (2026-07-03, v10): a turn swiped NOW executes ~1 frame (100 ms) LATER
# (device-measured: constant 1-frame lag, 21/21 turns), during which cell_progress advances
# velocity*dt/cell_len. So the runner LANDS at cell_progress + DCPROG_PER_VEL*velocity, and the landing
# offset = (that - 0.44)*45. Feeding this projected-landing cprog as an obs lets the policy aim the
# LANDING (not the swipe) at 0.44 at ANY speed, with no internal velocity math -> fixes the late-swipe
# off-center turn LANDING (dominant device death, CLAUDE.md §21). Device fit slope 0.0016 ~= sim
# dt/cell_len 0.1/60; one shared constant keeps sim-training and device-deploy consistent (<5% on the
# device cell_len, absorbed by DR).
# LANDING-ARC span (cells) — WHERE a jump taken now comes down. Used by landing_hazard.
# ⚠️ 2026-07-27 CORRECTION. The wave-8 value (0.0204*v - 0.94, scratchpad/jump_arc.py) is WRONG: it
# over-states the span by 13-17% at running speed. Three independent measurements agree against it:
#   * HUMAN captures at 30 Hz (the finest sampling available), continuous takeoff->touchdown over
#     n=428 jumps: span = 0.0143*v - 0.11  (2.06 / 2.31 / 2.59 / 2.86 cells at v150/170/190/210)
#   * the SIM under the SAME estimator (scratchpad/jump_span_compare.py): 0.0147*v + 0.06 -- i.e. the
#     sim's own arc was faithful all along, and it was the fit that disagreed
#   * direct AIRTIME: 0.800 s median on device AND human (n=1687), exactly the sim's jump_duration,
#     which forces span = jump_duration*v/cell_length = v/75 = 0.0133*v with ~zero intercept
# The 10 Hz policy captures the wave-8 fit was built on quantise takeoff and touchdown to 0.1 s, and
# that error grows with velocity (0.1*v/60 cells) -- which inflates the SLOPE exactly as observed.
# Why it mattered: landing_hazard rounds to an integer cell offset, so a 0.4-cell over-estimate at
# v200 flagged the WRONG cell ~40% of the time -- v90's headline feature, aimed one cell too far.
LAND_SPAN_A, LAND_SPAN_B = 0.0143, -0.11
# Scale that span by the ground-height change the arc crosses (see the landing_hazard block below).
# DEVICE-MEASURED, no fitting. Set LAND_SPAN_LEVEL_SCALED = False to reproduce the inputs a model
# trained before 2026-07-27 night saw (v78..v105 all trained on the FLAT span), the same way
# --land-span pins the pre-correction A/B pair.
LAND_SPAN_LEVEL_SCALED = True
LAND_SPAN_STEPDOWN, LAND_SPAN_STEPUP = 1.17, 0.74
DCPROG_PER_VEL = 0.1 / 60.0   # per-frame cprog advance per unit velocity (= step_dt / cell_length)

# PREDICTED-JUMP geometry (2026-07-04) — the CREVASSE analog of predicted_landing (device-measured
# human_calib_v2: step-UP crevasses jumped 100%, and the takeoff LEAD scales with velocity — the human
# takes off ~on the gap @v150 but ~1.5 cells early @v210). The sim's step-up death model
# (sim_env `_check_death`) is a parabola: airborne span = jump_duration*velocity/cell_length cells, and
# the runner must reach the raised LEDGE (2 cells ahead of the current cell when next_ground_change_higher
# fires: cell1=gap, cell2=ledge) NEAR the apex (frac 0.5) or it clips the wall (DeathType 4 Ledge). So the
# ideal takeoff is 0.5*span cells before the ledge. `predicted_jump` projects "how far am I from that ideal
# takeoff point", velocity-invariant, so the policy learns ONE threshold ("jump when predicted_jump ~=
# JUMP_NOW") that self-adjusts across speeds — the same trick predicted_landing used to break the turn wall
# at all speeds. Derived host-side from the already-streamed cell_progress+velocity+next_ground_change_higher
# (NO hook change; deployable). Constants match the sim's jump_duration/cell_length so sim-train == device.
JUMP_DURATION = 0.8    # s — device jump airtime (= sim cfg jump_duration); span = JUMP_DURATION*v/CELL_LENGTH
CELL_LENGTH = 60.0     # device units per path cell (= sim cfg cell_length; consistent w/ DCPROG_PER_VEL)
LEDGE_CELLS_AHEAD = 2.0  # when next_ground_change_higher fires: cell1 = gap, cell2 = raised ledge (apex here)
PJ_NONE = 1.0          # sentinel when no step-up crevasse is imminent (far from any ledge -> don't jump)
# ⚠️ 2026-07-27 — THE COMMENT ABOVE CLAIMS "velocity-invariant". IT IS NOT, and that is the mid->high
# crevasse defect the user has reported for weeks ("the policy jumps way too early").
#   raw = (LEDGE_CELLS_AHEAD - cp) - 0.5*span  is measured in CELLS, and since raw = span*(frac - 0.5),
#   a FIXED learned threshold r means frac = 0.5 + r/span -> as speed (span) grows the SAME obs value
#   sits CLOSER TO 0.5, i.e. the safety margin SHRINKS exactly when the arc is longest. Solving for the
#   chosen takeoff: cp = LEDGE_CELLS_AHEAD - 0.5*span - r, which FALLS ~0.4 from v150 to v210.
#   MEASURED ON DEVICE (scripts/takeoff_timing.py): the policy's takeoff cp falls 0.81 -> 0.49 over that
#   range (slope -0.0054) while the HUMAN RISES 0.53 -> 0.85 (slope +0.0054). The observation itself was
#   teaching the inversion, so no amount of reward shaping or death modelling could fix it.
# The flag emits the DIMENSIONLESS arc fraction instead, so one threshold = one physical margin at all
# speeds. Set False to reproduce a pre-2026-07-27 model's inputs (v78..v93 all trained on the old law).
# ⚠️ SET BACK TO False on 2026-07-27: the rescaling is defensible in principle, but v95 (which trained
# on it) was STILL inverted (-0.0042 +- 0.0013), so it never demonstrated a benefit — while the arms
# carrying it drifted into other trouble. The mid->high fix that DID work is `jump_exec_lat_jitter` 0
# (device-measured), not this. Keep the analysis in the note above; re-test the flag on its own later.
# ⚠️ SET BACK TO True 2026-07-30, on a DEVICE measurement of the current champion. A fresh 8-episode
# v111 batch (captures/w18_v111_base.jsonl, 33->38C) died 7-of-8 on DeathType 4 = LEDGE, i.e. the
# crevasse wall clip, and its takeoff slope is the WORST ever recorded here: -0.01051 +- 0.00262
# (n=30 crossings) against the HUMAN's +0.00828. The 2026-07-27 note that retired this flag argued
# the policy "is already well-timed IN SIM and only mistimes on device" -- that is NO LONGER TRUE and
# is now falsified: the SAME policy measured in the C++ trainer is inverted too, -0.00574 +- 0.00088
# over 150 crossings (6.5 s.e.), so the defect IS reproducible and therefore trainable.
# The reason it never self-corrected: the sim's level-3 graze was 0.7% vs the device's 3.3%, so the
# inverted takeoff was nearly FREE in-sim (`crevasse_takeoff_std` sat at 0.0 while this file's own
# calibration note says 0.2). v121 pairs this flag with std 0.1 (= 5.1% graze, device 3.3%, harsh
# side) -- the amplifier pattern: the law makes the correct behaviour LEARNABLE, the graze makes the
# wrong one COST something. Either alone has already failed (v95 = law only, still -0.0042).
# ⚠️ Any model trained BEFORE this date learned the old law -- pin this False to reproduce its inputs.
PJ_VELOCITY_INVARIANT = False


# ★ 2026-07-30 — THE LEAD TERM WAS THE BUG, and it is provable rather than another guess.
# predicted_jump aimed for APEX OVER THE CAP: raw = (LEDGE_CELLS_AHEAD - cp) - PJ_LEAD_FRAC*span with
# PJ_LEAD_FRAC = 0.5. A fixed learned threshold r then implies a takeoff at
#     cells-to-cap = (2r - 0.5) + PJ_LEAD_FRAC*span,
# which GROWS with speed: 1.20 -> 1.60 cells from v150 to v210, slope +0.0067. v111 measures +0.0125.
# But the HUMAN does the opposite -- device-measured cells-to-cap 0.70 / 0.35 / 0.22 across
# v140-180 / 180-200 / 200-240, slope -0.0029, i.e. they take off ~0.2 cells before the cap AT ANY
# SPEED and cross it while still ASCENDING. Apex-over-cap crosses it DESCENDING = the wall clip, and
# 7 of 8 deaths in the 2026-07-30 v111 batch were DeathType 4 (Ledge).
# ⚠️ AND THE "velocity-invariant" FLAG CANNOT FIX THIS: its form (2-cp)/span/2 gives
# cells-to-cap = 2r*span, ALSO growing with span -- identical implied slope -0.00667 for r=0.25.
# That is why v95 AND v121 both retrained and both stayed inverted (-0.0042, -0.00516). Setting
# PJ_LEAD_FRAC = 0.0 would make the implied takeoff distance CONSTANT (0.20 cells at every
# speed), which is the behaviour the human actually exhibits -- but see the REVERT note below:
# tried as v122 and it measured WORSE for reasons that had nothing to do with the geometry.
# ⚠️ REVERTED TO 0.5 (the champion's law) after v122 MEASURED WORSE: slope -0.00748 vs v111's
# -0.00537. Two reasons, both now understood:
#  (1) HARNESS: with lead=0, raw = (2-cp) in [1,2] and the mapping (raw+0.5)/2 SATURATES at 1.0 for
#      every cp <= 0.5 -- which is also PJ_NONE, the "no crevasse here" sentinel. Half the cell
#      carried no signal and looked like "nothing ahead".
#  (2) REDUNDANCY: with no span term the cue is a function of cp ALONE, so it duplicates the
#      cell_progress column the policy already has and adds nothing.
# ⇒ The lead term is what makes predicted_jump informative (it fuses cp WITH velocity). The problem
# is not that it fuses them, it is that the sim never charges for getting the fusion wrong: a bad
# takeoff costs a STUMBLE in-sim (5.1% graze) where the device KILLS (7 of 8 deaths = Ledge).
PJ_LEAD_FRAC = 0.5


def predicted_jump_obs(cell_progress, velocity, crevasse_ahead) -> float:
    """Velocity-invariant step-up jump-timing cue. When a zone crevasse is the next cell
    (``crevasse_ahead``), returns how far the runner is from the IDEAL takeoff point (small value ~=
    "take off NOW"); otherwise ``PJ_NONE`` (far). Monotonic in the distance-to-ideal-takeoff, computed
    identically in sim + on device so the learned threshold transfers.

    dist_to_ledge = LEDGE_CELLS_AHEAD - cell_progress (cells to the raised ledge);
    lead_needed    = 0.5 * span                       (take off half a span before the ledge = apex there);
    raw            = dist_to_ledge - lead_needed       (0 = take off now, >0 too early, <0 too late).
    """
    if not crevasse_ahead:
        return PJ_NONE
    v = float(velocity or 0.0)
    cp = float(cell_progress if cell_progress is not None else 0.0)
    span = JUMP_DURATION * v / CELL_LENGTH
    if PJ_VELOCITY_INVARIANT:
        # ARC FRACTION traversed on reaching the ledge if the runner takes off NOW. Dimensionless, so
        # ONE learned threshold means the same physical margin at every speed. 0.5 = apex exactly over
        # the ledge; BELOW 0.5 = still ascending there (safe); ABOVE 0.5 = already descending = the
        # wall clip. /2 keeps the old anchor (apex -> 0.25) and the PJ_NONE=1.0 sentinel.
        return float(np.clip((LEDGE_CELLS_AHEAD - cp) / max(span, 1e-6) / 2.0, 0.0, 1.0))
    raw = (LEDGE_CELLS_AHEAD - cp) - PJ_LEAD_FRAC * span
    # map the useful range [-0.5, 1.5] -> [0,1] (raw=0 "jump now" -> 0.25); clip keeps it bounded.
    return float(np.clip((raw + 0.5) / 2.0, 0.0, 1.0))


# PREDICTED-TURN geometry (WAVE 11 2026-07-09) — the TURN analog of predicted_jump, aimed at the
# dead-letter (v70's #1 ender: 2x pt-1 + 1 turn in its record batch; wave-10 proved pressure alone
# can't teach the pre-swipe). The device consume deadline is ~cp 0.56 on the junction and the swipe
# ARRIVES ~1 control frame after the decision, so the true decision deadline is 0.56 - v*dt/cell —
# at v210 barely 0.2 of the junction cell. The latency-immune play = swipe on the APPROACH cell (the
# game's HandleDelayedTurning buffers it; sim turn_approach_buffer models it). predicted_turn gives
# the policy the MARGIN-TO-DEADLINE as one velocity-invariant scalar: ~1.0 on the approach cell
# (swipe now = always consumed), decaying toward 0 across the junction cell as the deadline closes,
# 0 when no junction context (or dead-lettered). One learned threshold transfers across speeds.
# Host-derived from streamed cell_progress/velocity/is_turn/next_is_turn — NO hook change (17f ok).
TURN_DEADLINE_CP = 0.56   # device CanTurn window end (velocity-independent, measured 2026-07-05)
TURN_SEND_FRAMES = 1.0    # decision -> arrival ~1 control frame (device-measured turn pipeline)


def predicted_turn_obs(cell_progress, velocity, on_junction, next_is_turn) -> float:
    """Velocity-invariant swipe-now cue. Approach cell -> 1.0 (buffered swipe, always safe);
    junction cell -> remaining decision margin (deadline - cp - send-latency) scaled to [0,1],
    hitting 0 at the dead-letter point; elsewhere 0 (no junction context)."""
    if on_junction:
        v = float(velocity or 0.0)
        cp = float(cell_progress if cell_progress is not None else 0.0)
        margin = TURN_DEADLINE_CP - cp - TURN_SEND_FRAMES * v * DCPROG_PER_VEL
        return float(np.clip(margin / TURN_DEADLINE_CP, 0.0, 1.0) * 0.75)  # junction: 0..0.75
    if next_is_turn:
        return 1.0                                                          # approach: swipe now
    return 0.0


def lane_to_obs(lane) -> float:
    """Continuous lane -> [0,1] (center 0 -> 0.5, left wall -7 -> ~0.06, right wall +7 -> ~0.94)."""
    if lane is None:
        lane = 0.0
    return float(np.clip((float(lane) - LANE_OBS_MIN) / (LANE_OBS_MAX - LANE_OBS_MIN), 0.0, 1.0))


def narrow_lane_to_obs(ntl) -> float:
    """Map the narrow's safe lane (-1 L / 0 C / +1 R) onto the SAME continuous (center-0) scale as
    `lane`, so the policy compares the two and strafes toward the target. None -> center (no
    constraint). Target = the MEASURED strip center +/-4.5 (Region2D bands), not the coin-lane step."""
    if ntl is None:
        return lane_to_obs(0.0)
    return lane_to_obs(int(ntl) * NARROW_CENTER)


def path_type_obs(pt) -> float:
    """COARSE path_type category (2026-06-26) — the raw enum/37 was a sim-to-real LEAK: the device emits
    straight types the sim never did (pt 1 AND 2 for straights, + 8/9/10), so the policy saw OOD
    path_type values on device and mis-reacted (strafed off the edge). Map to 5 stable buckets that the
    sim and device AGREE on (action cues already live in is_turn/must_jump/must_slide/next_wide_gap/
    narrow_target_lane, so the fine enum is redundant): straight / turn / hole-piece / narrow / cap."""
    pt = int(pt) if pt is not None else -1
    if pt < 0:
        return 0.0
    if pt <= 2:
        return 0.16            # straight (StraightVert/Horz etc.)
    if pt <= 7:
        return 0.33            # turn / junction
    if pt <= 16:
        return 0.50            # hole / tree / holelong piece
    if pt <= 30:
        return 0.66            # narrow strip
    return 0.83                # level cap / crevasse (31-37)


def _norm_enum(value: int, max_value: float) -> float:
    """Normalize an int enum to [0,1]; the -1 'none' sentinel maps to 0.0.

    Note: this aliases sentinel -1 with real enum value 0 at junction edges.
    Acceptable for v0 (documented in the plan); add a companion 'valid' flag if needed.
    """
    if value is None or value < 0:
        return 0.0
    return float(np.clip(value, 0.0, max_value) / max_value)


def full_obs_len(frame_stack: int = 1, action_history_len: int = 0) -> int:
    """Total obs length = ``frame_stack`` stacked OBS_LEN-d frames + ``action_history_len`` one-hot
    action frames (7 each). frame_stack=1, K=0 -> the base OBS_LEN-d obs."""
    return max(1, int(frame_stack)) * OBS_LEN + int(action_history_len) * C.N_ACTIONS


def make_observation_space(frame_stack: int = 1, action_history_len: int = 0) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0,
                      shape=(full_obs_len(frame_stack, action_history_len),), dtype=np.float32)


def state_to_obs(state: dict) -> np.ndarray:
    """Convert a raw state dict to the base (OBS_LEN=28) normalized float32 observation vector.

    Tolerant of missing keys (falls back to ``constants.STATE_DEFAULTS``) so a
    malformed/truncated JSON line never crashes a step. Action history (if any) is appended
    separately by ``obs_with_history`` — this stays OBS_LEN-d for the fidelity harness/replay.
    """
    g = lambda k: state.get(k, C.STATE_DEFAULTS.get(k))  # noqa: E731
    out = np.empty(OBS_LEN, dtype=np.float32)
    for i, f in enumerate(OBS_BOOL_FIELDS):
        out[i] = 1.0 if g(f) else 0.0
    base = len(OBS_BOOL_FIELDS)
    v = float(g("velocity") or 0.0)
    out[base + 0] = float(np.clip((v - VEL_MIN) / (VEL_MAX - VEL_MIN), 0.0, 1.0))
    out[base + 1] = path_type_obs(g("path_type"))   # COARSE category (robust to device enum variety)
    out[base + 2] = _norm_enum(int(g("path_level")), PATH_LEVEL_MAX)
    out[base + 3] = _norm_enum(int(g("facing")), FACING_MAX)
    cp = g("cell_progress")
    out[base + 4] = float(np.clip(cp if cp is not None else 0.0, 0.0, 1.0))
    out[base + 5] = lane_to_obs(g("lane"))
    # narrow_target_lane: which lane to be in for the upcoming narrow (-1 L / 0 C / +1 R), mapped onto
    # the SAME continuous scale as `lane` so the policy compares them and strafes to match (None -> home).
    out[base + 6] = narrow_lane_to_obs(g("narrow_target_lane"))
    # predicted_landing: where a turn swiped THIS frame would LAND (in cell_progress), accounting for the
    # ~1-frame exec latency. The policy learns to swipe when this ~= 0.44 (centered) -> speed-invariant
    # turn timing. Derived host-side from the already-streamed cell_progress + velocity (no hook change).
    cp0 = cp if cp is not None else 0.0
    out[base + 7] = float(np.clip(cp0 + DCPROG_PER_VEL * v, 0.0, 1.0))
    # predicted_jump: velocity-invariant step-up crevasse takeoff cue (see predicted_jump_obs). Gated on
    # next_ground_change_higher (the zone-crevasse-ahead flag, set by the sim snapshot / device hook).
    out[base + 8] = predicted_jump_obs(cp0, v, g("next_ground_change_higher"))
    # next_path_level / next2_path_level: upcoming altitude 2-cell lookahead (device-verified readable).
    # direction of a crevasse = sign(next2_path_level - path_level). Same enum normalization as path_level.
    out[base + 9] = _norm_enum(int(g("next_path_level")), PATH_LEVEL_MAX)
    out[base + 10] = _norm_enum(int(g("next2_path_level")), PATH_LEVEL_MAX)
    # next2_narrow_lane: the 2-ahead narrow's side on the SAME lane scale (the jump->narrow mid-air
    # tilt fix — see OBS_EXTRA_FIELDS). None (no narrow / junction dropout) -> center, like narrow_target_lane.
    out[base + 11] = narrow_lane_to_obs(g("next2_narrow_lane"))
    # landing_hazard (WAVE 16): if a jump were taken NOW, which cell would the runner land in, and is
    # a must-jump hazard there? span in cells = SPAN_A*v + SPAN_B (device jump-arc fit, jump_arc.py);
    # landing offset = cell_progress + span. 1.0 = the landing cell carries a must-jump hazard, 0.0 =
    # clear, 0.5 = beyond the 3-cell lookahead (genuinely unknown -- do not fake certainty).
    _span = max(0.0, LAND_SPAN_A * v + LAND_SPAN_B)
    # ★ THE ARC IS LONGER OVER A DROP AND SHORTER OVER A RISE [DEVICE-MEASURED 2026-07-27,
    # scratchpad/stepdown_arc.py; two independent pipelines]:
    #                      flat            step-DOWN         step-UP
    #     HUMAN 30 Hz  0.767 s / 2 cells  0.900 (1.17x)/3   0.567 (0.74x)/2
    #     v92b  10 Hz  0.700 s / 2 cells  0.800 (1.14x)/3   0.500 (0.71x)/2
    # landing_hazard existed to answer "does my arc come down on a hazard", and it was computing the
    # landing cell from a FLAT span -- so it was wrong by a whole cell exactly at crevasses, which is
    # where chains kill. On DEVICE this is the deadliest pattern measured: BLIND_LANDING_JUMP fires
    # only 0.29/km but 57% of them (4 of 7) precede a death within 3 s.
    # Level -1 = the hook's junction dropout (nothing known past the corner) -> leave the span flat
    # rather than guess.
    if LAND_SPAN_LEVEL_SCALED:
        _l0 = int(g("path_level") or 2)
        _la = int(g("next2_path_level") or _l0)
        if _la <= 0:                            # junction dropout -> fall back to the 1-ahead cell
            _la = int(g("next_path_level") or _l0)
        if _la > 0 and _l0 > 0:
            if _la < _l0:
                _span *= LAND_SPAN_STEPDOWN
            elif _la > _l0:
                _span *= LAND_SPAN_STEPUP
    _off = int(cp0 + _span)                     # cells ahead where the arc comes down
    if _off <= 0:
        _lh = 1.0 if g("must_jump") else 0.0
    elif _off == 1:
        _lh = 1.0 if g("next_must_jump") else 0.0
    elif _off == 2:
        _lh = 1.0 if g("next2_must_jump") else 0.0
    elif _off == 3:
        _lh = 1.0 if g("next3_must_jump") else 0.0
    else:
        _lh = 0.5                               # past the lookahead horizon
    out[base + 13] = _lh
    # slide_landing_hazard (WAVE 18): the same landing cell, asked of the must-SLIDE ladder. SAME
    # span and SAME rounding as landing_hazard on purpose -- it is the same arc, so the two columns
    # differ only in which hazard they look for, and the policy can learn "arc ends on a slide" as a
    # distinct fact from "arc ends on a hole". 1.0 = the arc comes down ON a must-slide cell (the
    # measured death shape: arc slack +0.21, no action available on landing), 0.0 = clear,
    # 0.5 = past the +3 lookahead (unknown; do not fake certainty).
    if _off <= 0:
        _slh = 1.0 if g("must_slide") else 0.0
    elif _off == 1:
        _slh = 1.0 if g("next_must_slide") else 0.0
    elif _off == 2:
        _slh = 1.0 if g("next2_must_slide") else 0.0
    elif _off == 3:
        _slh = 1.0 if g("next3_must_slide") else 0.0
    else:
        _slh = 0.5
    out[base + 14] = _slh
    # landing_margin: signed cells between the arc's landing point and the NEAREST must-jump hazard.
    _land = cp0 + _span
    _lad = (g("must_jump"), g("next_must_jump"), g("next2_must_jump"), g("next3_must_jump"))
    _hz = [i for i, f in enumerate(_lad) if f]
    if not _hz:
        _lm = 1.0                                   # nothing to hit -> maximum room
    else:
        _near = min(_hz, key=lambda i: abs(_land - i))
        _lm = float(np.clip((_land - _near + 1.0) / 4.0, 0.0, 1.0))   # -1..+3 cells -> 0..1
    out[base + 15] = _lm
    # HORIZON SCALARS (WAVE 17): normalised distance to the next must-JUMP / must-SLIDE cell over a
    # HORIZON_CELLS window. 0.0 = it is HERE, 1.0 = nothing within the horizon. The booleans stop at
    # +3, which at v210 is 0.86 s -- barely one jump (0.8 s) -- so a three-obstacle chain's third
    # element is invisible when the first jump must be committed (measured: 100% of chains, device
    # AND human). These carry the RHYTHM instead: how much room is left after the thing in front.
    if HORIZON_OBS:
        out[base + 16] = _horizon_obs(g, "must_jump")
        out[base + 17] = _horizon_obs(g, "must_slide")
    # predicted_turn: approach-cell swipe cue / junction deadline margin (host-derived, WAVE 11).
    out[base + 12] = predicted_turn_obs(cp, v, g("is_turn"), g("next_is_turn"))
    return out


def _horizon_obs(g, kind: str) -> float:
    """Normalised cells to the next `kind` hazard, over HORIZON_CELLS.

    Reads the streamed per-offset booleans (must_*, next_*, next2_*, next3_*) and, when the host
    supplies them, the deeper `dist_next_must_*` scalar from the hook (offsets 4..6 are NOT derivable
    from the +3 stream -- that is why this needs hook 17k). Falls back to the boolean ladder so the
    obs is well-defined on captures/sims that predate the hook change.
    """
    direct = g(f"dist_next_{kind}")
    if direct is not None:
        d = float(direct)
        return min(max(d, 0.0), HORIZON_CELLS) / HORIZON_CELLS
    ladder = (kind, f"next_{kind}", f"next2_{kind}", f"next3_{kind}")
    for i, f in enumerate(ladder):
        if g(f):
            return i / HORIZON_CELLS
    return 1.0


def action_history_vec(history, action_history_len: int) -> np.ndarray:
    """One-hot encode the last ``action_history_len`` actions, oldest-first, left-padded with NoOp.

    ``history`` is the running list of past actions (most recent LAST). Gives the memoryless
    MLP policy a short memory so it knows it *just* turned/jumped — curbs the "multiple turns at a
    junction" spam (the policy re-fires a turn because a single-frame obs can't see it already did).
    """
    K = int(action_history_len)
    vec = np.zeros(K * C.N_ACTIONS, dtype=np.float32)
    if K <= 0:
        return vec
    h = [int(a) for a in list(history)[-K:]]
    h = [int(C.Action.NOOP)] * (K - len(h)) + h   # left-pad so index 0 = oldest
    for i, a in enumerate(h):
        if 0 <= a < C.N_ACTIONS:
            vec[i * C.N_ACTIONS + a] = 1.0
    return vec


def obs_with_history(state: dict, history, action_history_len: int = 0) -> np.ndarray:
    """Base obs (OBS_LEN) optionally concatenated with the one-hot action history (K*7)."""
    base = state_to_obs(state)
    if not action_history_len:
        return base
    return np.concatenate([base, action_history_vec(history, action_history_len)])


def stacked_obs(frame_deque, history, frame_stack: int = 1, action_history_len: int = 0) -> np.ndarray:
    """Full obs = the last ``frame_stack`` base frames (oldest->newest) + one-hot action history.

    ``frame_deque`` holds recent base (OBS_LEN-d) obs vectors (most recent LAST); it is left-padded by
    repeating the OLDEST available frame so the vector is fixed-size from the first step. Stacking
    frames gives the MLP the temporal context (how is_turn/facing/velocity are *evolving*) it needs
    to ANTICIPATE a junction and fire the turn on the exact frame — beating the 1-frame reactive
    limit that caps single-frame turn reliability.
    """
    N = max(1, int(frame_stack))
    frames = list(frame_deque)[-N:]
    if not frames:
        frames = [np.zeros(OBS_LEN, dtype=np.float32)]
    frames = [frames[0]] * (N - len(frames)) + frames    # left-pad with the oldest frame
    parts = frames
    if action_history_len:
        parts = parts + [action_history_vec(history, action_history_len)]
    return np.concatenate(parts).astype(np.float32)
