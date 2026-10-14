"""Shared reward / shaping logic so the live env and the headless sim agree byte-for-byte.

The canonical implementation lives in ``TempleRunEnv._shaping_reward`` (``env.py``). To keep
the sim's reward *identical* without re-touching the live env (and risking its behavior), the
exact same arithmetic is factored here as a pure function. ``sim_env.py`` calls it; ``env.py`` is
left untouched but the numbers it computes are reproduced verbatim by ``shaping_reward`` below
(same branches, same order, same constants). If you change the shaping in ``env.py``, mirror it
here (and vice-versa) — ``tests``/fidelity should keep them in lockstep.

Reward (both envs):
    reward = (score - prev_score) * reward_scale
    reward += shaping_reward(...)            # action_cost + reaction_bonus + turn_spam + idle
    if terminated: reward -= death_penalty
"""
from __future__ import annotations

from typing import Optional

from . import constants as C
from .obs import DCPROG_PER_VEL


def shaping_reward(
    action: int,
    prev: Optional[dict],
    last_action: Optional[int],
    *,
    action_cost: float = 0.0,
    reaction_bonus: float = 0.0,
    turn_spam_penalty: float = 0.0,
    idle_action_penalty: float = 0.0,
    redundant_action_penalty: float = 0.0,
    airborne_action_penalty: float = 0.0,
    jump_hold_penalty: float = 0.0,
    turn_buffer_bonus: float = 0.0,
    stepdown_bonus_gate: bool = False,
    landing_hazard_penalty: float = 0.0,
    turn_takeoff_penalty: float = 0.0,
    jumped_recently: bool = False,
    slid_recently: bool = False,
    strafe_penalty: float = 0.0,
    narrow_hold_bonus: float = 0.0,
    aim_center_weight: float = 0.0,
    velocity_drop_penalty: float = 0.0,
    off_center_penalty: float = 0.0,
    slide_jump_penalty: float = 0.0,
    prev_velocity: float = 0.0,
    velocity: float = 0.0,
    lane: float = 0.0,
) -> float:
    """Dense shaping term (0 unless a knob is set). Mirror of ``env._shaping_reward``.

    ``prev`` is the state the policy actually *saw* when it chose ``action`` (the obs from the
    previous step/reset), NOT the post-action state. ``last_action`` is the action taken on the
    previous step (for the turn-spam memory the policy lacks).
    """
    if (not action_cost and not reaction_bonus and not turn_spam_penalty
            and not idle_action_penalty and not redundant_action_penalty and not strafe_penalty
            and not narrow_hold_bonus and not velocity_drop_penalty and not off_center_penalty
            and not slide_jump_penalty and not airborne_action_penalty and not jump_hold_penalty
            and not turn_buffer_bonus and not landing_hazard_penalty
            and not turn_takeoff_penalty):
        return 0.0
    r = 0.0
    # JUMP into a slide-obstacle is FATAL on device [MEASURED 2026-07-03, ppo_speed_v8: 4/8 deaths] —
    # a TreeSlide/FlameTower is a LOW bar you pass UNDER; jumping lifts you into it. Dock a JUMP issued
    # when a slide-obstacle is HERE / AHEAD or a slide is already in progress, so the policy stops
    # popping up mid-slide (the "slid then jumped and died" bug). Mirrors the sim death model.
    # ★ LANDING-HAZARD PENALTY [2026-07-28]: dock a JUMP whose ARC IS PREDICTED TO COME DOWN ON A
    # MUST-JUMP HAZARD. This is the amplifier for a cue that is already correct but too quiet.
    # MEASURED: v107 dies 5 of 7 deaths with dt=4 Ledge ("my arc landed me somewhere fatal"), and the
    # fatal takeoff had landing_hazard = 1.0 ALREADY SET one frame early AND at takeoff -- so this is
    # NOT a perception gap. The counterfactual response is the problem: flipping landing_hazard on
    # real states moves P(JUMP) by only -0.0039 on a 0.036 base (~10% relative), nowhere near enough
    # to veto a jump the rest of the obs is demanding. (v92b -0.0078, v104 -0.0005: the cue also
    # ERODED through the lineage.) Same shape as the held-jump fix, which is the project's strongest
    # precedent: the death model made the world truthful, and airborne_action_penalty made it LOUD
    # enough to learn -- 10.67 -> 0.00 per km on device.
    # Recomputed here from the SAME streamed fields obs.py uses, so it is identical on device and in
    # sim, and un-gameable (it prices the arc, not the button).
    if landing_hazard_penalty and action == int(C.Action.JUMP) and prev is not None:
        from .obs import LAND_SPAN_A, LAND_SPAN_B
        _v = float(prev_velocity or prev.get("velocity") or 0.0)
        _sp = max(0.0, LAND_SPAN_A * _v + LAND_SPAN_B)
        _off = int(float(prev.get("cell_progress") or 0.0) + _sp)
        _lad = (prev.get("must_jump"), prev.get("next_must_jump"),
                prev.get("next2_must_jump"), prev.get("next3_must_jump"))
        if 0 <= _off <= 3 and _lad[_off]:
            r -= landing_hazard_penalty
    # ★ TURN-TAKEOFF PENALTY [2026-07-28]: dock a JUMP whose ARC IS STILL IN THE AIR WHEN AN
    # UNRESOLVED JUNCTION ARRIVES. USER: "it jumped in a turn, turned and landed in a hole."
    # MEASURED: 5.0% of takeoffs happen with a junction within 2 cells, and 8.7% of THOSE kill,
    # against ~1% for takeoffs generally -- an 8x lethality multiplier on one identifiable setup.
    # WHY IT IS STRUCTURAL, not a training gap: landing_hazard and the whole lookahead are computed
    # ALONG THE CURRENT HEADING, with an explicit junction dropout. When the turn fires mid-air the
    # runner comes down on a path the observation NEVER MODELLED, so no amount of exposure can teach
    # the landing -- the cue for it does not exist. Pricing the TAKEOFF is the tractable half: the
    # policy cannot know where it will land, but it can know not to leave the ground here, and
    # is_turn / next_is_turn are both already in the obs, so the counterfactual IS learnable.
    # Third use of the amplifier pattern (held-jump, landing-hazard, this). Prices the ARC and the
    # geometry rather than the button, so it cannot be gamed by a one-frame gap.
    if turn_takeoff_penalty and action == int(C.Action.JUMP) and prev is not None:
        from .obs import LAND_SPAN_A, LAND_SPAN_B
        _v = float(prev_velocity or prev.get("velocity") or 0.0)
        _sp = max(0.0, LAND_SPAN_A * _v + LAND_SPAN_B)
        _reach = float(prev.get("cell_progress") or 0.0) + _sp
        if prev.get("is_turn") or (_reach >= 1.0 and prev.get("next_is_turn")):
            r -= turn_takeoff_penalty
    if slide_jump_penalty and action == int(C.Action.JUMP) and prev is not None:
        if prev.get("must_slide") or prev.get("next_must_slide") or prev.get("is_sliding"):
            r -= slide_jump_penalty
    # VELOCITY-DROP penalty: dock a real velocity LOSS (a stumble / clipping a step-up wall slows the
    # runner). Discourages the too-early crevasse jump that clips the raised wall + gets slowed (user-
    # reported). Ignores the normal accel; only fires on a drop beyond a small deadband.
    if velocity_drop_penalty and prev_velocity - velocity > 3.0:
        r -= velocity_drop_penalty * (prev_velocity - velocity)
    # on/approaching a narrow the runner MUST hold a strafe to stay in the safe band (NoOp drifts to
    # center = off the strip = Fall). So holding the lane there is REQUIRED, not spam.
    strafes = (int(C.Action.STRAFE_LEFT), int(C.Action.STRAFE_RIGHT))
    on_narrow = prev is not None and (
        (isinstance(prev.get("path_type"), int) and 17 <= prev.get("path_type") <= 30)
        or prev.get("narrow_target_lane") is not None)
    if action_cost and action != int(C.Action.NOOP):
        # a hold-strafe on a narrow is FREE — else action_cost pays the policy to NoOp (drift off) -> the
        # constant-dither device failure. Only exempt the strafe (jump/slide still cost).
        if not (action in strafes and on_narrow):
            r -= action_cost
    # STRIP-HOLD bonus, GRADED by closeness to the target strip lane (device-measured: strips sit at
    # |lane| ~4.5, never center). The OLD binary "on a narrow CELL -> +bonus regardless of lane" gave NO
    # gradient toward the strip -> the policy learned to sit at center (NoOp) and just eat the rare narrow
    # death (device valid_v1: on a narrow the policy NEVER strafed, lane stayed 0 -> off the side strip ->
    # Fall). Scale the bonus by how close the runner is to the strip: max ON the strip, ~0 at center. This
    # is a GRADIENT that pulls the policy off-center ONTO the strip (the inverse of off_center_penalty,
    # which is for straights). Active from when narrow_target_lane is set (approach), so it learns to reach
    # the strip BEFORE the narrow. On the final narrow cell (target_lane goes None) fall back to a flat hold.
    if narrow_hold_bonus and on_narrow:
        ntl = prev.get("narrow_target_lane") if prev is not None else None
        if ntl is not None:
            # Reward reaching the correct SIDE, saturating once on the strip. The survivable band is
            # [+2,+7] (center 4.5 +- tol 2.5), and the EASIEST hold is full tilt at the +-7 edge (a stable
            # commanded point — no dither). An earlier tent-shaped reward PEAKED at 4.5 and under-rewarded
            # the +-7 hold, so the policy (which can't cheaply sit at exactly 4.5) got ~0 reward and never
            # learned to strafe at all (device+sim: |lane|-on-narrow ~0, drifts to center, Fall). Reward
            # `lane*sign` (progress onto the correct side), FULL once |lane|>=3 — covers both the oracle's
            # ~2.84 dither AND the +-7 full-tilt hold. A real GRADIENT off center toward the strip.
            sign = 1.0 if ntl > 0 else -1.0
            closeness = max(0.0, min(1.0, (lane * sign) / 3.0))   # 0 at center/wrong side, 1 for |lane|>=3 correct side
            r += narrow_hold_bonus * closeness
        else:
            r += narrow_hold_bonus                          # on-strip already; flat hold
    # OFF-CENTER (RECENTER incentive): being off-center on a STRAIGHT is nearly free otherwise (action_cost
    # only pays the strafe ACTION; once drifted, NoOp is free) -> the policy strafes randomly on straights,
    # arrives at the next turn off-center, and turn_lane_carry (~3x) amplifies it into a >kill landing ->
    # the dominant device death once turns are handled (lane driven to +-9..-23, 2026-07-02 valid_v1_16f).
    # Dock |lane| beyond a small deadband when NOT on/approaching a narrow (narrows REQUIRE holding the
    # side strip) so the policy actively holds center on straights -> hits turns centered -> lands safe.
    if off_center_penalty and not on_narrow:
        off = abs(lane) - 1.5
        if off > 0.0:
            r -= off_center_penalty * off
    if reaction_bonus and prev is not None:
        # Reward the reaction on the obstacle whether it's the CURRENT cell (must_jump) or AHEAD
        # (next_must_jump) — the anticipatory jump (next_*) is the one that actually clears the entry
        # check, so it MUST be rewarded (the old shaping only rewarded must_*, which punished the
        # correct early jump and rewarded the trailing re-jump -> the double-jump artifact).
        if (prev.get("must_jump") or prev.get("next_must_jump")) and action == int(C.Action.JUMP):
            # SOFT step-down [2026-07-27]: with stepdown_grounded_ok the sim stops KILLING a grounded
            # mid->low crossing, but reaction_bonus still PAYS 0.1 for jumping it, so v103 kept jumping
            # 98.5% of them (v102: 98.6% -- i.e. the world fix alone moved nothing). Removing the
            # death removed the requirement; this removes the incentive. The bonus exists to reward a
            # NECESSARY reaction, and on a step-down to low ground there is nothing to clear -- the
            # human jumps only 29.4% of them. A jump is still legal and still free; it is just no
            # longer paid for. Only fires when the must_jump comes SOLELY from soft cells.
            hard = ((prev.get("must_jump") and not prev.get("soft_stepdown"))
                    or (prev.get("next_must_jump") and not prev.get("next_soft_stepdown")))
            if hard or not stepdown_bonus_gate:
                r += reaction_bonus
        elif (prev.get("must_slide") or prev.get("next_must_slide")) and action == int(C.Action.SLIDE):
            r += reaction_bonus
        elif prev.get("is_turn"):
            # AIM-CENTER on the LANDING, not the swipe (2026-07-03, v10): the turn executes ~1 frame later,
            # landing at cell_progress + Δ(velocity) (Δ = DCPROG_PER_VEL*vel). Reward a junction-cell swipe
            # whose predicted LANDING is centered (~0.44) — so the policy swipes EARLIER at speed (it has
            # the predicted_landing obs to do so). The old version rewarded a centered SWIPE cprog, which
            # on device lands late (swipe 0.44 -> land 0.6-0.8 at speed = the dominant turn death).
            cp = float(prev.get("cell_progress") or 0.44) + DCPROG_PER_VEL * float(prev_velocity)
            center = 0.44
            aim = max(0.0, 1.0 - aim_center_weight * abs(cp - center) / 0.16)  # 1 at center, ~0 at +/-0.16
            side_ok = ((action == int(C.Action.TURN_LEFT) and prev.get("can_turn_left"))
                       or (action == int(C.Action.TURN_RIGHT) and prev.get("can_turn_right")))
            if side_ok:
                r += reaction_bonus * aim
    # TURN-BUFFER (PRE-SWIPE) bonus [WAVE 13, 2026-07-14 — the turn dead-letter credit-assignment fix]:
    # under turn_send_lat, a swipe ON the junction cell DEAD-LETTERS (arrives past the consume deadline ->
    # off-track Fall); the ONLY latency-robust play is to PRE-SWIPE on the APPROACH cell (next_is_turn),
    # which LATCHES into the game's HandleDelayedTurning and fires CENTERED, immune to latency
    # (turn_approach_buffer, from_buffer). The reaction_bonus turn branch above rewards the *reactive*
    # junction swipe (prev.is_turn) — the wrong, dead-lettering behavior — while the pre-swipe earned ZERO
    # shaping. Five RL arms plateaued because the dense reward pointed AT the reactive swipe and only the
    # sparse, delayed death pointed away. This term aligns the dense reward WITH survival: credit a
    # valid-direction turn issued on the approach cell (the correct exit side per next_can_turn_*). It is
    # UN-GAMEABLE (spam is eaten by turn_drop_debounce; a reactive junction swipe doesn't latch), the same
    # shape as the jump/slide reaction_bonus that works. Default 0 = off (equivalence/replay unchanged).
    if turn_buffer_bonus and prev is not None:
        if (action in (int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT))
                and not prev.get("is_turn") and prev.get("next_is_turn")):
            want_left = action == int(C.Action.TURN_LEFT)
            if (prev.get("next_can_turn_left") if want_left else prev.get("next_can_turn_right")):
                r += turn_buffer_bonus
    # Turn-spam: a turn immediately after another turn is the junction exploit -> dock it.
    if turn_spam_penalty and last_action is not None:
        turns = (int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT))
        if action in turns and last_action in turns:
            r -= turn_spam_penalty
    # Idle jump/slide: fired with NO obstacle ahead or here (not must_* and not next_must_*) ->
    # erratic "excitement", dock it. (Keyed on next_* too so the correct anticipatory jump is NOT
    # docked — only a truly purposeless jump is.)
    if idle_action_penalty and prev is not None:
        if (action == int(C.Action.JUMP)
                and not prev.get("must_jump") and not prev.get("next_must_jump")):
            r -= idle_action_penalty
        elif (action == int(C.Action.SLIDE)
                and not prev.get("must_slide") and not prev.get("next_must_slide")):
            r -= idle_action_penalty
        # IDLE TURN: a turn with NO junction context (not on/approaching a turn cell) is a spurious
        # turn -> on device it rotates off the path (FATAL, measured 2026-07-02). The only prior turn
        # penalty was turn_spam (turn-AFTER-turn); a LONE spurious turn was nearly free (action_cost).
        # Dock it so shaping curbs it before the policy has to learn from the Fall.
        elif (action in (int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT))
                and not prev.get("is_turn") and not prev.get("next_is_turn")):
            r -= idle_action_penalty
    # Redundant jump/slide: the obstacle is the CURRENT cell (must_jump) but NOT ahead
    # (next_must_jump=0) -> the player has already entered/cleared it; jumping again is the wasteful
    # "double-jump". Dock it. (A death-preventing jump fires on the lookahead next_must_jump, which is
    # NOT penalized; so this only suppresses the trailing re-jump, never a real reaction.)
    if redundant_action_penalty and prev is not None:
        if (action == int(C.Action.JUMP)
                and prev.get("must_jump") and not prev.get("next_must_jump")):
            r -= redundant_action_penalty
        elif (action == int(C.Action.SLIDE)
                and prev.get("must_slide") and not prev.get("next_must_slide")):
            r -= redundant_action_penalty
    # AIRBORNE-action (jump-spam) penalty [USER-observed + device forensic 2026-07-07, v57 6813m death]:
    # pressing JUMP while ALREADY airborne (is_jumping) does NOTHING useful in-sim (the takeoff is
    # ignored — see sim_env _apply_action's `_jump_t <= 0` guard), but on DEVICE it QUEUES a re-jump on
    # landing -> perpetual airborne -> the takeoff at the next hole is mistimed -> clip (dt=4 Ledge). The
    # idle penalty checks the OBSTACLE context, not the airborne STATE, so it can't catch this spam. Dock
    # a jump pressed while jumping (and a slide while sliding) so the policy stops the reflexive re-press.
    # ⚠️ 2026-07-27 REHABILITATED. This penalty was retired on 2026-07-08 on the conclusion that
    # "airborne presses are HARMLESS on device (dropped) -- so it solves a non-problem". That is WRONG,
    # and the wave-12 landing-lock forensic proves it: a press held through the arc is dropped WHILE
    # AIRBORNE but fires a fresh jump ON TOUCHDOWN (10% solid / 28-46% gap). It is the user's #1
    # reported death ("on high ground it jumps for no reason right after clearing a crevasse").
    # It is now paired with the held_jump_rejump_* death model, and the pairing is the point: the model
    # makes the world punish holding TRUTHFULLY, this penalty only AMPLIFIES a gradient that is now
    # genuinely there. That is why v58/v60/v61's shaping failed and this need not -- they taxed a
    # behaviour the world never punished, so the policy could game them (1-frame gaps) at zero cost.
    # ⚠️ KNOWN FALSE POSITIVE: on the LANDING frame prev.is_jumping is still True, so this also docks
    # the DELIBERATE landing-frame re-jump that post_land_gap_rejump_prob models as a REAL device
    # affordance in gap chains. Keep the coefficient modest for that reason (v60 over-corrected into
    # hole deaths at 0.12 on the related jump_hold_penalty).
    if airborne_action_penalty and prev is not None:
        if action == int(C.Action.JUMP) and prev.get("is_jumping"):
            r -= airborne_action_penalty
        elif action == int(C.Action.SLIDE) and prev.get("is_sliding"):
            r -= airborne_action_penalty
    # JUMP-HOLD penalty [USER-observed + device-decoded 2026-07-08, v59 ab_v59_r1 ep3/ep4 dt=4 Ledge]:
    # the policy HOLDS jump (presses JUMP on consecutive frames). A single press already fires the full
    # 0.8s jump, so the hold is pure spam — and it's LETHAL on HIGH-GROUND: the instant jump-1 lands, the
    # still-held press fires a seamless jump-2 (device is_jumping never drops between them) whose arc
    # descends into the next step-down crevasse / long-hole packed in the high-ground zone -> Ledge/Fall.
    # The airborne penalty MISSED this: in-sim jump-1's landing leaves a 1-frame is_jumping=False gap, so
    # the re-jump press has prev.is_jumping=False and wasn't docked -> the policy learned the landing-frame
    # re-jump is free (it is NOT on device, where the jumps overlap). Keying on last_action (the policy's
    # OWN action stream, identical sim<->device) catches the hold regardless of the airborne-state timing:
    # dock a JUMP that follows a JUMP (and a SLIDE after a SLIDE) -> the policy learns press-once-and-release.
    # v61 (2026-07-08): WINDOW + GATE. v60's "consecutive JUMP" check was GAMED — the policy inserted
    # 1-frame NoOp gaps (JMP,-,JMP) so no two presses were adjacent, yet still re-jumped on landing (ab_v60
    # ep4 dt=4). So key on `jumped_recently` = a JUMP within the last ~6 frames (a jump lasts 8 frames -> a
    # window can't be dodged by gaps), and GATE on prev.must_jump: NEVER dock a jump AT an obstacle (the
    # legit jump + the closed-loop retry for dropped jumps) -> stops the over-correction that made v60
    # jump-shy on holes. Only the purposeless post-obstacle re-jump (the lethal high-ground double-jump) docks.
    if jump_hold_penalty and prev is not None:
        if (action == int(C.Action.JUMP) and jumped_recently
                and not prev.get("must_jump")):
            r -= jump_hold_penalty
        elif (action == int(C.Action.SLIDE) and slid_recently
                and not prev.get("must_slide")):
            r -= jump_hold_penalty
    # STRAFE dock: curbs the strafe-spam (high ent_coef + action-history let it back). On the DEVICE an
    # off-track strafe drives the continuous lane far OOD (sim lane is 3 discrete values) -> the policy
    # never saw lane=-15 in training -> erratic -> walks off the edge (the 2026-06-26 device failure).
    # strafe IS needed at NARROWS -> only dock when there's NO narrow ahead (narrow_target_lane is None).
    # Uses a DEDICATED strafe_penalty (hard dock) so non-narrow strafing is strongly suppressed without
    # over-penalising idle jumps/slides; falls back to idle_action_penalty if strafe_penalty unset.
    strafe_dock = strafe_penalty if strafe_penalty else idle_action_penalty
    if (strafe_dock and action in (int(C.Action.STRAFE_LEFT), int(C.Action.STRAFE_RIGHT))
            and (prev is None or prev.get("narrow_target_lane") is None)):
        r -= strafe_dock
    return r
