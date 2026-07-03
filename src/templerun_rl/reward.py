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
            and not slide_jump_penalty):
        return 0.0
    r = 0.0
    # JUMP into a slide-obstacle is FATAL on device [MEASURED 2026-07-03, ppo_speed_v8: 4/8 deaths] —
    # a TreeSlide/FlameTower is a LOW bar you pass UNDER; jumping lifts you into it. Dock a JUMP issued
    # when a slide-obstacle is HERE / AHEAD or a slide is already in progress, so the policy stops
    # popping up mid-slide (the "slid then jumped and died" bug). Mirrors the sim death model.
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
