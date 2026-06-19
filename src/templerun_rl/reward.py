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
) -> float:
    """Dense shaping term (0 unless a knob is set). Mirror of ``env._shaping_reward``.

    ``prev`` is the state the policy actually *saw* when it chose ``action`` (the obs from the
    previous step/reset), NOT the post-action state. ``last_action`` is the action taken on the
    previous step (for the turn-spam memory the policy lacks).
    """
    if (not action_cost and not reaction_bonus and not turn_spam_penalty
            and not idle_action_penalty and not redundant_action_penalty):
        return 0.0
    r = 0.0
    if action_cost and action != int(C.Action.NOOP):
        r -= action_cost
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
            if action == int(C.Action.TURN_LEFT) and prev.get("can_turn_left"):
                r += reaction_bonus
            elif action == int(C.Action.TURN_RIGHT) and prev.get("can_turn_right"):
                r += reaction_bonus
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
    # STRAFE dock: curbs the strafe-spam (high ent_coef + action-history let it back). But strafe IS
    # needed at NARROWS now -> only dock when there's NO narrow ahead (narrow_target_lane is None), so
    # purposeful narrow strafes are free while spam on plain track is penalized.
    if (idle_action_penalty and action in (int(C.Action.STRAFE_LEFT), int(C.Action.STRAFE_RIGHT))
            and (prev is None or prev.get("narrow_target_lane") is None)):
        r -= idle_action_penalty
    return r
