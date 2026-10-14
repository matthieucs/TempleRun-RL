#!/usr/bin/env python3
"""WAVE 13 mechanism probe for the turn dead-letter fix (Lever A = turn_buffer_bonus).

The plateau arms all reach ~1050 isolated median but we never measured WHY: do they pre-swipe on the
approach cell (latency-robust) or react on the junction (dead-letters)? Episode distance is noisy; this
reads the behavior directly. For each junction the policy encounters under the isolated turn-cfg, classify:
  PRE-SWIPE  : a valid-direction turn issued on the APPROACH cell (next_is_turn, not is_turn) -> latches
  REACTIVE   : a valid-direction turn issued ON the junction cell (is_turn) -> dead-letters under lat>0
  NEITHER    : no valid turn issued near the junction (missed / wrong side)
Reports the fraction of junctions handled by each route. A working Lever A shifts REACTIVE -> PRE-SWIPE.

Usage: python scratchpad/preswipe_fraction.py <ckpt1> [<ckpt2> ...] [--episodes N]
"""
import sys, argparse
sys.path.insert(0, 'src')
import numpy as np, gymnasium as gym, templerun_rl  # noqa: F401
from stable_baselines3 import PPO
from templerun_rl import constants as C

TURN_CFG = {  # DEVICE_TRUE_CFG + isolated turn stress (matches eval_ckpt_torch override)
    "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": True,
    "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1,
    "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "domain_rand": False,
    "turn_approach_buffer": True, "turn_linger_frames": 3, "curriculum_max_start_dist": 0.0,
    "stepup_clip_band": 0.25, "wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
    "crevasse_apex_all": True,
    "turn_send_lat_frames": 0.7, "turn_offjunction_drop": True, "turn_drop_debounce_s": 0.3,
    "post_land_gap_rejump_prob": 0.46,
}
TL, TR = int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT)


def infer_k_fs(obs_dim):
    from templerun_rl import obs as O
    for fs in (1, 2, 3, 4):
        rem = obs_dim - fs * O.OBS_LEN
        if rem >= 0 and rem % C.N_ACTIONS == 0 and rem // C.N_ACTIONS <= 8:
            return rem // C.N_ACTIONS, fs
    raise ValueError(f"cannot infer K/FS from obs dim {obs_dim}")


def _correct_turn_prob(model, obs, ncanl, ncanr):
    """Policy's probability mass on the CORRECT-direction turn (leading indicator: moves before the
    deterministic argmax flips). Uses the SB3 action distribution."""
    import torch
    obs_t = model.policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        probs = model.policy.get_distribution(obs_t).distribution.probs.cpu().numpy()[0]
    if ncanl and not ncanr:
        return float(probs[TL])
    if ncanr and not ncanl:
        return float(probs[TR])
    return float(probs[TL] + probs[TR])   # ambiguous junction: either turn counts


def classify(model, episodes, seed):
    K, FS = infer_k_fs(int(model.observation_space.shape[0]))
    env = gym.make("TempleRunSim-v0", config=TURN_CFG,
                   action_history_len=K, frame_stack=FS).unwrapped
    pre = react = neither = 0
    ap_prob_sum, ap_prob_n = 0.0, 0   # mean P(correct turn) on approach cells (leading indicator)
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        # REGION LATCH: a junction region = frames where the approach (next_is_turn) OR the junction
        # (is_turn) flag is up. A pre-swiped turn fires on junction ENTRY (is_turn may never latch a full
        # frame), so we tally on the FALLING edge of the region, classifying by what fired inside it.
        region = False
        handled_pre = handled_react = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            st = info["raw_state"]
            is_turn = bool(st.get("is_turn"))
            next_is_turn = bool(st.get("next_is_turn"))
            canl, canr = bool(st.get("can_turn_left")), bool(st.get("can_turn_right"))
            ncanl, ncanr = bool(st.get("next_can_turn_left")), bool(st.get("next_can_turn_right"))
            in_region = next_is_turn or is_turn
            if next_is_turn and not is_turn:   # approach cell: record leading-indicator prob
                ap_prob_sum += _correct_turn_prob(model, obs, ncanl, ncanr); ap_prob_n += 1
            if in_region:
                # approach cell -> pre-swipe; junction cell -> reactive (both valid-direction only)
                if next_is_turn and not is_turn and ((a == TL and ncanl) or (a == TR and ncanr)):
                    handled_pre = True
                if is_turn and ((a == TL and canl) or (a == TR and canr)):
                    handled_react = True
            if region and not in_region:                 # left the junction region -> tally once
                if handled_pre:
                    pre += 1
                elif handled_react:
                    react += 1
                else:
                    neither += 1
                handled_pre = handled_react = False
            region = in_region
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
    tot = pre + react + neither
    ap_prob = ap_prob_sum / ap_prob_n if ap_prob_n else 0.0
    return pre, react, neither, tot, ap_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=24680)
    args = ap.parse_args()
    print(f"isolated turn-cfg (lat0.7 + drop + debounce), {args.episodes} eps")
    print(f"{'checkpoint':46} {'PRE%':>6} {'REACT%':>7} {'MISS%':>6}  {'P(turn|approach)':>16}  {'n_j':>5}")
    for ck in args.checkpoints:
        model = PPO.load(ck, device="cpu")
        pre, react, neither, tot, ap_prob = classify(model, args.episodes, args.seed)
        if tot == 0:
            print(f"{ck.split('/')[-1]:46}  (no junctions reached)")
            continue
        print(f"{ck.split('/')[-1]:46} {100*pre/tot:6.1f} {100*react/tot:7.1f} "
              f"{100*neither/tot:6.1f}  {ap_prob:16.6f}  {tot:5d}")


if __name__ == "__main__":
    main()
