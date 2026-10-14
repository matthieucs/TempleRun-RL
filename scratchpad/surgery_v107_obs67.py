"""WAVE 17 surgery: v107 (obs65 = 37 base + 28 hist, 512x512) -> obs67 by inserting TWO zero columns
for the horizon scalars `dist_next_must_jump` / `dist_next_must_slide` at full-vector indices 37,38
(the end of the base block, before the action history). Zero-init -> v107's behaviour is preserved
EXACTLY; the retrain learns the new cue.

WHY (USER, 2026-07-27, after watching v104 and then v107 die the same way): "The struggle is really
when you have 3 obstacles in a row. It really handles it poorly." and "it took a turn with an
obstacle. Just after the turn, there is an obstacle. In that case you have to jump before the
obstacle or during the turn."

MEASURED on those same device captures:
  * 3-obstacle chains kill 14.3% per encounter (2 of 14) and BOTH deaths land on the THIRD obstacle,
    while the human dies on 0 of 45.
  * 100% of chains -- device AND human -- have that third obstacle OUTSIDE the +3-cell boolean
    horizon at the moment the first jump has to be committed. At v210 a jump lasts 0.8 s and +3 cells
    is 0.86 s: the policy could see barely one jump ahead, while the human sees the whole screen.
So this is a PERCEPTION gap, not a training gap -- no amount of post-turn/chain exposure can teach
avoidance of something invisible at decision time. Precedent: the last lookahead obs added
(next2_must_jump, v51) was the single biggest device gain in the project, +108%.

Needs hook 17k on the phone (the +4..+6 obstacles are NOT derivable from the old stream) and
obs.HORIZON_OBS = True.
"""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO

OLD_PATH = 'runs/ppo_v107_calibrated_final.zip'
NEW_PATH = 'runs/ppo_v107_obs67_base.zip'
INSERT_AT = 37          # first of the two new base columns
N_NEW = 2

old = PPO.load(OLD_PATH, device='cpu')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[512, 512]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 65 and new.observation_space.shape[0] == 67

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 65 and nv.shape[1] == 67:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]
        w[:, INSERT_AT + N_NEW:] = ov[:, INSERT_AT:]
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), f'(zero cols @{INSERT_AT},{INSERT_AT+1})')
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

rng = np.random.default_rng(0); mism = 0
for _ in range(400):
    o_old = rng.random(65).astype(np.float32)
    o_new = np.insert(o_old, INSERT_AT, [rng.random(), rng.random()]).astype(np.float32)
    if int(old.predict(o_old, deterministic=True)[0]) != int(new.predict(o_new, deterministic=True)[0]):
        mism += 1
print(f'equivalence mismatches (want 0): {mism} /400')
new.save(NEW_PATH)
print('saved', NEW_PATH)
