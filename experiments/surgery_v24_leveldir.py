"""Warm-start surgery: ppo_v14_obs28_predjump (obs 56 = 28 base + 28 action-hist) -> obs 58 (30 base + 28)
by INSERTING two zero columns for the new upcoming-altitude obs (next_path_level, next2_path_level) at base
indices 28,29 (end of base, before action history). Zero-init -> the new obs start with ZERO influence, so
v14's turn/narrow/crevasse behaviour + the predicted_jump wiring are preserved EXACTLY; the retrain then
learns to read the device-verified crevasse DIRECTION signal (sign(next2_path_level - path_level))."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np
from stable_baselines3 import PPO

old = PPO.load('runs/ppo_v14_obs28_predjump.zip')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[128, 128]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 56 and new.observation_space.shape[0] == 58, "unexpected widths"

INSERT_AT = 28   # base cols 0..27 kept; insert 2 zero cols (28,29); action-history 28..55 -> 30..57
old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 56 and nv.shape[1] == 58:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]              # base cols 0..27 (incl predicted_jump @27)
        # w[:, 28:30] stays 0                              # next_path_level, next2_path_level -> zero
        w[:, INSERT_AT + 2:] = ov[:, INSERT_AT:]          # action-history cols shifted right by 2
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero cols @28,29)')
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

# EQUIVALENCE: inserting ARBITRARY values at cols 28,29 must not change the action (their weight=0).
rng = np.random.default_rng(0); mism = 0
for _ in range(300):
    o56 = rng.random(56).astype(np.float32)
    o58 = np.insert(o56, INSERT_AT, [rng.random(), rng.random()]).astype(np.float32)
    a_old = old.predict(o56, deterministic=True)[0]
    a_new = new.predict(o58, deterministic=True)[0]
    if int(a_old) != int(a_new): mism += 1
print('equivalence mismatches (want 0):', mism, '/300')
new.save('runs/ppo_v14_obs30_leveldir.zip')
print('saved runs/ppo_v14_obs30_leveldir.zip')
