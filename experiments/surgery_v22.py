"""Warm-start surgery: v14 (obs 60 = 32 base + 28 action-history) -> obs 55 (27 base + 28) by dropping the
5 power-up columns (base indices 5-9). Exact: power-ups were always 0, so 0*weight contributes nothing."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np
from stable_baselines3 import PPO

old = PPO.load('runs/ppo_v14_best.zip')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1,
               config=dict(small_narrow_frac=0.35)).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[128, 128]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 60 and new.observation_space.shape[0] == 55

keep = list(range(0, 5)) + list(range(10, 60))   # drop base cols 5..9 (power-ups); keep the rest + action-hist
assert len(keep) == 55

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 60 and nv.shape[1] == 55:
        new_sd[k] = ov[:, keep].clone()
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape))
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

# EQUIVALENCE CHECK: random obs with power-ups=0 -> old(60) and new(55) must give the same action logits.
rng = np.random.default_rng(0); mism = 0
for _ in range(200):
    o60 = rng.random(60).astype(np.float32); o60[5:10] = 0.0   # power-ups zero (as on device/sim)
    o55 = o60[keep]
    a_old = old.predict(o60, deterministic=True)[0]
    a_new = new.predict(o55, deterministic=True)[0]
    if int(a_old) != int(a_new): mism += 1
print('equivalence mismatches (want 0):', mism, '/200')
new.save('runs/ppo_v14_obs27.zip')
print('saved runs/ppo_v14_obs27.zip')
