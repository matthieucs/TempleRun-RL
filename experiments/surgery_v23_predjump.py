"""Warm-start surgery: v14_obs27 (obs 55 = 27 base + 28 action-hist) -> obs 56 (28 base + 28) by INSERTING
a zero column for the new `predicted_jump` obs at base index 27 (right after predicted_landing @26). The
new input weight column is zero -> the new obs starts with ZERO influence, so v14's turn/narrow/crevasse
behaviour is preserved EXACTLY; the retrain then learns to use the new velocity-invariant step-up cue."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np
from stable_baselines3 import PPO

old = PPO.load('runs/ppo_v14_obs27.zip')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[128, 128]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 55 and new.observation_space.shape[0] == 56, "unexpected widths"

# new col j -> source col in old: [0..26]=old[0..26]; [27]=ZERO (predicted_jump); [28..55]=old[27..54].
INSERT_AT = 27
old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 55 and nv.shape[1] == 56:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]        # base cols 0..26
        # w[:, INSERT_AT] stays 0                    # predicted_jump (new) -> zero influence
        w[:, INSERT_AT + 1:] = ov[:, INSERT_AT:]    # action-history cols shifted right by 1
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero col @%d)' % INSERT_AT)
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

# EQUIVALENCE: for any obs, inserting an ARBITRARY value at col 27 must not change the action (weight=0).
rng = np.random.default_rng(0); mism = 0
for _ in range(300):
    o55 = rng.random(55).astype(np.float32)
    o56 = np.insert(o55, INSERT_AT, rng.random()).astype(np.float32)   # arbitrary predicted_jump value
    a_old = old.predict(o55, deterministic=True)[0]
    a_new = new.predict(o56, deterministic=True)[0]
    if int(a_old) != int(a_new): mism += 1
print('equivalence mismatches (want 0):', mism, '/300')
new.save('runs/ppo_v14_obs28_predjump.zip')
print('saved runs/ppo_v14_obs28_predjump.zip')
