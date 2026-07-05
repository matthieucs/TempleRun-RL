"""Warm-start surgery: ppo_v34_carryfix_2000000 (obs 58 = 30 base + 28 action-hist) -> obs 60 (32 + 28)
by inserting two zero columns for next2_must_jump/next2_must_slide at base indices 30,31. Zero-init ->
v34's full skill set (turns/crevasses/jump rules) preserved EXACTLY; v35 then learns the 2-ahead
obstacle lookahead (chained-jump planning @v200 — the 4633/6399-m death class)."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np
from stable_baselines3 import PPO

old = PPO.load('runs/ppo_v34_carryfix_2000000_steps.zip')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[128, 128]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 58 and new.observation_space.shape[0] == 60, "unexpected widths"

INSERT_AT = 30
old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 58 and nv.shape[1] == 60:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]
        w[:, INSERT_AT + 2:] = ov[:, INSERT_AT:]
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero cols @30,31)')
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

rng = np.random.default_rng(0); mism = 0
for _ in range(300):
    o58 = rng.random(58).astype(np.float32)
    o60 = np.insert(o58, INSERT_AT, [rng.random(), rng.random()]).astype(np.float32)
    a_old = old.predict(o58, deterministic=True)[0]
    a_new = new.predict(o60, deterministic=True)[0]
    if int(a_old) != int(a_new): mism += 1
print(f'equivalence mismatches: {mism}/300')
assert mism == 0, "surgery broke equivalence"
new.save('runs/ppo_v34_obs32_look2.zip')
print('saved runs/ppo_v34_obs32_look2.zip')
