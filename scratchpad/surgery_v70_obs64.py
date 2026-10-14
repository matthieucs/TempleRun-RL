"""WAVE 11 surgery: ppo_v70@20M (obs63 = 35 base + 28 hist, 512x512) -> obs64 by inserting one zero
column for `predicted_turn` at FULL-vector index 35 (after next2_narrow_lane @34, before the action
history). Zero-init -> v70's behaviour preserved EXACTLY; the retrain learns the swipe cue."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO

OLD_PATH = 'runs/ppo_v70_consolidated_20000768_steps.zip'
NEW_PATH = 'runs/ppo_v70_obs64_pturn.zip'
INSERT_AT = 35

old = PPO.load(OLD_PATH, device='cpu')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[512, 512]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 63 and new.observation_space.shape[0] == 64

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 63 and nv.shape[1] == 64:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]
        w[:, INSERT_AT + 1:] = ov[:, INSERT_AT:]
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero col @35)')
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

rng = np.random.default_rng(0); mism = 0
for _ in range(400):
    o_old = rng.random(63).astype(np.float32)
    o_new = np.insert(o_old, INSERT_AT, rng.random()).astype(np.float32)
    if int(old.predict(o_old, deterministic=True)[0]) != int(new.predict(o_new, deterministic=True)[0]):
        mism += 1
print(f'equivalence mismatches (want 0): {mism} /400')
new.save(NEW_PATH)
print('saved', NEW_PATH)
