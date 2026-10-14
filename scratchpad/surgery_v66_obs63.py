"""Warm-start surgery (WAVE 9, 2026-07-09): ppo_v66_obs62_next3 (obs62 = 34 base + 28 hist, net 512x512)
-> obs63 (35 base + 28 hist) by INSERTING one zero column for the new 2-ahead-narrow-side obs
(next2_narrow_lane) at FULL-vector index 34 (right after next2_path_level @33, before the action
history). Zero-init -> v66's behaviour preserved EXACTLY; the retrain learns the narrow side signal
(the jump->narrow mid-air tilt fix). The obs62 artifact STAYS deployable on hook 17a — this forks the
obs63/17f lineage."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO

OLD_PATH = 'runs/ppo_v66_obs62_next3.zip'
NEW_PATH = 'runs/ppo_v66_obs63_n2nl.zip'
INSERT_AT = 34   # cols 0..33 kept (23 bools + 4 scalars + 7 extras); insert 1 zero col @34 (next2_narrow_lane)

old = PPO.load(OLD_PATH, device='cpu')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[512, 512]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 62 and new.observation_space.shape[0] == 63, "unexpected widths"

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 62 and nv.shape[1] == 63:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]
        w[:, INSERT_AT + 1:] = ov[:, INSERT_AT:]      # action-hist cols shifted right by 1
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero col @34)')
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

rng = np.random.default_rng(0); mism = 0
for _ in range(400):
    o62 = rng.random(62).astype(np.float32)
    o63 = np.insert(o62, INSERT_AT, rng.random()).astype(np.float32)
    a_old = old.predict(o62, deterministic=True)[0]
    a_new = new.predict(o63, deterministic=True)[0]
    if int(a_old) != int(a_new):
        mism += 1
print(f'equivalence mismatches (want 0): {mism} /400')
new.save(NEW_PATH)
print('saved', NEW_PATH)
