"""Warm-start surgery: ppo_v66 (obs 62-2=60 base 32... obs60 = 32 base + 28 action-hist, net 512x512)
-> obs 62 (34 base + 28 hist) by INSERTING two zero columns for the new 3-ahead-obstacle obs
(next3_must_jump, next3_must_slide) at FULL-vector indices 21,22 (right after next2_must_slide @20,
before the first scalar `velocity`). Zero-init -> v66's behaviour preserved EXACTLY; the retrain
learns the 3-ahead signal. v66 (obs60) STAYS deployable on hook 16z — this FORKS the obs62/17a lineage."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO

OLD_PATH = 'runs/ppo_v66_512_5500928_steps.zip'
NEW_PATH = 'runs/ppo_v66_obs62_next3.zip'
INSERT_AT = 21   # cols 0..20 kept (21 bools incl next2_must_slide@20); insert 2 zero cols (21,22)

old = PPO.load(OLD_PATH, device='cpu')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[512, 512]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 60 and new.observation_space.shape[0] == 62, "unexpected widths"

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 60 and nv.shape[1] == 62:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]
        w[:, INSERT_AT + 2:] = ov[:, INSERT_AT:]      # scalars/extras/hist shifted right by 2
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero cols @21,22)')
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

rng = np.random.default_rng(0); mism = 0
for _ in range(400):
    o60 = rng.random(60).astype(np.float32)
    o62 = np.insert(o60, INSERT_AT, [rng.random(), rng.random()]).astype(np.float32)
    a_old = old.predict(o60, deterministic=True)[0]
    a_new = new.predict(o62, deterministic=True)[0]
    if int(a_old) != int(a_new):
        mism += 1
print(f'equivalence mismatches (want 0): {mism} /400')
new.save(NEW_PATH)
print('saved', NEW_PATH)
