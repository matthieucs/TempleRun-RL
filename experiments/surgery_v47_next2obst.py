"""Warm-start surgery: ppo_v47 (obs 58 = 30 base + 28 action-hist, net 256x256) -> obs 60
(32 base + 28 action-hist) by INSERTING two zero columns for the new 2-ahead-obstacle obs
(next2_must_jump, next2_must_slide) at FULL-vector indices 19,20 (right after
next_ground_change_higher @18, before the first scalar `velocity`).

Zero-init -> the two new obs start with ZERO influence, so v47's entire hole-landing / crevasse /
turn behaviour is preserved EXACTLY; the retrain then learns to read the 2-ahead chained-obstacle
signal. v47 (obs58) STAYS deployable on the old hook — this is a FORK to a new obs60 lineage.
"""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO

OLD_PATH = 'runs/ppo_v47_land_1703936_steps.zip'
NEW_PATH = 'runs/ppo_v47_obs60_next2.zip'
INSERT_AT = 19   # full-vector index: cols 0..18 kept (19 bools); insert 2 zero cols (19,20); 19..57 -> 21..59

old = PPO.load(OLD_PATH, device='cpu')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[256, 256]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 58 and new.observation_space.shape[0] == 60, "unexpected widths"

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 58 and nv.shape[1] == 60:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]          # bool cols 0..18 (incl next_ground_change_higher@18)
        # w[:, 19:21] stays 0                          # next2_must_jump, next2_must_slide -> zero influence
        w[:, INSERT_AT + 2:] = ov[:, INSERT_AT:]      # scalars/extras/action-history shifted right by 2
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero cols @19,20)')
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

# EQUIVALENCE: inserting ARBITRARY values at cols 19,20 must not change the action (their weight=0).
rng = np.random.default_rng(0); mism = 0
for _ in range(400):
    o58 = rng.random(58).astype(np.float32)
    o60 = np.insert(o58, INSERT_AT, [rng.random(), rng.random()]).astype(np.float32)
    a_old = old.predict(o58, deterministic=True)[0]
    a_new = new.predict(o60, deterministic=True)[0]
    if int(a_old) != int(a_new):
        mism += 1
print('equivalence mismatches (want 0):', mism, '/400')
assert mism == 0, "surgery not equivalence-preserving!"
new.save(NEW_PATH)
print('saved', NEW_PATH)
