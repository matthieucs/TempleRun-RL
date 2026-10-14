"""Function-preserving net widen: v56 (256x256, obs60) -> 512x512. For every Linear, copy the old
weight/bias into the TOP-LEFT block of a zero-filled larger tensor. The new hidden units get zero
input weight + zero bias -> tanh(0)=0 output, and zero OUTPUT weight (the next layer's new columns
are zero) -> they contribute NOTHING. So the widened net is BIT-IDENTICAL to v56 at init; the new
512-net capacity is dormant and activates during training. Verified by an action-equivalence check."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO

OLD = 'runs/ppo_v56_challenge_best.zip'
NEW = 'runs/ppo_v56_512.zip'

old = PPO.load(OLD, device='cpu')
assert old.observation_space.shape[0] == 60, old.observation_space.shape
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[512, 512]))
print('old', old.observation_space.shape, old.policy.mlp_extractor.policy_net,
      '\n-> new 512x512')

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    else:
        w = torch.zeros_like(nv)
        if ov.dim() == 2:
            w[:ov.shape[0], :ov.shape[1]] = ov      # top-left block
        else:
            w[:ov.shape[0]] = ov                    # bias / 1-D
        new_sd[k] = w
        print('WIDEN', k, tuple(ov.shape), '->', tuple(nv.shape))
new.policy.load_state_dict(new_sd)

# EQUIVALENCE: the widened net must produce the SAME action as v56 on random obs (dormant new units).
rng = np.random.default_rng(0); mism = 0
for _ in range(500):
    o = rng.random(60).astype(np.float32)
    if int(old.predict(o, deterministic=True)[0]) != int(new.predict(o, deterministic=True)[0]):
        mism += 1
print('equivalence mismatches (want 0):', mism, '/500')
assert mism == 0, "widen not function-preserving!"
new.save(NEW)
print('saved', NEW)
