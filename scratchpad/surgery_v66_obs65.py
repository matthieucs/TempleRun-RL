"""WAVE 16 surgery: ppo_v66_obs64_base (obs64 = 36 base + 28 hist, 512x512) -> obs65 by inserting one
zero column for `landing_hazard` at FULL-vector index 36 (last base field, before the action history).
Zero-init -> the base policy's behaviour is preserved EXACTLY; the retrain learns the new cue.

WHY: a behavioural audit of 44 device deaths (scripts/death_behaviour.py) found 60% are "my jump arc
landed me somewhere fatal" -- 55% COMPOUND TRAP (airborne into the fatal hazard from a jump that WAS
justified) + 5% RANDOM-JUMP TRAP -- with takeoff->fatal distance clustering at 2.0-3.5 cells, exactly
the arc span. 84% of deaths had another hazard within 5 cells behind the fatal one (33/37 at gap 1-2).
The policy already gets next1/next2/next3_must_jump but must do velocity-dependent arithmetic to
relate them to its own arc, which it demonstrably does not. landing_hazard collapses that into one
velocity-invariant scalar (1.0 hazard at the landing cell / 0.0 clear / 0.5 beyond the horizon) --
the same trick that worked for predicted_landing (turns) and predicted_jump (crevasse takeoff)."""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO

OLD_PATH = 'runs/ppo_v66_obs64_base.zip'
NEW_PATH = 'runs/ppo_v66_obs65_land.zip'
INSERT_AT = 36

old = PPO.load(OLD_PATH, device='cpu')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[512, 512]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == 64 and new.observation_space.shape[0] == 65

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == 64 and nv.shape[1] == 65:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]
        w[:, INSERT_AT + 1:] = ov[:, INSERT_AT:]
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero col @36)')
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

rng = np.random.default_rng(0); mism = 0
for _ in range(400):
    o_old = rng.random(64).astype(np.float32)
    o_new = np.insert(o_old, INSERT_AT, rng.random()).astype(np.float32)
    if int(old.predict(o_old, deterministic=True)[0]) != int(new.predict(o_new, deterministic=True)[0]):
        mism += 1
print(f'equivalence mismatches (want 0): {mism} /400')
new.save(NEW_PATH)
print('saved', NEW_PATH)
