"""Generic obs65 -> obs66 surgery (zero column for slide_landing_hazard, index 37).

Behaviour-preserving by construction, so a model widened here evaluates EXACTLY as its obs65 self --
which is what lets the pre-obs66 arms (v113/v114/v115) be re-scored against v111 in the current env.
"""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO
from templerun_rl import obs as O

INSERT_AT = O.OBS_LEN - 1
OLD_W, NEW_W = O.OBS_LEN - 1 + 28, O.OBS_LEN + 28
assert O.OBS_FIELDS[INSERT_AT] == 'slide_landing_hazard'

for src in sys.argv[1:]:
    dst = src.replace('.zip', '_obs66.zip')
    old = PPO.load(src, device='cpu')
    if old.observation_space.shape[0] != OLD_W:
        print('SKIP %s (obs %d)' % (src, old.observation_space.shape[0])); continue
    env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
    new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[512, 512]))
    osd, nsd = old.policy.state_dict(), new.policy.state_dict()
    for k in nsd:
        if k not in osd: continue
        ov, nv = osd[k], nsd[k]
        if ov.shape == nv.shape:
            nsd[k] = ov.clone()
        elif ov.dim() == 2 and ov.shape[1] == OLD_W and nv.shape[1] == NEW_W:
            w = torch.zeros_like(nv)
            w[:, :INSERT_AT] = ov[:, :INSERT_AT]; w[:, INSERT_AT + 1:] = ov[:, INSERT_AT:]
            nsd[k] = w
    new.policy.load_state_dict(nsd)
    rng = np.random.default_rng(0); mism = 0
    for _ in range(300):
        a = rng.random(OLD_W).astype(np.float32)
        b = np.insert(a, INSERT_AT, rng.random()).astype(np.float32)
        if int(old.predict(a, deterministic=True)[0]) != int(new.predict(b, deterministic=True)[0]):
            mism += 1
    assert mism == 0, ('%s NOT behaviour-preserving: %d/300' % (src, mism))
    new.save(dst); print('%-52s -> %s  (0/300 mismatches)' % (src.split('/')[-1], dst.split('/')[-1]))
