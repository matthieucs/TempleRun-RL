"""Network-widen surgery: v36 (128x128) -> 256x256, zero-padding the new neurons so the wider net is
FUNCTIONALLY IDENTICAL at init (v36's skills preserved exactly), then training grows the new capacity.
Goal: break the narrow<->crevasse interference (a 128x128 net can't hold reach-center-narrows AND
centered-crevasses; it oscillates). More hidden units = room for both."""
import sys; sys.path.insert(0, "src")
import gymnasium as gym, templerun_rl, torch, numpy as np
from stable_baselines3 import PPO

OLD, NEW = 128, 256
old = PPO.load("runs/ppo_v36_compound_2000000_steps.zip")
env = gym.make("TempleRunSim-v0", action_history_len=4, frame_stack=1).unwrapped
new = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[NEW, NEW]))

osd, nsd = old.policy.state_dict(), new.policy.state_dict()
def widen(name, ov, nv):
    w = torch.zeros_like(nv)
    if ov.dim() == 2:
        r, c = ov.shape
        w[:r, :c] = ov          # top-left block = old weights; new rows/cols stay 0
    else:
        w[:ov.shape[0]] = ov    # bias: first OLD entries = old, rest 0
    return w

mism = 0
for k in nsd:
    if k not in osd:
        print("NEW-ONLY", k); continue
    ov, nv = osd[k], nsd[k]
    if ov.shape == nv.shape:
        nsd[k] = ov.clone()
    else:
        nsd[k] = widen(k, ov, nv)
        print(f"WIDEN {k}: {tuple(ov.shape)} -> {tuple(nv.shape)}")
new.policy.load_state_dict(nsd)

# EQUIVALENCE: the widened net must produce the SAME action as v36 for random obs (new neurons = 0).
for _ in range(300):
    o = np.random.default_rng().random(58).astype(np.float32)
    if int(old.predict(o, deterministic=True)[0]) != int(new.predict(o, deterministic=True)[0]):
        mism += 1
print(f"equivalence mismatches: {mism}/300")
assert mism == 0, "widen broke equivalence"
new.save("runs/ppo_v36_256.zip")
print("saved runs/ppo_v36_256.zip (256x256, v36-equivalent)")
