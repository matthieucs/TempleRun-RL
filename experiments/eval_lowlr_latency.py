"""Eval v25c (low-LR warm-start) checkpoints on HIGH-SPEED crevasses UNDER jump latency (the deployment-
relevant hardness). Win = hi_med >> v14's 130 (i.e. the policy learned to PRE-AIM). Also report survival
WITHOUT latency (should stay near-oracle -> no forgetting)."""
import sys, glob, re; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, numpy as np
from stable_baselines3 import PPO

def hi_med(path, lat, jit, n=40):
    m = PPO.load(path)
    cfg = dict(turn_prob=0, turn_prob_late=0, obstacle_prob=0, zone_change_every_cells=6,
               curriculum_max_start_dist=8000, curriculum_deep_start_frac=1.0, curriculum_dist_power=1.0,
               domain_rand=False, jump_exec_lat_frames=lat, jump_exec_lat_jitter=jit)
    env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1, config=cfg).unwrapped
    ds = []
    for ep in range(n):
        o, _ = env.reset(seed=5000 + ep); done = False; d0 = env._distance_real
        while not done:
            a, _ = m.predict(o, deterministic=True); o, r, term, trunc, info = env.step(int(a)); done = term or trunc
        ds.append(env._distance_real - d0)
    return np.median(ds), sum(1 for d in ds if d < 200)

models = ['runs/ppo_v14_obs30_leveldir.zip'] + sorted(
    [p for p in glob.glob('runs/ppo_v25c_lowlr_*_steps.zip') if int(re.search(r'_(\d+)_steps',p).group(1)) in (300000,400000,500000,600000)], key=lambda p: int(re.search(r'_(\d+)_steps', p).group(1)))
print(f"{'model':<42} {'hiMed@lat2±1':>13} {'early_deaths':>12} {'hiMed@nolat':>12}")
best = None
for p in models:
    lm, led = hi_med(p, 2, 1)
    nolat, _ = hi_med(p, 0, 0)
    tag = ''
    if 'v14' not in p and (best is None or lm > best[1]):
        best = (p, lm); tag = '  <- best'
    print(f"{p.split('/')[-1]:<42} {lm:>13.0f} {led:>12} {nolat:>12.0f}{tag}")
if best: print("\nBEST v25c:", best[0], "hiMed@lat=%.0f (v14@lat=130)" % best[1])
