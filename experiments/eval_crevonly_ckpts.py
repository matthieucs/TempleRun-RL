"""Evaluate Phase-A crevasse-only checkpoints; pick the BEST (crevasse survival, weighted to HIGH speed
where the new direction+predicted_jump obs should help). Run AFTER Phase A finishes (avoid CPU contention)."""
import sys, glob, re; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, numpy as np
from stable_baselines3 import PPO

CFG = dict(turn_prob=0.0, turn_prob_late=0.0, obstacle_prob=0.0, zone_change_every_cells=6,
           curriculum_max_start_dist=8000, curriculum_deep_start_frac=0.6, curriculum_dist_power=1.0,
           domain_rand=False)

def eval_model(path, n_lo=40, n_hi=40):
    m = PPO.load(path)
    # low-speed band (normal start) and high-speed band (deep start) survival, separately.
    def run(deep, n):
        cfg = dict(CFG); cfg["curriculum_deep_start_frac"] = 1.0 if deep else 0.0
        if not deep: cfg["curriculum_max_start_dist"] = 0.0
        env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1, config=cfg).unwrapped
        dists = []
        for ep in range(n):
            o, _ = env.reset(seed=1000 + ep)
            done = False; d0 = env._distance_real
            while not done:
                a, _ = m.predict(o, deterministic=True)
                o, r, term, trunc, info = env.step(int(a)); done = term or trunc
            dists.append(env._distance_real - d0)
        return np.median(dists), np.mean(dists)
    lo_med, lo_mean = run(False, n_lo)
    hi_med, hi_mean = run(True, n_hi)
    return lo_med, hi_med, hi_mean

ckpts = ['runs/ppo_v14_obs30_leveldir.zip'] + sorted(
    glob.glob('runs/ppo_v25a_crevonly_[12]00000_steps.zip'),
    key=lambda p: int(re.search(r'_(\d+)_steps', p).group(1)))
print(f"{'checkpoint':<44} {'lo_med':>8} {'hi_med':>8} {'hi_mean':>8}  score")
best = None
for c in ckpts:
    lo, hi, him = eval_model(c)
    score = hi_med = hi   # high-speed median = the crevasse skill we care about most
    tag = ''
    if best is None or score > best[1]:
        best = (c, score); tag = '  <- best'
    print(f"{c.split('/')[-1]:<44} {lo:>8.0f} {hi:>8.0f} {him:>8.0f}  {score:>6.0f}{tag}")
print("\nBEST:", best[0], "hi_med=%.0f" % best[1])
