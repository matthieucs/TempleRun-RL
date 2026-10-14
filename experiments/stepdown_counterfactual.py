"""Is WALKING a step-down actually free in the sim? Force NoOp whenever soft_stepdown is set and
compare survival with the unmodified policy. If forced-walk dies more, the policy jumps for a REASON
and no penalty will move it; if it is free, the habit is pure inertia and only the dose is wrong."""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, json
sys.path.insert(0,"src"); sys.path.insert(0,"cpp_sim")
import numpy as np, statistics as st
from stable_baselines3 import PPO
from cpp_vec_env import CppTempleRunVecEnv

cfg=dict(domain_rand=False,turn_approach_buffer=True,turn_linger_frames=3,curriculum_max_start_dist=5000.0)
cfg.update(json.load(open('experiments/_v128_cfg.json')))
m=PPO.load('runs/ppo_v120_obs67_base.zip',device='cpu')
NE=96
def run(force_walk):
    env=CppTempleRunVecEnv(num_envs=NE,config=cfg,seed=11,max_episode_steps=8000,
                           action_history_len=4,frame_stack=1)
    o=env.reset(); done=np.zeros(NE,bool); dist=[None]*NE; forced=0
    for _ in range(8005):
        a,_=m.predict(np.asarray(o),deterministic=True); a=np.asarray(a).reshape(-1).copy()
        if force_walk:
            for i in range(NE):
                if done[i]: continue
                r=env.raw_dict(i)
                if a[i]==2 and (r.get('soft_stepdown') or r.get('next_soft_stepdown')):
                    a[i]=6; forced+=1          # NoOp instead of JUMP
        o,_,dn,inf=env.step(a)
        for i in np.nonzero(dn)[0]:
            if not done[i]:
                done[i]=True; dist[i]=float(inf[i].get('distance') or 0)
        if done.all(): break
    env.close()
    got=[d for d in dist if d is not None]
    return st.median(got), st.mean(got), forced
a=run(False); b=run(True)
print("policy as-is        : median %6.0f  mean %6.0f" % (a[0],a[1]))
print("forced WALK at step-downs: median %6.0f  mean %6.0f   (%d jumps suppressed)" % (b[0],b[1],b[2]))
print("=> walking is %s" % ("FREE (habit is inertia -> dose is the only issue)" if b[0]>=a[0]*0.95
                            else "COSTLY (the policy jumps for a reason)"))
