"""Eval that measures the DEVICE-PREDICTIVE metric: the policy's turn swipe-cp distribution.
With the tail-risk sim, a device-safe policy aims EARLY (cp<0.50). Reports frac of swipes in the
reliable zone (cp 0.35-0.50) vs marginal (>0.50), plus speed-bucketed distance/deaths."""
import sys, numpy as np
sys.path.insert(0,'src')
import gymnasium as gym, templerun_rl
from stable_baselines3 import PPO
CFG=dict(curriculum_max_start_dist=8000.0, domain_rand=False, hook_turn_buffer=True, turn_approach_buffer=True)
def evalm(path,n=120,seed0=40000):
    m=PPO.load(path)
    env=gym.make("TempleRunSim-v0",max_episode_steps=100000,action_history_len=4,frame_stack=1,config=CFG).unwrapped
    B={'slow':[],'fast':[]}; swipe_cps=[]; turns=0
    for ep in range(n):
        o,i=env.reset(seed=seed0+ep); st=i['raw_state']; v0=float(st.get('velocity',100))
        b='fast' if v0>=180 else ('slow' if v0<140 else None); since_t=99; cause='trunc'; steps=0; prev=st
        while True:
            a,_=m.predict(o,deterministic=True); a=int(a)
            if a in(4,5) and st.get('is_turn'):
                swipe_cps.append(float(st.get('cell_progress',0))); turns+=1
            o,r,term,trunc,info=env.step(a); st=info['raw_state']; steps+=1
            pt=st.get('path_type'); since_t=0 if prev.get('is_turn') else since_t+1; prev=st
            if term:
                if since_t<=2: cause='turn'
                elif pt in(33,34,35,36): cause='crevasse'
                elif isinstance(pt,int) and 17<=pt<=30: cause='narrow'
                else: cause='other'
                break
            if trunc or steps>4000: break
        if b: B[b].append((info['distance'],cause))
    sc=np.array(swipe_cps) if swipe_cps else np.array([0.44])
    out={'swipe_med':float(np.median(sc)),'frac_reliable':100*float(((sc>=0.35)&(sc<0.50)).mean()),
         'frac_marginal':100*float((sc>=0.50).mean()),'n_turns':turns}
    import collections
    for k,rows in B.items():
        if not rows: out[k]=None; continue
        c=collections.Counter(x[1] for x in rows if x[1]!='trunc')
        out[k]=dict(n=len(rows),med=float(np.median([x[0] for x in rows])),turn=c.get('turn',0),crev=c.get('crevasse',0),other=c.get('other',0),nar=c.get('narrow',0))
    return out
for name in sys.argv[1:]:
    r=evalm(name); tag=name.split('/')[-1].replace('.zip','')
    print("\n=== %s  swipe-cp med=%.2f  RELIABLE(0.35-0.50)=%.0f%%  MARGINAL(>0.50)=%.0f%%  (device-safe = high reliable, low marginal) ==="%(tag,r['swipe_med'],r['frac_reliable'],r['frac_marginal']))
    for b in ('slow','fast'):
        s=r[b]
        if s: print("  %-4s n=%2d med=%5.0f deaths: turn=%d crev=%d narrow=%d other=%d"%(b,s['n'],s['med'],s['turn'],s['crev'],s['nar'],s['other']))
