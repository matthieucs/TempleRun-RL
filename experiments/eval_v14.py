import sys, numpy as np
sys.path.insert(0,'src')
import gymnasium as gym, templerun_rl
from stable_baselines3 import PPO
CFG={'curriculum_max_start_dist':8000.0,'domain_rand':False,'turn_approach_buffer':True}
def evalm(path,n=120,seed0=40000):
    m=PPO.load(path)
    env=gym.make("TempleRunSim-v0",max_episode_steps=100000,action_history_len=4,frame_stack=1,config=CFG).unwrapped
    B={'slow':[],'fast':[]}; nlane=[]; offctr=0; turns=0
    for ep in range(n):
        o,i=env.reset(seed=seed0+ep); st=i['raw_state']; prev=st; v0=float(st.get('velocity',100))
        b='fast' if v0>=180 else ('slow' if v0<140 else None); since_t=99; cause='trunc'; steps=0
        while True:
            a,_=m.predict(o,deterministic=True); a=int(a)
            o,r,term,trunc,info=env.step(a); st=info['raw_state']; steps+=1
            pt=st.get('path_type'); lane=st.get('lane',0.0)
            since_t=0 if prev.get('is_turn') else since_t+1
            if isinstance(pt,int) and 17<=pt<=30: nlane.append(abs(lane))
            if prev.get('is_turn') and a in(4,5):
                turns+=1
                if abs(prev.get('lane',0))>3: offctr+=1
            prev=st
            if term:
                if since_t<=2: cause='turn'
                elif pt in(33,34,35,36): cause='crevasse'
                elif isinstance(pt,int) and 17<=pt<=30: cause='narrow'
                elif abs(lane)>8: cause='lateral'
                else: cause='other'
                break
            if trunc or steps>3000: break
        if b: B[b].append((info['distance'],cause))
    out={'nlane':float(np.mean(nlane)) if nlane else 0,'offctr':100*offctr/max(1,turns)}
    for k,rows in B.items():
        if not rows: out[k]=None; continue
        import collections; c=collections.Counter(x[1] for x in rows if x[1]!='trunc')
        out[k]=dict(n=len(rows),med=float(np.median([x[0] for x in rows])),
                    surv=100*np.mean([1 for x in rows if x[1]=='trunc'])/max(1,len(rows))*len(rows)/max(1,len(rows)),
                    crev=c.get('crevasse',0),turn=c.get('turn',0),other=c.get('other',0),nar=c.get('narrow',0))
    return out
for name in sys.argv[1:]:
    r=evalm(name); tag=name.split('/')[-1].replace('.zip','')
    print(f"\n=== {tag} nlane={r['nlane']:.1f} offctr-turn={r['offctr']:.0f}% ===")
    for b in ('slow','fast'):
        s=r[b]
        if s: print(f"  {b:4} n={s['n']:2d} med={s['med']:5.0f} deaths: crev={s['crev']} turn={s['turn']} narrow={s['nar']} other={s['other']}")
