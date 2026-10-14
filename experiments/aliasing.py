"""ALIASING TEST: do FATAL states look like SURVIVED states in the policy's own observation?

If the obs vector at the last actionable frame before a death has near-identical neighbours among
frames the policy survived, then the same input maps to different required actions -- a partial
observability limit that NO extra column fixes on its own (it needs memory/recurrence or a genuinely
new state variable). If fatal states are well separated, the information is there and the gap is
optimisation/skill.
"""
import sys, glob, json, collections
sys.path.insert(0,"src")
import numpy as np
from templerun_rl import obs as O

fatal=[]; safe=[]; seen=set()
for f in glob.glob('captures/w1*_v1*.jsonl'):
    rs=[json.loads(l) for l in open(f) if l.strip()]
    eps=collections.defaultdict(list)
    for r in rs:
        if 'velocity' in r: eps[(f,r.get('ep'))].append(r)
    for e,fr in eps.items():
        di=next((i for i,x in enumerate(fr) if x.get('death_type')),None)
        key=round(float(fr[di].get('distance') or 0),3) if di is not None else None
        if key is not None:
            if key in seen or key<200: continue
            seen.add(key)
        g=None
        if di is not None:
            for j in range(di,max(-1,di-25),-1):
                if not fr[j].get('is_jumping') and not fr[j].get('is_sliding'): g=j; break
        for i,x in enumerate(fr):
            if x.get('is_jumping') or x.get('is_sliding') or x.get('is_dead'): continue
            v=O.state_to_obs(x)
            if di is not None and g is not None and i==g: fatal.append(v)
            elif di is None or i < di-40: safe.append(v)
S=np.array(safe,dtype=np.float32); F=np.array(fatal,dtype=np.float32)
print("fatal states %d | safe states %d | obs dim %d" % (len(F),len(S),S.shape[1]))
d=[]
for v in F:
    dist=np.linalg.norm(S-v,axis=1)
    d.append(np.sort(dist)[:5].mean())
base=[]
rng=np.random.default_rng(0)
for v in S[rng.choice(len(S),size=min(300,len(S)),replace=False)]:
    dist=np.linalg.norm(S-v,axis=1)
    base.append(np.sort(dist)[1:6].mean())
print("mean L2 to 5 nearest SAFE states:")
print("   from a FATAL state : %.4f" % np.mean(d))
print("   from a SAFE  state : %.4f  (baseline density)" % np.mean(base))
r=np.mean(d)/max(1e-9,np.mean(base))
print("   ratio %.2f  -> %s" % (r, "ALIASED: fatal states sit inside the safe cloud (partial observability)"
      if r < 1.5 else "SEPARATED: the obs distinguishes them -> skill/optimisation gap"))
