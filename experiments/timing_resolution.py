"""Is the safe action WINDOW narrower than one control step? The policy controls at 10 Hz while the
game runs at 60 Hz, so it can only place a takeoff on a grid of v*0.1/cell_length cells. If the
survivable window is comparable to that grid spacing, timing failures are STRUCTURAL, not skill.

window  = spread of landing margins among SURVIVED takeoffs (p25..p75, in cells)
grid    = cells advanced per control step = velocity * 0.1 / 60
"""
import sys, glob, json, collections, statistics as st
sys.path.insert(0,"src")
from templerun_rl import obs as O

def margin(p):
    v=float(p.get('velocity') or 0); cp=float(p.get('cell_progress') or 0)
    sp=max(0.0,O.LAND_SPAN_A*v+O.LAND_SPAN_B)
    land=cp+sp
    lad=[p.get('must_jump'),p.get('next_must_jump'),p.get('next2_must_jump'),p.get('next3_must_jump')]
    hz=[i for i,x in enumerate(lad) if x]
    if not hz: return None,v
    h=min(hz,key=lambda i:abs(land-i))
    return land-h, v

res=[]; seen=set()
for f in glob.glob('captures/w1*_v1*.jsonl'):
    rs=[json.loads(l) for l in open(f) if l.strip()]
    eps=collections.defaultdict(list)
    for r in rs:
        if 'velocity' in r: eps[(f,r.get('ep'))].append(r)
    for e,fr in eps.items():
        di=next((i for i,x in enumerate(fr) if x.get('death_type')),None)
        if di is not None:
            k=round(float(fr[di].get('distance') or 0),3)
            if k in seen or k<200: continue
            seen.add(k)
        for i in range(1,len(fr)):
            if fr[i].get('action')!=2: continue
            p=fr[i-1]
            if p.get('is_jumping') or p.get('is_dead'): continue
            m,v=margin(p)
            if m is None: continue
            died = di is not None and 0<=di-i<=15
            res.append((m,v,died))
surv=[m for m,v,d in res if not d]
vel=[v for m,v,d in res]
q1,q3=st.quantiles(surv,n=4)[0],st.quantiles(surv,n=4)[2]
print("survivable landing-margin window : %.2f .. %.2f cells  = WIDTH %.2f cells (n=%d)" % (q1,q3,q3-q1,len(surv)))
for v in (150,180,210):
    print("   one control step @v%d (10 Hz)  = %.2f cells   -> window is %.1f control steps"
          % (v, v*0.1/60.0, (q3-q1)/(v*0.1/60.0)))
print("   median takeoff velocity = %.0f" % st.median(vel))
print()
print("at 20 Hz the grid halves; at 30 Hz it thirds.")
