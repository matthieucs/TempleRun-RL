"""For every death: was the fatal hazard VISIBLE in the observation while the runner could still act?

Walk back from the death to the last frame where the runner was GROUNDED (able to jump/slide). At
that frame, read the SHIPPED obs ladder (must_jump / next / next2 / next3) and the landing cue.
  VISIBLE + actionable -> the policy HAD the information and failed = SKILL / training gap
  NOT visible          -> the information was never there = PERCEPTION gap
Anything else (already airborne the whole time) -> COMMITTED: the mistake happened earlier.
"""
import sys, glob, json, collections
sys.path.insert(0,"src")
from templerun_rl import obs as O
LH = O.OBS_FIELDS.index("landing_hazard")

def scan(pats,label):
    tally=collections.Counter(); seen=set(); lead=[]
    for pat in pats:
        for f in glob.glob(pat):
            rs=[json.loads(l) for l in open(f) if l.strip()]
            eps=collections.defaultdict(list)
            for r in rs:
                if 'velocity' in r: eps[(f,r.get('ep'))].append(r)
            for e,fr in eps.items():
                di=next((i for i,x in enumerate(fr) if x.get('death_type')),None)
                if di is None: continue
                k=round(float(fr[di].get('distance') or 0),3)
                if k in seen or k<200: continue
                seen.add(k)
                # last grounded frame before death = last moment an action was possible
                g=None
                for j in range(di,max(-1,di-25),-1):
                    if not fr[j].get('is_jumping') and not fr[j].get('is_sliding'):
                        g=j; break
                if g is None:
                    tally['COMMITTED (airborne throughout)']+=1; continue
                p=fr[g]
                vis = any(p.get(x) for x in ('must_jump','next_must_jump','next2_must_jump','next3_must_jump',
                                             'must_slide','next_must_slide','next2_must_slide','next3_must_slide'))
                if vis:
                    tally['VISIBLE at last actionable frame -> SKILL']+=1
                    lead.append(di-g)
                else:
                    tally['NOT visible -> PERCEPTION']+=1
    tot=sum(tally.values())
    print("\n%s  (%d deaths)" % (label,tot))
    for k,v in tally.most_common():
        print("   %-42s %2d = %3.0f%%" % (k,v,100*v/max(1,tot)))
    if lead:
        import statistics as st
        print("   frames of grace when VISIBLE: median %.1f (min %d, max %d)" % (st.median(lead),min(lead),max(lead)))

scan(['captures/w19_v133*.jsonl'],"v133")
scan(['captures/w18_v120_device.jsonl','captures/w18_v120_clean.jsonl'],"v120")
scan(['captures/human_2026*.jsonl'],"HUMAN")
