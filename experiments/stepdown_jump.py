"""USER 2026-07-30: "penalize the policy from random jumping the mid to low transition. It is
useless and it can be a source of death."

A step-DOWN to lower ground is foot-crossable (human walks 17/26 and never dies), but a jump there
launches a 2.5-3 cell arc that can terminate in a hazard = the dominant death mode. Measures how
often each agent JUMPS at a descending level change, and how often that precedes a death.
"""
import sys, glob, json, collections
CREV=set(range(33,37))
def scan(pats,label):
    n=jumped=fatal=fatal_j=0; seen=set()
    for pat in pats:
        for f in glob.glob(pat):
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
                prev=None
                for i,x in enumerate(fr):
                    pt=int(x.get('path_type') or 0)
                    lvl=int(x.get('path_level') or 0); nl=int(x.get('next2_path_level') or 0)
                    down = pt in CREV and nl>0 and lvl>0 and nl<lvl      # descending level change
                    if down and prev!='d':
                        n+=1
                        win=range(max(1,i-4),min(len(fr),i+5))
                        j=any(fr[k2].get('is_jumping') and not fr[k2-1].get('is_jumping') for k2 in win)
                        if j: jumped+=1
                        if di is not None and 0<=di-i<=20:
                            fatal+=1
                            if j: fatal_j+=1
                    prev='d' if down else None
    print("%-22s step-downs %3d | JUMPED %3d = %3.0f%% | deaths within 2s %2d (of which after a jump %d)"
          % (label,n,jumped,100*jumped/max(1,n),fatal,fatal_j))
scan(['captures/w18_v120_device.jsonl','captures/w18_v120_clean.jsonl'],"v120 (champion)")
scan(['captures/w18_v126*.jsonl'],"v126")
scan(['captures/w18_v111_base.jsonl'],"v111")
scan(['captures/human_2026*.jsonl'],"HUMAN")
