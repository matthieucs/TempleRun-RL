"""Are the SHORT-WARNING hole crossings the ones just after a JUNCTION?

obs.py forces next2/next3_must_jump False if any cell on the way is a turn (the documented junction
dropout), so a hole shortly after a corner is structurally invisible until the runner is on it.
"""
import sys, glob, json, collections, statistics as st
HOLE=set(range(11,17))
def scan(pats,label):
    grp={'after_turn':[], 'clear':[]}; fatal={'after_turn':0,'clear':0}; n={'after_turn':0,'clear':0}
    seen=set()
    for pat in pats:
        for f in glob.glob(pat):
            rs=[json.loads(l) for l in open(f) if l.strip()]
            eps=collections.defaultdict(list)
            for r in rs:
                if 'velocity' in r: eps[r.get('ep')].append(r)
            for e,fr in eps.items():
                di=next((i for i,x in enumerate(fr) if x.get('death_type')),None)
                if di is not None:
                    k=round(float(fr[di].get('distance') or 0),3)
                    if k in seen or k<200: continue
                    seen.add(k)
                prev_pt=None
                for i,x in enumerate(fr):
                    pt=int(x.get('path_type') or 0)
                    if pt in HOLE and prev_pt not in HOLE:
                        # was there a junction in the 12 frames before this hole?
                        turn = any(fr[j].get('is_turn') for j in range(max(0,i-12), i))
                        g='after_turn' if turn else 'clear'
                        w=0
                        for j in range(i-1, max(-1,i-13), -1):
                            if fr[j].get('next_must_jump') or fr[j].get('next2_must_jump') or fr[j].get('next3_must_jump'): w+=1
                            else: break
                        grp[g].append(w); n[g]+=1
                        if di is not None and 0 <= di-i <= 3: fatal[g]+=1
                    prev_pt=pt
    print("\n%s" % label)
    for g in ('clear','after_turn'):
        if not grp[g]: continue
        print("   %-11s n=%4d | warning median %.1f (p25 %.0f) | deaths %2d = %.2f%% per crossing"
              % (g,n[g],st.median(grp[g]),sorted(grp[g])[len(grp[g])//4],fatal[g],100*fatal[g]/max(1,n[g])))
scan(['captures/w18_v120_device.jsonl','captures/w18_v120_device_b.jsonl','captures/w18_v120_clean.jsonl'],"v120 device")
scan(['captures/w18_v111_base.jsonl'],"v111 device")
scan(['captures/human_2026*.jsonl'],"HUMAN")
