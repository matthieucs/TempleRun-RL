"""MODE B (hole, jumped late or never): is it a PERCEPTION gap or a DECISION gap?

For every hole crossing, measure how many frames of WARNING the policy had (next_must_jump /
next2 / next3 true before reaching the hole cell) and when it actually took off. If fatal crossings
had as much warning as survived ones, the cue is adequate and the fix is in the recipe; if they had
LESS, the policy is being told too late and the fix is an observation.
"""
import sys, glob, json, collections
sys.path.insert(0,"src")
HOLE=set(range(11,17))

def scan(pats,label):
    warn={'died':[], 'lived':[]}; lead={'died':[], 'lived':[]}; seen=set()
    for pat in pats:
        for f in glob.glob(pat):
            rs=[json.loads(l) for l in open(f) if l.strip()]
            eps=collections.defaultdict(list)
            for r in rs:
                if 'velocity' in r: eps[r.get('ep')].append(r)
            for e,fr in eps.items():
                di=next((i for i,x in enumerate(fr) if x.get('death_type')),None)
                key=round(float(fr[di].get('distance') or 0),3) if di is not None else None
                if key is not None:
                    if key in seen or key<200: continue
                    seen.add(key)
                # find hole-cell entries
                prev_pt=None
                for i,x in enumerate(fr):
                    pt=int(x.get('path_type') or 0)
                    if pt in HOLE and prev_pt not in HOLE:
                        fatal = di is not None and 0 <= di-i <= 3
                        k='died' if fatal else 'lived'
                        # frames of warning: consecutive frames before i with any must-jump flag
                        w=0
                        for j in range(i-1, max(-1,i-13), -1):
                            if fr[j].get('next_must_jump') or fr[j].get('next2_must_jump') or fr[j].get('next3_must_jump'):
                                w+=1
                            else: break
                        warn[k].append(w)
                        # takeoff lead: frames between last jump onset and the hole entry
                        on=[j for j in range(max(1,i-12),i+1)
                            if fr[j].get('is_jumping') and not fr[j-1].get('is_jumping')]
                        lead[k].append(i-on[-1] if on else -1)
                    prev_pt=pt
    import statistics as st
    print("\n%s" % label)
    for k in ('lived','died'):
        if not warn[k]: continue
        nolead=sum(1 for x in lead[k] if x<0)
        print("   %-6s n=%4d | warning frames median %.1f (p25 %.0f) | takeoff lead median %.1f | NEVER jumped %d = %.0f%%"
              % (k,len(warn[k]),st.median(warn[k]),
                 sorted(warn[k])[len(warn[k])//4], st.median([x for x in lead[k] if x>=0] or [-1]),
                 nolead,100*nolead/len(lead[k])))

scan(['captures/w18_v120_device.jsonl','captures/w18_v120_device_b.jsonl','captures/w18_v120_clean.jsonl'],"v120 device")
scan(['captures/human_2026*.jsonl'],"HUMAN")
