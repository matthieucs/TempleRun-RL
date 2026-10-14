"""Same check, restricted to the CURRENT CHAMPION's own device captures. The 176-file pool is
dominated by years of weaker policies dying early (Fall/Ledge); what matters is what kills v111."""
import sys, glob, json
sys.path.insert(0, "src")
from collections import Counter
from templerun_rl import obs as O
exec(open("scratchpad/slh_device_check.py").read().split("files = [")[0].split('"""')[2])

IDX = O.OBS_FIELDS.index("slide_landing_hazard")
files = sorted(set(glob.glob("captures/*v111*.jsonl") + glob.glob("captures/w17_v111*.jsonl")
                   + glob.glob("captures/video/*.jsonl")))
print("v111 captures:", [f.split("/")[-1] for f in files])
tot, flagged, mix, dists = 0, 0, Counter(), []
n1 = d1 = 0
for f in files:
    for fr in episodes(f):
        di = next((i for i, x in enumerate(fr) if x.get("death_type")), None)
        for i in range(1, len(fr)):
            if fr[i].get("action") != 2: continue
            p = fr[i - 1]
            if p.get("is_jumping") or p.get("is_dead"): continue
            if float(O.state_to_obs(p)[IDX]) == 1.0:
                n1 += 1
                if di is not None and 0 <= di - i <= 15 and int(fr[di]["death_type"]) in (3, 5):
                    d1 += 1
        if di is None: continue
        tot += 1
        mix[int(fr[di]["death_type"])] += 1
        dists.append(float(fr[di].get("distance") or 0))
        for i in range(max(1, di - 15), di + 1):
            if fr[i].get("action") != 2: continue
            p = fr[i - 1]
            if p.get("is_jumping") or p.get("is_dead"): continue
            if float(O.state_to_obs(p)[IDX]) == 1.0:
                flagged += 1; break
print("\nv111 device: %d deaths, distances %s" % (tot, sorted(int(d) for d in dists)))
print("death-type mix:", dict(mix.most_common()), " (3=Slide 5=Burnt 1=Fall 4=Ledge 2=Tree)")
print("cue fired on %d takeoffs, %d fatal (%.1f%%)" % (n1, d1, 100*d1/max(1,n1)))
print("CEILING: %d of %d deaths (%.1f%%) flagged by the cue one jump ahead" % (flagged, tot, 100*flagged/max(1,tot)))
