import sys, json, numpy as np
sys.path.insert(0,'src')
import gymnasium as gym, templerun_rl

# Load v14 capture; for each junction TL/TR transition, set sim state to the device pre-state,
# apply the turn, and check sim dead vs device dead (within a few frames).
rows=[]
for line in open('captures/v14_16i.jsonl'):
    try: rows.append(json.loads(line))
    except: pass
by_ep={}
for r in rows: by_ep.setdefault(r.get('ep'),[]).append(r)

def dev_died_soon(fr,i,w=6):
    return any(fr[j].get('is_dead') for j in range(i,min(i+w+1,len(fr))))

for carry in [3.0,2.0,1.0,0.5,0.25,0.0]:
    env=gym.make('TempleRunSim-v0',config=dict(turn_lane_carry=carry)).unwrapped
    agree=0; tot=0; land_errs=[]
    for ep,fr in by_ep.items():
        for i,d in enumerate(fr):
            if d.get('action') in (4,5) and d.get('is_turn'):
                tot+=1
                env.reset()
                # inject device pre-state
                env._lane=float(d.get('lane',0)); env._velocity=float(d.get('velocity',100))
                env._facing=int(d.get('facing',1) or 1)
                want_left = d.get('action')==4
                cp=float(d.get('cell_progress',0))
                cell=dict(is_turn=True,can_turn_left=True,can_turn_right=True)
                env._execute_turn(cell,want_left,cp)
                sim_dead = env._is_dead
                dev_dead = dev_died_soon(fr,i)
                if sim_dead==dev_dead: agree+=1
                # landing error vs device landing (max|lane| next 6)
                dev_land=max((abs(fr[j].get('lane',0)) for j in range(i+1,min(i+7,len(fr)))),default=abs(d.get('lane',0)))
                sim_land=abs(env._lane) if not sim_dead else 99
                if not sim_dead and not dev_dead: land_errs.append(abs(sim_land-dev_land))
    le=np.median(land_errs) if land_errs else float('nan')
    print('carry=%.2f  junction dead-agree=%3.0f%% (%d/%d)  land_err(both-live)=%.1f'%(carry,100*agree/max(1,tot),agree,tot,le))
