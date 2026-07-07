"""Measure the sim's crevasse-cap graze rate BY LEVEL with crevasse_graze_worldy on, driving a real
policy. Target (device, hook 16y): ~32% graze @level-3 caps, ~0% @level-2. Instruments _stumble/_kill
on crevasse apex cells."""
import sys; sys.path.insert(0,'src')
import numpy as np, templerun_rl
from collections import defaultdict
from stable_baselines3 import PPO
from templerun_rl.sim_env import TempleRunSimEnv, _MUST_JUMP, OBST_LEVEL_CHANGE
import templerun_rl.obs as O

CFG = {"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":True,"jump_drop_prob":0.1,
       "compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,
       "stepup_clip_band":0.25,"wide_gap_landing_model":True,"wide_gap_land_grace":0.35,
       "crevasse_graze_worldy":True,"turn_approach_buffer":True,"turn_linger_frames":3,
       "curriculum_max_start_dist":0.0}

def run(model_path, n_eps=50, seed0=1000):
    m = PPO.load(model_path, device="cpu")
    K = max(0,(int(m.observation_space.shape[0])-O.OBS_LEN)//7)
    env = TempleRunSimEnv(config=CFG, max_episode_steps=4000, reward_mode="distance",
                          action_history_len=K, frame_stack=1)
    counts = defaultdict(lambda: {"clear":0,"graze":0,"kill":0})
    orig = env._resolve_obstacle_on_enter
    def patched(cell):
        obst=int(cell.get("obstacles",0)); lvl=int(cell.get("level",2))
        # is this a crevasse apex (last-crev) cell?
        nxt = env._track[env._cell_index+1] if env._cell_index+1<len(env._track) else {}
        is_cap = (obst==OBST_LEVEL_CHANGE and cell.get("wide_gap") and not nxt.get("wide_gap")) or bool(cell.get("stepup"))
        dead0=env._is_dead; st0=env._stumble_t
        orig(cell)
        if is_cap:
            if env._is_dead and not dead0: counts[lvl]["kill"]+=1
            elif env._stumble_t>st0: counts[lvl]["graze"]+=1
            else: counts[lvl]["clear"]+=1
    env._resolve_obstacle_on_enter = patched
    for ep in range(n_eps):
        o,i = env.reset(seed=seed0+ep); import numpy as _np
        frames=[O.state_to_obs(i["raw_state"])]; hist=[]
        def obsv():
            base=O.state_to_obs(env._snapshot())
            if K: 
                import numpy as np2
                h=[int(a) for a in hist[-K:]]; h=[6]*(K-len(h))+h
                oh=np2.zeros(K*7,dtype=np2.float32)
                for j,a in enumerate(h): oh[j*7+a]=1.0
                return np2.concatenate([base,oh])
            return base
        done=False; steps=0
        obs=obsv()
        while not done and steps<4000:
            a,_=m.predict(obs,deterministic=True); a=int(a)
            o,r,t,tr,info=env.step(a); hist.append(a); obs=obsv()
            done=t or tr; steps+=1
    print(f"\n{model_path}  ({n_eps} eps)")
    print(f"  {'lvl':>3} {'n':>5} {'clear':>6} {'graze':>6} {'kill':>5} {'graze%':>7} {'kill%':>6}")
    for lvl in sorted(counts):
        c=counts[lvl]; n=c["clear"]+c["graze"]+c["kill"]
        if n==0: continue
        print(f"  {lvl:>3} {n:>5} {c['clear']:>6} {c['graze']:>6} {c['kill']:>5} {100*c['graze']/n:>6.0f}% {100*c['kill']/n:>5.0f}%")

for mp in sys.argv[1:] or ["runs/ppo_v51_next2_best.zip"]:
    run(mp)
