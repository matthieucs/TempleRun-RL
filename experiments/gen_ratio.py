"""Hazard GENERATION ratio, sim worlds vs device. An encounter COUNT is immune to both confounds
that wrecked the lethality comparison (death attribution, policy strength) -- a hazard is generated
whether or not anyone dies on it."""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, json
sys.path.insert(0,"src"); sys.path.insert(0,"cpp_sim")
import numpy as np
from stable_baselines3 import PPO
from cpp_vec_env import CppTempleRunVecEnv
def kind(pt):
    pt=int(pt or 0)
    if 31<=pt<=32: return "flame"
    if 33<=pt<=36: return "crevasse"
    if 11<=pt<=16: return "hole"
    return None
model = PPO.load("runs/ppo_v111_obs66_slh.zip", device="cpu")
for name in ("_v108_cfg.json","_v114_cfg.json"):
    cfg=json.load(open("experiments/"+name))
    cfg.update(dict(domain_rand=False,turn_approach_buffer=True,turn_linger_frames=3,
                    curriculum_max_start_dist=0.0))
    enc={}
    NE=96
    for sd in range(3):
        env=CppTempleRunVecEnv(num_envs=NE,config=cfg,seed=sd,max_episode_steps=8000,
                               action_history_len=4,frame_stack=1)
        assert env.cfg["curriculum_max_start_dist"]==0.0
        o=env.reset(); seen=np.zeros(NE,bool); prev=[None]*NE
        for _ in range(8005):
            for i in range(NE):
                if seen[i]: continue
                k=kind(env.raw_dict(i).get("path_type"))
                if k!=prev[i]:
                    if k: enc[k]=enc.get(k,0)+1
                    prev[i]=k
            a,_=model.predict(np.asarray(o),deterministic=True)
            o,_,dn,_=env.step(np.asarray(a).reshape(-1))
            for i in np.nonzero(dn)[0]: seen[i]=True
            if seen.all(): break
        env.close()
    t=sum(enc.values())
    print("%-16s %s   flame:hole = %.2f" % (name,
          "  ".join("%s %5d (%4.1f%%)"%(k,enc.get(k,0),100*enc.get(k,0)/max(1,t)) for k in ("crevasse","flame","hole")),
          enc.get("flame",0)/max(1,enc.get("hole",1))))
print("%-16s %s   flame:hole = %.2f" % ("DEVICE v107+v111",
      "  crevasse   203 (24.0%)  flame   228 (27.1%)  hole   410 (48.8%)", 228/410))
