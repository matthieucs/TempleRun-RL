"""Does the drill actually make the cue FIRE at takeoff? The horizon obs died because its column was
starved -- a zero-init column only develops if the situation it describes occurs often enough. This
measures the precondition BEFORE spending the arm: rate of slide_landing_hazard==1 at real takeoffs,
under the policy that will train, in each candidate world."""
import sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "cpp_sim")
from templerun_rl import obs as O
from cpp_vec_env import CppTempleRunVecEnv
from stable_baselines3 import PPO

IDX = O.OBS_FIELDS.index("slide_landing_hazard")
model = PPO.load("runs/ppo_v111_obs66_slh.zip", device="cpu")   # == v111, on obs66

for name in ("_v108_cfg.json", "_v117_cfg.json"):
    cfg = json.load(open("scratchpad/" + name))
    cfg.update(dict(domain_rand=False, turn_approach_buffer=True, turn_linger_frames=3,
                    curriculum_max_start_dist=5000.0))
    env = CppTempleRunVecEnv(num_envs=128, config=cfg, seed=3, max_episode_steps=8000,
                             action_history_len=4, frame_stack=1)
    assert env.cfg["curriculum_max_start_dist"] == 5000.0
    o = env.reset()
    prev_air = np.zeros(128, bool)
    takeoffs = fired = 0
    for _ in range(2500):
        a, _ = model.predict(np.asarray(o), deterministic=True)
        col = np.asarray(o)[:, IDX]
        air = np.asarray([env.raw_dict(e)["is_jumping"] for e in range(128)], bool)
        tk = (np.asarray(a).reshape(-1) == 2) & ~air
        takeoffs += int(tk.sum())
        fired += int((tk & (col == 1.0)).sum())
        o, _, _, _ = env.step(np.asarray(a).reshape(-1))
    env.close()
    print("%-18s takeoffs %6d   cue==1 at takeoff %5d  = %.2f%%"
          % (name, takeoffs, fired, 100 * fired / max(1, takeoffs)))
print("\ndevice reference: 51 / 26854 takeoffs = 0.19%")
