"""WAVE 18 surgery: v111 (obs65 = 37 base + 28 hist, 512x512) -> obs66 by inserting ONE zero column
for `slide_landing_hazard` at FULL-vector index 37 (last base field, before the action history).
Zero-init -> the base policy's behaviour is preserved EXACTLY; the retrain learns the new cue.

WHY: `landing_hazard` (obs65, the cue that made v111 champion) asks "does my arc come down on a
must-JUMP cell". It says nothing about the hazard you CANNOT answer once airborne. You cannot slide
in mid-air, so for a must-SLIDE hazard the entire counterfactual lives at TAKEOFF -- by the time it
is in front of you the only saving action is already unavailable.

MEASURED, 100 device events (every takeoff whose predicted arc spans a must-slide hazard, 45 captures):

                   n    takeoff cp   span   hazard at   ARC SLACK   pressed slide
    survived      73       0.53      2.61      +2.0       +1.54          34%
    died          27       0.62      2.69      +3.0       +0.21          89%

arc_slack = (cell_progress + span) - hazard_offset. Survivors fly over the hazard and land beyond it;
victims' arcs END ON IT. 89% of the victims pressed SLIDE and were still in the air -- they read the
situation correctly and died anyway. Those takeoffs are 100% FORCED (a must_jump demanded them, zero
discretionary), so this is not a decision error, it is a TIMING error: 0.53 vs 0.62 takeoff cp is the
whole difference. ~33% of all deaths have this shape (88% of Flame deaths are airborne).

`int(landing) == hazard_offset` is exactly that discriminator (slack < 1 vs >= 1), so the new column
is the same one-scalar collapse that produced predicted_landing, predicted_jump and landing_hazard --
applied to the must_slide ladder. Host-derived: NO hook change (17j/17k both fine).
"""
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, templerun_rl, torch, numpy as np  # noqa: F401
from stable_baselines3 import PPO
from templerun_rl import obs as O

OLD_PATH = 'runs/ppo_v111_lh030_final.zip'
NEW_PATH = 'runs/ppo_v111_obs66_slh.zip'
INSERT_AT = O.OBS_LEN - 1          # last base field, before the action history
OLD_W, NEW_W = O.OBS_LEN - 1 + 28, O.OBS_LEN + 28

assert O.OBS_FIELDS[INSERT_AT] == 'slide_landing_hazard', O.OBS_FIELDS[INSERT_AT]

old = PPO.load(OLD_PATH, device='cpu')
env = gym.make('TempleRunSim-v0', action_history_len=4, frame_stack=1).unwrapped
new = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[512, 512]))
print('old obs', old.observation_space.shape, '-> new obs', new.observation_space.shape)
assert old.observation_space.shape[0] == OLD_W, old.observation_space.shape
assert new.observation_space.shape[0] == NEW_W, new.observation_space.shape

old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
for k in new_sd:
    if k not in old_sd:
        print('NEW-ONLY', k); continue
    ov, nv = old_sd[k], new_sd[k]
    if ov.shape == nv.shape:
        new_sd[k] = ov.clone()
    elif ov.dim() == 2 and ov.shape[0] == nv.shape[0] and ov.shape[1] == OLD_W and nv.shape[1] == NEW_W:
        w = torch.zeros_like(nv)
        w[:, :INSERT_AT] = ov[:, :INSERT_AT]
        w[:, INSERT_AT + 1:] = ov[:, INSERT_AT:]
        new_sd[k] = w
        print('SURGERY', k, tuple(ov.shape), '->', tuple(nv.shape), '(zero col @%d)' % INSERT_AT)
    else:
        print('!!SKIP shape-mismatch', k, tuple(ov.shape), tuple(nv.shape))
new.policy.load_state_dict(new_sd)

# The new column must be a DEAD INPUT at init: feeding it any value must not change a single action.
rng = np.random.default_rng(0); mism = 0
for _ in range(400):
    o_old = rng.random(OLD_W).astype(np.float32)
    o_new = np.insert(o_old, INSERT_AT, rng.random()).astype(np.float32)
    if int(old.predict(o_old, deterministic=True)[0]) != int(new.predict(o_new, deterministic=True)[0]):
        mism += 1
print(f'equivalence mismatches (want 0): {mism} /400')
assert mism == 0, 'surgery is NOT behaviour-preserving -- do not train on this base'
new.save(NEW_PATH)
print('saved', NEW_PATH)
