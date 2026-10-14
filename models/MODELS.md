# Shipped policy

| Policy | Device runs | Median | Best |
|---|:---:|:---:|:---:|
| `templerun_runner_v1.zip` | 14 | 4,736 m | 16,782 m |

- Stable-Baselines3 PPO, MLP `[512, 512]`, observation width 67 (`src/templerun_rl/obs.py`).
- Trained in the simulator (`cpp_sim/`), evaluated on a physical phone (Redmi, Android 15,
  Temple Run v1.39.0) through the Frida hook `frida/temple_run_hook.js`.
- The median comes from a 14-run benchmark; the best run comes from a separate 20-attempt recording
  session. Benchmark episodes were capped at 4,000 frames, which cut off one run that was still
  alive past 15 km, so the median slightly understates the policy.

Run it on the phone:

```bash
python scripts/device/capture_calib.py models/templerun_runner_v1.zip --episodes 5 --turn-cooldown 0 --max-steps 8000
```

Fine-tune it in the simulator: `bash configs/train.sh`.
