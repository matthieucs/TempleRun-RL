#!/usr/bin/env python3
"""WHERE DOES THE TRAINING WALL-CLOCK GO? sim step vs PPO update.

USER 2026-07-27, seeing the GPU at ~12%: "I'd really like it to be faster and leverage more of my
gpu." Measured cause: the trainer pegs exactly ONE core (100% of 800%) while the GPU idles — it is
launch-bound, not compute-bound. The minibatch update is ALREADY CUDA-graphed; the env step is not.
This splits the per-iteration cost so the ceiling of graphing the sim is known BEFORE doing the work
(Amdahl: if the sim is 40% of the time, graphing it perfectly caps the speedup at 1.7x).

    python experiments/profile_train_loop.py --num-envs 256 --n-steps 16 --batch-size 512
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
import torch  # noqa: E402
from templerun_rl.sim_env_torch import TorchTempleRunVecEnv  # noqa: E402


def timeit(fn, n, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--net", type=int, default=512)
    p.add_argument("--config", default="experiments/_v95_cfg.json")
    p.add_argument("--iters", type=int, default=60)
    a = p.parse_args()

    cfg = json.load(open(_ROOT / a.config)) if a.config else {}
    env = TorchTempleRunVecEnv(a.num_envs, config=cfg, device="cuda", seed=0,
                               max_episode_steps=8000, reward_mode="distance",
                               action_history_len=4, frame_stack=1)
    obs = env.reset()
    obs_dim = obs.shape[-1]
    act = torch.full((a.num_envs,), 6, dtype=torch.long, device="cuda")

    t_step = timeit(lambda: env.step_tensor(act), a.iters)

    # a stand-in for the policy fwd+bwd at the real geometry (512,512 MLP, batch_size rows)
    net = torch.nn.Sequential(torch.nn.Linear(obs_dim, a.net), torch.nn.Tanh(),
                              torch.nn.Linear(a.net, a.net), torch.nn.Tanh(),
                              torch.nn.Linear(a.net, 7)).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    mb = torch.randn(a.batch_size, obs_dim, device="cuda")

    def update():
        opt.zero_grad(set_to_none=True)
        net(mb).square().mean().backward()
        opt.step()

    t_upd = timeit(update, a.iters)
    # policy forward during rollout (one per env step, num_envs rows)
    ro = torch.randn(a.num_envs, obs_dim, device="cuda")
    with torch.no_grad():
        t_fwd = timeit(lambda: net(ro), a.iters)

    n_mb = max(1, (a.num_envs * a.n_steps) // a.batch_size)
    sim_total = a.n_steps * (t_step + t_fwd)
    upd_total = a.n_epochs * n_mb * t_upd
    tot = sim_total + upd_total
    steps_per_iter = a.num_envs * a.n_steps
    print(f"\ngeometry: num_envs={a.num_envs} n_steps={a.n_steps} batch={a.batch_size} "
          f"epochs={a.n_epochs} -> {n_mb} minibatches/epoch, {steps_per_iter} timesteps/iter")
    print(f"  env.step()            {1000*t_step:7.3f} ms  x{a.n_steps}")
    print(f"  policy fwd (rollout)  {1000*t_fwd:7.3f} ms  x{a.n_steps}")
    print(f"  update (fwd+bwd+adam) {1000*t_upd:7.3f} ms  x{a.n_epochs*n_mb}")
    print(f"  ---")
    print(f"  ROLLOUT total {1000*sim_total:8.1f} ms = {100*sim_total/tot:4.1f}%")
    print(f"  UPDATE  total {1000*upd_total:8.1f} ms = {100*upd_total/tot:4.1f}%")
    print(f"  iteration     {1000*tot:8.1f} ms -> {steps_per_iter/tot:,.0f} timesteps/s "
          f"({1e6*tot/steps_per_iter/60:.1f} min per 1M)")
    print(f"\n  CEILING if the sim step became FREE: {tot/(tot-a.n_steps*t_step):.2f}x")


if __name__ == "__main__":
    main()
