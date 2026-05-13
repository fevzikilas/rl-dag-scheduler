# train.py — PPO training with periodic eval against HEFT
#
# Run: python -m rl_scheduler.train

import os, sys, random, warnings, csv, time
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm as _tqdm

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

# MaskablePPO (sb3-contrib) preferred; fallback to standard PPO
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    _USE_MASK = True
    print("[train] MaskablePPO (sb3-contrib) enabled.")
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    _USE_MASK = False
    print("[train] sb3-contrib not found — using standard PPO (no action masking).")

import config
from data_loader import load_all_graphs
from hpc_env     import HPCClusterEnv
from gnn_policy  import NodeEncoder, HPCFeaturesExtractor
from baselines   import heft, round_robin


def _wrap(env: HPCClusterEnv) -> HPCClusterEnv:
    if _USE_MASK:
        return ActionMasker(env, lambda e: e.get_action_mask())
    return env


def make_env(graphs, encoder, device, seed_offset=0,
             gpu_memory=None, n_streams=None):
    def _init():
        env = HPCClusterEnv(graphs, encoder, device=device,
                            gpu_memory=gpu_memory, n_streams=n_streams)
        return _wrap(env)
    return _init


def run_agent_on_graph(model, g, encoder, device,
                       gpu_memory=None, n_streams=None) -> float:
    """Run the trained agent on a single graph; return makespan."""
    env  = _wrap(HPCClusterEnv([g], encoder, device=device,
                               gpu_memory=gpu_memory, n_streams=n_streams))
    obs, _ = env.reset()
    done   = False
    info   = {}
    while not done:
        if _USE_MASK:
            mask           = env.action_masks()
            action, _state = model.predict(obs, action_masks=mask, deterministic=True)
        else:
            action, _state = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info.get("makespan", float("inf"))


def eval_agent(model, graphs, encoder, device, n_ep: int = 20,
               gpu_memory=None, n_streams=None) -> float:
    sample = random.sample(graphs, min(n_ep, len(graphs)))
    ms     = [run_agent_on_graph(model, g, encoder, device, gpu_memory, n_streams)
              for g in sample]
    return float(np.mean(ms))


def eval_baselines(graphs, n_ep: int = 20):
    sample   = random.sample(graphs, min(n_ep, len(graphs)))
    gpu_mem  = config.GPU_MEMORY_DEFAULT
    n_s      = config.N_STREAMS_DEFAULT
    heft_ms  = [heft(g, gpu_memory=gpu_mem, n_streams=n_s)["makespan"] for g in sample]
    rr_ms    = [round_robin(g, gpu_memory=gpu_mem, n_streams=n_s)["makespan"] for g in sample]
    return float(np.mean(heft_ms)), float(np.mean(rr_ms))


class SchedulingEvalCallback(BaseCallback):
    """
    Evaluates RL vs HEFT at fixed intervals using the same hardware settings.
    Saves best checkpoint by RL/HEFT ratio. Logs each eval to CSV.
    """

    def __init__(self, test_graphs, encoder, device, heft_mean,
                 eval_freq: int, n_eval_ep: int, checkpoint_dir: str,
                 log_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.test_graphs    = test_graphs
        self.encoder        = encoder
        self.device         = device
        self.heft_mean      = heft_mean
        self.eval_freq      = eval_freq
        self.n_eval_ep      = n_eval_ep
        self.checkpoint_dir = checkpoint_dir
        self.log_path       = log_path
        self.best_ratio     = float("inf")
        self._calls_since   = 0
        # Write CSV header
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "rl_makespan", "heft_makespan", "ratio",
                 "ep_rew_mean", "wall_time_s"]
            )
        self._train_start = time.time()

    def _on_step(self) -> bool:
        self._calls_since += 1
        if self._calls_since < self.eval_freq:
            return True
        self._calls_since = 0

        # fixed hardware so RL and HEFT are compared on equal footing
        gpu_mem = config.GPU_MEMORY_DEFAULT
        n_s     = config.N_STREAMS_DEFAULT

        ms    = eval_agent(self.model, self.test_graphs, self.encoder,
                           self.device, self.n_eval_ep,
                           gpu_memory=gpu_mem, n_streams=n_s)
        ratio = ms / max(self.heft_mean, 1e-9)
        tag   = "BETTER" if ratio < 1.0 else "WORSE"

        # ep_rew_mean from SB3 logger (may be absent early on)
        ep_rew = self.logger.name_to_value.get("rollout/ep_rew_mean", float("nan"))
        wall   = time.time() - self._train_start

        if self.verbose:
            _tqdm.write(
                f"\n[Step {self.num_timesteps:>9,}]  "
                f"RL makespan: {ms:.4f}  |  HEFT: {self.heft_mean:.4f}  "
                f"ratio: {ratio:.3f}  ({tag} than HEFT)"
            )

        if ratio < self.best_ratio:
            self.best_ratio = ratio
            path = os.path.join(self.checkpoint_dir, "best_model")
            self.model.save(path)
            if self.verbose:
                _tqdm.write(f"  → New best (ratio={ratio:.4f}) saved to '{path}'")

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [self.num_timesteps, f"{ms:.6f}", f"{self.heft_mean:.6f}",
                 f"{ratio:.4f}", f"{ep_rew:.4f}", f"{wall:.0f}"]
            )

        return True


def main():
    print("=" * 65)
    print("  HPC Task Scheduling — RL Agent (PPO + GAT)")
    print("=" * 65)

    # Reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── 1. Data ───────────────────────────────────────────────────────
    train_graphs, test_graphs = load_all_graphs(seed=config.SEED)

    # ── 2. GNN Encoder (frozen during PPO; weights from random init) ──
    #    For a better start, consider pre-training on a node-property
    #    regression task before running PPO.
    encoder = NodeEncoder().to(device)
    encoder.eval()   # always eval mode (no gradient needed here)
    n_enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"NodeEncoder parameters: {n_enc_params:,}")

    # ── 3. Vectorised Environment ─────────────────────────────────────
    N_ENVS = 1     # increase if you have multiple CPUs and want parallelism
    print(f"Creating {N_ENVS} environment(s) …")
    vec_env = VecMonitor(
        DummyVecEnv([make_env(train_graphs, encoder, device, i) for i in range(N_ENVS)])
    )


    policy_kwargs = dict(
        features_extractor_class  = HPCFeaturesExtractor,
        features_extractor_kwargs = dict(features_dim=256),
        net_arch = dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate  = config.PPO_LR,
        n_steps        = config.PPO_N_STEPS,
        batch_size     = config.PPO_BATCH_SIZE,
        n_epochs       = config.PPO_N_EPOCHS,
        gamma          = config.PPO_GAMMA,
        gae_lambda     = config.PPO_GAE_LAMBDA,
        clip_range     = config.PPO_CLIP,
        ent_coef       = config.PPO_ENT_COEF,
        policy_kwargs  = policy_kwargs,
        verbose        = 1,
        seed           = config.SEED,
        device         = device,
    )
    n_policy_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy parameters: {n_policy_params:,}\n")


    print("Computing baselines on test set …")
    heft_mean, rr_mean = eval_baselines(test_graphs, n_ep=min(20, len(test_graphs)))
    print(f"  HEFT mean makespan       : {heft_mean:.4f}")
    print(f"  Round Robin mean makespan: {rr_mean:.4f}\n")


    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    log_path = os.path.join(config.CHECKPOINT_DIR, "eval_log.csv")

    callback = SchedulingEvalCallback(
        test_graphs    = test_graphs,
        encoder        = encoder,
        device         = device,
        heft_mean      = heft_mean,
        eval_freq      = config.EVAL_FREQ,
        n_eval_ep      = config.N_EVAL_EPISODES,
        checkpoint_dir = config.CHECKPOINT_DIR,
        log_path       = log_path,
        verbose        = 1,
    )

    print(f"Training for {config.TOTAL_TIMESTEPS:,} timesteps …")
    model.learn(
        total_timesteps = config.TOTAL_TIMESTEPS,
        callback        = callback,
        log_interval    = config.LOG_INTERVAL,
        progress_bar    = True,
    )

    model.save(os.path.join(config.CHECKPOINT_DIR, "final_model"))

    # final eval on fixed hardware
    gpu_mem  = config.GPU_MEMORY_DEFAULT
    n_s      = config.N_STREAMS_DEFAULT
    final_ms = eval_agent(model, test_graphs, encoder, device, n_ep=50,
                          gpu_memory=gpu_mem, n_streams=n_s)
    heft_final, rr_final = eval_baselines(test_graphs, n_ep=min(50, len(test_graphs)))
    print("\n" + "=" * 65)
    print("  FINAL RESULTS (test set, fixed: 8 GB / 4 streams)")
    print("=" * 65)
    print(f"  RL Agent makespan       : {final_ms:.4f}")
    print(f"  HEFT makespan           : {heft_final:.4f}")
    print(f"  Round Robin makespan    : {rr_final:.4f}")
    ratio_final = final_ms / max(heft_final, 1e-9)
    print(f"  RL / HEFT ratio         : {ratio_final:.3f}")
    best_ratio  = callback.best_ratio
    print(f"  Best eval ratio (saved) : {best_ratio:.4f}")
    print(f"  CSV log                 : {log_path}")
    if ratio_final < 1.0:
        print("  ✓ RL OUTPERFORMS HEFT!")
    else:
        print("  ✗ RL does not beat HEFT yet — see eval_log.csv for learning curve.")
    print("=" * 65)


if __name__ == "__main__":
    main()
