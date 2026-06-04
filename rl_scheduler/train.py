"""Train PPO agent on DAG scheduling task."""

import os, sys, random, warnings, csv, time
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm as _tqdm

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    _USE_MASK = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    _USE_MASK = False

import config
from data_loader import load_all_graphs
from hpc_env     import HPCClusterEnv
from gnn_policy  import HPCFeaturesExtractor
from baselines   import heft, round_robin


def _wrap(env: HPCClusterEnv) -> HPCClusterEnv:
    if _USE_MASK:
        return ActionMasker(env, lambda e: e.get_action_mask())
    return env


def make_env(graphs, device, seed_offset=0, gpu_memory=None, n_streams=None):
    def _init():
        env = HPCClusterEnv(graphs, device=device,
                            gpu_memory=gpu_memory, n_streams=n_streams)
        return _wrap(env)
    return _init


def run_agent_on_graph(model, g, device, gpu_memory=None, n_streams=None) -> tuple:
    """Run agent on one graph, return RL makespan."""
    env = _wrap(HPCClusterEnv([g], device=device,
                              gpu_memory=gpu_memory, n_streams=n_streams))
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        if _USE_MASK:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info.get("makespan", float("inf"))


def eval_agent(model, graphs, device, gpu_memory=None, n_streams=None) -> float:
    """Run agent on all graphs, return mean makespan."""
    makespans = [run_agent_on_graph(model, g, device, gpu_memory, n_streams)
                 for g in graphs]
    return float(np.mean(makespans))


def eval_baselines(graphs):
    gpu_mem  = config.GPU_MEMORY_DEFAULT
    n_s      = config.N_STREAMS_DEFAULT
    heft_ms  = [heft(g, gpu_memory=gpu_mem, n_streams=n_s)["makespan"] for g in graphs] 
    rr_ms    = [round_robin(g, gpu_memory=gpu_mem, n_streams=n_s)["makespan"] for g in graphs]
    return float(np.mean(heft_ms)), float(np.mean(rr_ms))


class SchedulingEvalCallback(BaseCallback):
    """Evaluate RL vs HEFT every N steps, save best checkpoint."""
    def __init__(self, test_graphs, device, eval_freq: int, n_eval_ep: int,
                 checkpoint_dir: str, log_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.test_graphs = test_graphs
        self.device = device
        self.eval_freq = eval_freq
        self.checkpoint_dir = checkpoint_dir
        self.log_path = log_path
        self.best_ratio = float("inf")
        self._calls_since = 0

        rng = random.Random(config.SEED)
        self.eval_graphs = rng.sample(test_graphs, min(n_eval_ep, len(test_graphs)))
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "rl_makespan", "heft_makespan", "ratio"])
        self._train_start = time.time()

    def _on_step(self) -> bool:
        self._calls_since += 1
        if self._calls_since < self.eval_freq:
            return True
        self._calls_since = 0

        gpu_mem = config.GPU_MEMORY_DEFAULT
        n_s = config.N_STREAMS_DEFAULT

        rl_ms = eval_agent(self.model, self.eval_graphs, self.device,
                           gpu_memory=gpu_mem, n_streams=n_s)
        heft_ms = eval_baselines(self.eval_graphs)[0]
        ratio = rl_ms / max(heft_ms, 1e-9)

        if self.verbose:
            tag = "BETTER" if ratio < 1.0 else "WORSE"
            _tqdm.write(
                f"\n[Step {self.num_timesteps:>9,}] "
                f"RL: {rl_ms:.4f}  HEFT: {heft_ms:.4f}  "
                f"ratio: {ratio:.3f} ({tag})"
            )

        if ratio < self.best_ratio:
            self.best_ratio = ratio
            path = os.path.join(self.checkpoint_dir, "best_model")
            self.model.save(path)
            if self.verbose:
                _tqdm.write(f"  → Best ratio {ratio:.4f}, saved to {path}")

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, f"{rl_ms:.6f}",
                                    f"{heft_ms:.6f}", f"{ratio:.4f}"])
        return True


def main():
    print("=" * 65)
    print("  Task Scheduling — PPO Agent")
    print("=" * 65)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    train_graphs, test_graphs = load_all_graphs(seed=config.SEED)

    N_ENVS = 1
    print(f"Creating {N_ENVS} environment(s) …")
    vec_env = VecMonitor(
        DummyVecEnv([make_env(train_graphs, device, i) for i in range(N_ENVS)])
    )


    policy_kwargs = dict(
        features_extractor_class  = HPCFeaturesExtractor,
        features_extractor_kwargs = dict(features_dim=256),
        net_arch = dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate  = get_linear_fn(config.PPO_LR, config.PPO_LR * 0.1, 1.0),
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

    # Load BC pretrained weights if available (run pretrain.py first)
    bc_path = os.path.join(config.CHECKPOINT_DIR, "bc_pretrain.zip")
    if os.path.exists(bc_path):
        print(f"Loading BC pretrained weights from '{bc_path}' …")
        bc_model = MaskablePPO.load(bc_path, device=device)
        model.policy.load_state_dict(bc_model.policy.state_dict())
        del bc_model
        print("  BC weights loaded — starting PPO from HEFT-imitating policy.\n")
    else:
        print("No bc_pretrain.zip found — starting PPO from scratch.")
        print("  Tip: run 'python -m rl_scheduler.pretrain' first for faster convergence.\n")


    print("Computing baselines on test set …")
    heft_mean, rr_mean = eval_baselines(test_graphs)
    print(f"  HEFT mean makespan       : {heft_mean:.4f}")
    print(f"  Round Robin mean makespan: {rr_mean:.4f}\n")


    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    log_path = os.path.join(config.CHECKPOINT_DIR, "eval_log.csv")

    callback = SchedulingEvalCallback(
        test_graphs    = test_graphs,
        device         = device,
        eval_freq      = config.EVAL_FREQ,
        n_eval_ep      = config.N_EVAL_EPISODES,
        checkpoint_dir = config.CHECKPOINT_DIR,
        log_path       = log_path,
        verbose        = 1,
    )

    print(f"Training for {config.TOTAL_TIMESTEPS:,} timesteps …")
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=callback,
        log_interval=config.LOG_INTERVAL,
        progress_bar=True,
    )

    model.save(os.path.join(config.CHECKPOINT_DIR, "final_model"))

    gpu_mem = config.GPU_MEMORY_DEFAULT
    n_s = config.N_STREAMS_DEFAULT
    final_rl = eval_agent(model, test_graphs, device, gpu_memory=gpu_mem, n_streams=n_s)
    final_heft, final_rr = eval_baselines(test_graphs)
    ratio = final_rl / max(final_heft, 1e-9)

    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    print(f"  RL makespan:     {final_rl:.4f}")
    print(f"  HEFT makespan:   {final_heft:.4f}")
    print(f"  RoundRobin:      {final_rr:.4f}")
    print(f"  RL / HEFT ratio: {ratio:.3f}")
    print(f"  Best ratio:      {callback.best_ratio:.4f}")
    print(f"  Log file:        {log_path}")
    if ratio < 1.0:
        print("  ✓ RL beats HEFT!")
    else:
        print("  ✗ RL does not beat HEFT on average.")
    print("=" * 65)


if __name__ == "__main__":
    main()
