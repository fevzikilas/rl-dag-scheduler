"""Behavioral cloning: train policy to imitate HEFT decisions."""

import os, sys, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    _USE_MASK = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    _USE_MASK = False

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils   import obs_as_tensor, get_linear_fn

import config
from data_loader import load_all_graphs
from hpc_env     import HPCClusterEnv
from gnn_policy  import HPCFeaturesExtractor
from baselines   import heft


def _wrap(env: HPCClusterEnv):
    if _USE_MASK:
        return ActionMasker(env, lambda e: e.get_action_mask())
    return env


def collect_heft_dataset(graphs: list, gpu_memory: float, n_streams: int,
                         seed: int = 42):
    """Collect (observation, action) pairs by replaying HEFT decisions."""
    rng = random.Random(seed)
    shuffled = list(graphs)
    rng.shuffle(shuffled)

    all_obs: list = []
    all_act: list = []

    for g in tqdm(shuffled, desc="Collecting HEFT demos"):
        result     = heft(g, gpu_memory=gpu_memory, n_streams=n_streams)
        assignment = result["assignment"]   # {node_name: stream_id}

        # We need direct access to the inner env to read the ready queue,
        # so keep a reference to the unwrapped env.
        inner = HPCClusterEnv([g], device="cpu",
                               gpu_memory=gpu_memory, n_streams=n_streams)
        env   = _wrap(inner)

        obs, _ = env.reset()
        done   = False

        while not done:
            task_idx  = inner.ready_queue[0]
            task_node = inner.nodes[task_idx]
            action    = int(assignment.get(task_node, 0))
            action    = min(action, inner.n_streams - 1)   # safety clamp

            all_obs.append({k: v.copy() for k, v in obs.items()})
            all_act.append(action)

            obs, _r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return all_obs, all_act


def build_model(vec_env, device: str) -> MaskablePPO:
    policy_kwargs = dict(
        features_extractor_class  = HPCFeaturesExtractor,
        features_extractor_kwargs = dict(features_dim=256),
        net_arch = dict(pi=[256, 128], vf=[256, 128]),
    )
    return MaskablePPO(
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
        verbose        = 0,
        seed           = config.SEED,
        device         = device,
    )


def train_bc(model: MaskablePPO, obs_list: list, act_list: list,
             device: str, n_epochs: int = 30, batch_size: int = 256):
    """Train policy with supervised learning on HEFT demonstrations."""
    policy    = model.policy
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    n       = len(act_list)
    indices = list(range(n))
    best_loss = float("inf")

    for epoch in range(n_epochs):
        random.shuffle(indices)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]

            obs_batch = {
                k: np.stack([obs_list[i][k] for i in batch_idx], axis=0)
                for k in obs_list[0].keys()
            }
            obs_tensor = obs_as_tensor(obs_batch, device)
            act_tensor = torch.tensor([act_list[i] for i in batch_idx],
                                      dtype=torch.long, device=device)

            if _USE_MASK:
                masks = np.ones((len(batch_idx), config.MAX_STREAMS), dtype=bool)
                _, log_probs, _ = policy.evaluate_actions(
                    obs_tensor, act_tensor, action_masks=masks
                )
            else:
                _, log_probs, _ = policy.evaluate_actions(obs_tensor, act_tensor)

            loss = -log_probs.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        marker   = " *" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
        print(f"  Epoch {epoch + 1:3d}/{n_epochs}  loss: {avg_loss:.4f}{marker}")

    policy.eval()
    print(f"\nBest BC loss: {best_loss:.4f}")


def measure_agreement(model: MaskablePPO, obs_list: list, act_list: list,
                      device: str, sample: int = 2000) -> float:
    """Measure fraction of HEFT decisions the policy reproduces."""
    policy = model.policy
    policy.eval()

    rng     = random.Random(0)
    idx     = rng.sample(range(len(obs_list)), min(sample, len(obs_list)))

    obs_batch = {
        k: np.stack([obs_list[i][k] for i in idx], axis=0)
        for k in obs_list[0].keys()
    }
    obs_tensor = obs_as_tensor(obs_batch, device)

    with torch.no_grad():
        if _USE_MASK:
            masks   = np.ones((len(idx), config.MAX_STREAMS), dtype=bool)
            actions, _ = policy.predict(obs_batch, action_masks=masks,
                                        deterministic=True)
        else:
            actions, _ = policy.predict(obs_batch, deterministic=True)

    targets = np.array([act_list[i] for i in idx])
    return float((actions == targets).mean())


def main():
    print("=" * 65)
    print("  Behavioral Cloning — imitating HEFT before PPO")
    print("=" * 65)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    train_graphs, _ = load_all_graphs(seed=config.SEED)

    gpu_mem = config.GPU_MEMORY_DEFAULT
    n_s     = config.N_STREAMS_DEFAULT

    # --- Collect HEFT demonstrations ---
    print(f"Collecting HEFT demos from {len(train_graphs)} training graphs …")
    obs_list, act_list = collect_heft_dataset(train_graphs, gpu_mem, n_s,
                                               seed=config.SEED)
    print(f"  Collected {len(obs_list):,} (obs, action) pairs\n")

    # --- Build model (same arch as train.py) ---
    print("Building model …")
    vec_env = VecMonitor(DummyVecEnv([
        lambda: _wrap(HPCClusterEnv(train_graphs, device="cpu",
                                    gpu_memory=gpu_mem, n_streams=n_s))
    ]))
    model   = build_model(vec_env, device)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy parameters: {n_params:,}\n")

    # --- Behavioral cloning ---
    print("Training with behavioral cloning (30 epochs) …")
    train_bc(model, obs_list, act_list, device, n_epochs=30, batch_size=256)

    # --- Agreement check ---
    agreement = measure_agreement(model, obs_list, act_list, device)
    print(f"\nBC policy agrees with HEFT on {agreement * 100:.1f}% of decisions")

    # --- Save ---
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(config.CHECKPOINT_DIR, "bc_pretrain")
    model.save(save_path)
    print(f"Pretrained model saved → '{save_path}.zip'")
    print("\nNext step: python -m rl_scheduler.train")
    print("  (train.py auto-loads BC weights if bc_pretrain.zip exists)")


if __name__ == "__main__":
    main()
