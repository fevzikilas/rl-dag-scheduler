"""Hyperparameters and hardware configuration."""

# GPU is modeled as N parallel CUDA streams sharing one VRAM pool.
# Both stream count and VRAM are randomized during training for hardware generalization.

MAX_STREAMS       = 8
N_STREAMS_MIN     = 2
N_STREAMS_MAX     = MAX_STREAMS
N_STREAMS_DEFAULT = 4

GPU_FLOPS_CAP = 10e12
GPU_MEMORY_DEFAULT = 8e9
GPU_MEMORY_MIN     = 4e9
GPU_MEMORY_MAX     = 100e9

# Node features (5 dimensions)
NODE_FEAT_DIM = 5

# Structural task features (6 dimensions: includes rank_u for priority)
TASK_FEAT_DIM = 6

# GNN architecture
HIDDEN_DIM   = 128
N_GAT_HEADS  = 4
N_GAT_LAYERS = 3
GAT_DROPOUT  = 0.1

# PPO hyperparameters
PPO_LR         = 3e-4      # decays to 3e-5 over training
PPO_N_STEPS    = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS   = 10
PPO_GAMMA      = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP       = 0.2
PPO_ENT_COEF   = 0.05
TOTAL_TIMESTEPS = 2_000_000

# Reward shaping
IDLE_PENALTY = 0.1

# Training and evaluation
TRAIN_RATIO     = 0.80
SEED            = 42
LOG_INTERVAL    = 10
EVAL_FREQ       = 20_000
N_EVAL_EPISODES = 35
CHECKPOINT_DIR  = "checkpoints"
