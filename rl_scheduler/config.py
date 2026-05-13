# config.py — hardware specs and training hyperparameters

# The GPU is modelled as N parallel CUDA streams sharing one VRAM pool.
# Both stream count and VRAM are randomised per training episode so the
# policy generalises to unseen hardware at eval time.

MAX_STREAMS       = 8   # obs/action space is fixed to this upper bound
N_STREAMS_MIN     = 2
N_STREAMS_MAX     = MAX_STREAMS
N_STREAMS_DEFAULT = 4   # used for evaluation

N_STREAMS = N_STREAMS_DEFAULT  # alias
N_PROCS   = N_STREAMS_DEFAULT  # alias

GPU_FLOPS_CAP = 10e12   # 10 TFLOPS total; split equally across streams

GPU_MEMORY_DEFAULT = 8e9    # 8 GB — evaluation default
GPU_MEMORY_MIN     = 4e9    # 4 GB
GPU_MEMORY_MAX     = 100e9  # 100 GB — covers RTX to H100

# Node features: flops, mem_byte, transfer_byte, depth, in_degree
NODE_FEAT_DIM = 5

# GNN encoder
HIDDEN_DIM   = 128
N_GAT_HEADS  = 4
N_GAT_LAYERS = 3
GAT_DROPOUT  = 0.1

# PPO
PPO_LR         = 3e-4
PPO_N_STEPS    = 1024
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS   = 10
PPO_GAMMA      = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP       = 0.2
PPO_ENT_COEF   = 0.01
TOTAL_TIMESTEPS = 1_000_000

# Reward shaping
IDLE_PENALTY           = 0.01
INVALID_ACTION_PENALTY = -50.0

# Training / evaluation settings
TRAIN_RATIO     = 0.80
SEED            = 42
LOG_INTERVAL    = 10
EVAL_FREQ       = 20_000
N_EVAL_EPISODES = 20
CHECKPOINT_DIR  = "checkpoints"
