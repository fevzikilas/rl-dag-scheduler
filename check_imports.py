import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rl_scheduler'))

mods = [
    ('config',       'import config'),
    ('data_loader',  'from data_loader import load_all_graphs'),
    ('gnn_policy',   'from gnn_policy import NodeEncoder, HPCFeaturesExtractor'),
    ('hpc_env',      'from hpc_env import HPCClusterEnv'),
    ('baselines',    'from baselines import heft, round_robin'),
    ('train-imports','from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor; from stable_baselines3.common.callbacks import BaseCallback'),
]

for name, stmt in mods:
    try:
        exec(stmt)
        print(f'  OK  {name}')
    except Exception as e:
        print(f'FAIL  {name}: {e}')
        traceback.print_exc()
        break

print('done')
