"""Build NB2: RL Test Flow — Steps 6-7 (Stage-2 + Optuna HPO)."""
import json
from pathlib import Path

def cc(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def mc(src): return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

cells.append(mc(
    "# RL Test Flow Optimization — Notebook 2 of 3: Stage-2 + HPO\n\n"
    "**Prerequisite**: NB1 complete + `rl-stage1-results` dataset added as input.\n"
    "**RL training device**: CPU (intentional for SB3 MLP workloads).\n\n"
    "| Step | Description | Est. Time |\n"
    "|------|-------------|----------|\n"
    "| 6 | Stage-2: top-2 algos × 3 seeds × 500K steps | ~5-6 h |\n"
    "| 7 | Optuna HPO: 30 trials × 100K steps | ~2-3 h |\n\n"
    "> **After NB2 completes**: Output tab → Create Dataset → name it **`rl-stage2-results`**\n"
    "> Then open NB3 and add both `rl-stage1-results` AND `rl-stage2-results` as inputs."
))

# Reinstall + load NB1 results
cells.append(mc("## Step 0: Reinstall + Load NB1 Results"))
cells.append(cc(
    "import subprocess, sys, os, json\n"
    "subprocess.run(['pip', 'install', '-q',\n"
    "    'stable-baselines3[extra]', 'sb3-contrib', 'gymnasium',\n"
    "    'optuna', 'mlflow', 'pyarrow', 'matplotlib', 'numpy', 'torch'], check=True)\n\n"
    "!rm -rf rl-test-flow-optimization\n"
    "!git clone https://github.com/rajendarmuddasani/rl-test-flow-optimization.git\n"
    "os.chdir('rl-test-flow-optimization')\n"
    "sys.path.insert(0, '.')\n\n"
    "import torch, numpy as np, pandas as pd\n"
    "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE — not required for this run\"}')\n"
    "print('RL train device: cpu (forced intentionally for SB3 MlpPolicy)')\n\n"
    "# Load NB1 results\n"
    "NB1_PATH = '/kaggle/input/rl-stage1-results/stage1_results.json'\n"
    "with open(NB1_PATH) as f:\n"
    "    nb1 = json.load(f)\n\n"
    "baseline_results  = nb1['baselines']\n"
    "stage1_results    = nb1['stage1']\n"
    "top2_algos        = nb1['top2_algos']\n"
    "best_bl           = nb1['best_base_reward']\n"
    "baseline_df       = pd.DataFrame(baseline_results).T\n"
    "stage1_df         = pd.DataFrame(stage1_results).T\n\n"
    "print(f'NB1 results loaded:')\n"
    "print(f'  Top-2 algos: {top2_algos}')\n"
    "print(f'  Best baseline reward: {best_bl:+.2f}')\n"
    "print(f'  Stage-1 best: {stage1_df[\"mean_reward\"].idxmax()} = {stage1_df[\"mean_reward\"].max():+.2f}')\n"
))

# Stage-2
cells.append(mc(
    "## Step 6: Stage-2 — Deep Training Top-2 (500K steps × 3 seeds)\n\n"
    "3 random seeds to measure variance. RL is stochastic — one run could be lucky.\n"
    "Mean ± std across seeds is the production-grade measurement standard."
))
cells.append(cc(
    "import time as _time\n"
    "from src.environment import TestFlowEnv\n"
    "from src.agent import ALGO_REGISTRY, evaluate_trained_model\n"
    "from pathlib import Path\n\n"
    "STAGE2_STEPS = 500_000\n"
    "SEEDS = [42, 123, 777]\n"
    "SAVE_DIR2 = Path('/kaggle/working/rl_stage2')\n"
    "SAVE_DIR2.mkdir(parents=True, exist_ok=True)\n\n"
    "# Copy NB1 models into working dir for reference\n"
    "import shutil\n"
    "nb1_models = Path('/kaggle/input/rl-stage1-results')\n"
    "if nb1_models.exists():\n"
    "    shutil.copytree(nb1_models, SAVE_DIR2 / 'stage1_ref', dirs_exist_ok=True)\n\n"
    "stage2_results = {}\n\n"
    "print(f'Stage-2: {STAGE2_STEPS:,} steps × {len(SEEDS)} seeds')\n"
    "print(f'Algorithms: {top2_algos}')\n"
    "print('=' * 70)\n\n"
    "for algo_name in top2_algos:\n"
    "    train_fn = ALGO_REGISTRY[algo_name]\n"
    "    seed_metrics = []\n"
    "    for seed in SEEDS:\n"
    "        print(f'\\n>>> {algo_name.upper()} seed={seed}')\n"
    "        env = TestFlowEnv(n_tests=200, cost_budget=50.0, time_budget=120.0)\n"
    "        t0 = _time.time()\n"
    "        model = train_fn(\n"
    "            env,\n"
    "            total_timesteps=STAGE2_STEPS,\n"
    "            output_dir=str(SAVE_DIR2 / f'{algo_name}_seed{seed}'),\n"
    "            seed=seed,\n"
    "        )\n"
    "        elapsed = _time.time() - t0\n"
    "        metrics = evaluate_trained_model(env, model, n_episodes=500)\n"
    "        metrics['seed'] = seed\n"
    "        metrics['train_time_sec'] = round(elapsed, 1)\n"
    "        seed_metrics.append(metrics)\n"
    "        print(f'  reward={metrics[\"mean_reward\"]:+.2f} | acc={metrics[\"accuracy\"]:.3f} | {elapsed/60:.1f}m')\n"
    "    stage2_results[algo_name] = seed_metrics\n\n"
    "print(f'\\n{\"=\"*60}')\n"
    "print('Stage-2 Seed Stability')\n"
    "print(f'{\"Algorithm\":20s} {\"Mean Reward\":>12s} {\"Std Reward\":>12s} {\"Mean Acc\":>10s}')\n"
    "print('-'*60)\n"
    "for algo in top2_algos:\n"
    "    rewards = [m['mean_reward'] for m in stage2_results[algo]]\n"
    "    accs    = [m['accuracy']    for m in stage2_results[algo]]\n"
    "    print(f'{algo:20s} {np.mean(rewards):>+12.2f} {np.std(rewards):>12.3f} {np.mean(accs):>10.3f}')\n"
))

# Optuna HPO
cells.append(mc(
    "## Step 7: Optuna HPO — 30 Trials × 100K Steps\n\n"
    "Tree-structured Parzen Estimator (TPE) search over:\n"
    "- `learning_rate`: 1e-5 to 1e-2 (log scale)\n"
    "- `gamma`: 0.90 to 0.999\n"
    "- `batch_size`: 32, 64, 128, 256\n"
    "- `net_arch`: [64,64], [128,128], [256,256], [256,256,128]\n\n"
    "AMD/production standard: 30+ trials is the minimum for reliable HPO."
))
cells.append(cc(
    "from src.agent import run_optuna_hpo\n\n"
    "best_algo = top2_algos[0]\n"
    "print(f'Running Optuna HPO for: {best_algo}')\n"
    "print(f'  Trials: 30  |  Steps/trial: 100,000')\n"
    "print(f'  Search: lr[1e-5..1e-2], gamma[0.9..0.999], batch_size[32/64/128/256]')\n\n"
    "hpo_result = run_optuna_hpo(\n"
    "    env_cls=TestFlowEnv,\n"
    "    env_kwargs={'n_tests': 200, 'cost_budget': 50.0, 'time_budget': 120.0},\n"
    "    algo=best_algo,\n"
    "    n_trials=30,\n"
    "    timesteps=100_000,\n"
    ")\n\n"
    "print(f'\\n{\"=\"*50}')\n"
    "print('OPTUNA HPO COMPLETE')\n"
    "print(f'  Best reward:  {hpo_result[\"best_value\"]:+.2f}')\n"
    "print(f'  Best params:')\n"
    "for k, v in hpo_result['best_params'].items():\n"
    "    print(f'    {k}: {v}')\n"
    "print(f'  Total trials: {hpo_result[\"n_trials\"]}')\n"
))

# Save NB2 outputs
cells.append(mc(
    "## Save NB2 Outputs\n\n"
    "> After this cell: Output tab → Create Dataset → **`rl-stage2-results`**\n"
    "> Add both `rl-stage1-results` and `rl-stage2-results` as inputs to NB3."
))
cells.append(cc(
    "import json\n\n"
    "save_data2 = {\n"
    "    'baselines':      baseline_results,\n"
    "    'stage1':         stage1_results,\n"
    "    'stage2':         stage2_results,\n"
    "    'hpo':            hpo_result,\n"
    "    'top2_algos':     top2_algos,\n"
    "    'best_algo':      best_algo,\n"
    "    'best_base_reward': float(best_bl),\n"
    "    'best_params':    hpo_result['best_params'],\n"
    "}\n\n"
    "with open(SAVE_DIR2 / 'stage2_results.json', 'w') as f:\n"
    "    json.dump(save_data2, f, indent=2, default=str)\n\n"
    "print('=== NB2 ARTIFACTS ===')\n"
    "total = 0\n"
    "for ff in sorted(SAVE_DIR2.rglob('*')):\n"
    "    if ff.is_file() and ff.suffix in ['.json', '.zip']:\n"
    "        print(f'  {str(ff.relative_to(SAVE_DIR2)):60s} {ff.stat().st_size/1e3:6.0f} KB')\n"
    "        total += ff.stat().st_size\n"
    "print(f'\\nTotal: {total/1e6:.1f} MB')\n"
    "print('\\nNB2 complete!')\n"
    "print('Next: Output tab → Create Dataset → rl-stage2-results')\n"
    "print('Then open NB3 and add both rl-stage1-results + rl-stage2-results')\n"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path('/Users/rajendarmuddasani/AIML/58_End2End/projects/rl-test-flow-optimization/notebooks/train_rl_kaggle_nb2.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
with open(out) as f:
    loaded = json.load(f)
print(f"NB2: {out.stat().st_size:,} bytes | {len(loaded['cells'])} cells | valid JSON ✓")
