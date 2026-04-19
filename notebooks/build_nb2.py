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
    "| Step | Description | Est. Time (Kaggle T4 CPU) |\n"
    "|------|-------------|---------------------------|\n"
    "| 0 | Reinstall + load NB1 results | ~3 min |\n"
    "| 6 | Stage-2: top-2 algos × 3 seeds × 500K steps | ~60-70 min |\n"
    "| 7 | Optuna HPO: 30 trials × 100K steps on best algo | ~55-65 min |\n"
    "| Save | Serialize everything to `/kaggle/working/rl_stage2/` | ~1 min |\n\n"
    "**Total: ~2-2.5 hours**  (well within Kaggle's 12-hour limit)\n\n"
    "## Timing Reference (from NB1 on this Kaggle session)\n"
    "| Algorithm | 200K steps | 100K steps | 500K steps |\n"
    "|-----------|-----------|------------|------------|\n"
    "| DQN (best) | 7.4 min | ~3.7 min | ~18.5 min |\n"
    "| PPO (2nd) | 7.1 min | ~3.6 min | ~17.8 min |\n\n"
    "> **After NB2 completes**: Output tab → Create Dataset → name it **`rl-stage2-results`**\n"
    "> Then open NB3 and add both `rl-stage1-results` AND `rl-stage2-results` as inputs.\n\n"
    "## Evaluation Fix (applied in this NB)\n"
    "NB1 revealed that post-training evaluation was broken: a deterministic A2C/PPO policy\n"
    "loops on duplicate actions, accumulating -1×410 = -410 reward/episode. Fixed in `agent.py`:\n"
    "- MaskablePPO: passes `action_masks` to `model.predict()` (the correct sb3-contrib API)\n"
    "- PPO/DQN/A2C: if predicted action is invalid (duplicate/unaffordable), auto-STOP instead"
))

# Step 0: Reinstall + load NB1 results
cells.append(mc("## Step 0: Reinstall + Load NB1 Results"))
cells.append(cc(
    "import subprocess, sys, os, json, time as _time\n"
    "subprocess.run(['pip', 'install', '-q',\n"
    "    'stable-baselines3[extra]', 'sb3-contrib', 'gymnasium',\n"
    "    'optuna', 'mlflow', 'pyarrow', 'matplotlib', 'numpy', 'torch'], check=True)\n\n"
    "!rm -rf rl-test-flow-optimization\n"
    "!git clone https://github.com/rajendarmuddasani/rl-test-flow-optimization.git\n"
    "os.chdir('rl-test-flow-optimization')\n"
    "sys.path.insert(0, '.')\n\n"
    "# Verify evaluation fix is present\n"
    "import inspect\n"
    "from src.agent import evaluate_trained_model\n"
    "assert 'is_maskable' in inspect.getsource(evaluate_trained_model), \\\n"
    "    'ERROR: Old evaluation code without action-mask fix! Check git clone.'\n"
    "print('Code version verified: action-mask evaluation fix present ✓')\n\n"
    "import torch, numpy as np, pandas as pd\n"
    "gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'\n"
    "print(f'GPU: {gpu}  |  RL train device: cpu (forced for SB3 MlpPolicy)')\n\n"
    "# Load NB1 results\n"
    "NB1_PATH = '/kaggle/input/rl-stage1-results/stage1_results.json'\n"
    "with open(NB1_PATH) as f:\n"
    "    nb1 = json.load(f)\n\n"
    "baseline_results = nb1['baselines']\n"
    "stage1_results   = nb1['stage1']\n"
    "top2_algos       = nb1['top2_algos']\n"
    "best_bl          = nb1['best_base_reward']\n"
    "baseline_df      = pd.DataFrame(baseline_results).T\n"
    "stage1_df        = pd.DataFrame(stage1_results).T\n\n"
    "print(f'\\nNB1 results loaded successfully:')\n"
    "print(f'  Top-2 algos for Stage-2: {top2_algos}')\n"
    "print(f'  Best baseline reward: {best_bl:+.2f} (cost_efficient heuristic)')\n"
    "print(f'  Stage-1 winner: {stage1_df[\"mean_reward\"].idxmax()} '\n"
    "      f'= {stage1_df[\"mean_reward\"].max():+.2f}')\n"
    "print('\\nStage-1 table:')\n"
    "print(stage1_df[['mean_reward','accuracy','mean_cost','mean_tests']].to_string())\n"
))

# Step 6: Stage-2
cells.append(mc(
    "## Step 6: Stage-2 — Deep Training Top-2 (500K steps × 3 seeds)\n\n"
    "3 random seeds quantify variance — RL optimisation is stochastic.\n"
    "A single run could be lucky or unlucky; mean ± std across seeds is the production-grade metric.\n\n"
    "| What we measure | Why it matters |\n"
    "|-----------------|----------------|\n"
    "| Mean reward | Raw RL performance |\n"
    "| Std across seeds | Reproducibility — low std = stable policy |\n"
    "| Accuracy | % chips correctly classified (defect vs clean) |\n"
    "| Mean cost/chip | AMD production metric — lower = cheaper test run |\n\n"
    "Expected: each algo × seed run ≈ 18-19 min. Total ≈ 60-65 min."
))
cells.append(cc(
    "import sys\n"
    "from src.environment import TestFlowEnv\n"
    "from src.agent import ALGO_REGISTRY, evaluate_trained_model\n"
    "from pathlib import Path\n\n"
    "def fp(*a, **k): print(*a, **k); sys.stdout.flush()\n\n"
    "STAGE2_STEPS = 500_000\n"
    "SEEDS        = [42, 123, 777]\n"
    "SAVE_DIR2    = Path('/kaggle/working/rl_stage2')\n"
    "SAVE_DIR2.mkdir(parents=True, exist_ok=True)\n\n"
    "# Copy NB1 stage1_results.json into working dir\n"
    "import shutil\n"
    "shutil.copy(NB1_PATH, SAVE_DIR2 / 'stage1_results.json')\n\n"
    "stage2_results = {}\n"
    "total_runs = len(top2_algos) * len(SEEDS)\n"
    "run_idx = 0\n"
    "overall_t0 = _time.time()\n\n"
    "fp(f'Stage-2: {STAGE2_STEPS:,} steps × {len(SEEDS)} seeds × {len(top2_algos)} algos')\n"
    "fp(f'Total runs: {total_runs}  |  Est. ~{total_runs * 18:.0f} min total')\n"
    "fp('=' * 70)\n\n"
    "for algo_name in top2_algos:\n"
    "    train_fn = ALGO_REGISTRY[algo_name]\n"
    "    seed_metrics = []\n"
    "    for seed in SEEDS:\n"
    "        run_idx += 1\n"
    "        elapsed_so_far = (_time.time() - overall_t0) / 60\n"
    "        fp(f'\\n[Run {run_idx}/{total_runs}] {algo_name.upper()} seed={seed}')\n"
    "        fp(f'  Start: {_time.strftime(\"%H:%M:%S\")}  '\n"
    "           f'| elapsed so far: {elapsed_so_far:.1f}m  '\n"
    "           f'| est. remaining: {(total_runs - run_idx + 1) * 18:.0f}m')\n"
    "        env = TestFlowEnv(n_tests=200, cost_budget=50.0, time_budget=120.0)\n"
    "        t0 = _time.time()\n"
    "        model = train_fn(\n"
    "            env,\n"
    "            total_timesteps=STAGE2_STEPS,\n"
    "            output_dir=str(SAVE_DIR2 / f'{algo_name}_seed{seed}'),\n"
    "            seed=seed,\n"
    "        )\n"
    "        elapsed = _time.time() - t0\n"
    "        fp(f'  Training done in {elapsed/60:.1f}m. Evaluating 100 episodes...')\n"
    "        metrics = evaluate_trained_model(env, model, n_episodes=100)\n"
    "        metrics['seed']           = seed\n"
    "        metrics['train_time_sec'] = round(elapsed, 1)\n"
    "        seed_metrics.append(metrics)\n"
    "        fp(f'  DONE: reward={metrics[\"mean_reward\"]:+.2f} | '\n"
    "           f'acc={metrics[\"accuracy\"]:.3f} | '\n"
    "           f'cost={metrics[\"mean_cost\"]:.2f} | '\n"
    "           f'tests={metrics[\"mean_tests\"]:.1f}')\n"
    "    stage2_results[algo_name] = seed_metrics\n\n"
    "fp(f'\\n{\"=\"*70}')\n"
    "fp(f'STAGE-2 COMPLETE  |  Total: {(_time.time()-overall_t0)/60:.1f} min')\n"
    "fp(f'\\n{\"Algorithm\":20s} {\"Mean Reward\":>12s} {\"Std\u00b1\":>8s} {\"Accuracy\":>10s} {\"Cost\":>8s}')\n"
    "fp('-' * 70)\n"
    "for algo in top2_algos:\n"
    "    rewards = [m['mean_reward'] for m in stage2_results[algo]]\n"
    "    accs    = [m['accuracy']    for m in stage2_results[algo]]\n"
    "    costs   = [m['mean_cost']   for m in stage2_results[algo]]\n"
    "    fp(f'{algo:20s} {np.mean(rewards):>+12.2f} {np.std(rewards):>8.3f} '\n"
    "       f'{np.mean(accs):>10.3f} {np.mean(costs):>8.2f}')\n"
    "fp(f'\\nBaseline best: {best_bl:+.2f} (cost_efficient heuristic)')\n"
))

# Step 7: Optuna HPO
cells.append(mc(
    "## Step 7: Optuna HPO — 30 Trials × 100K Steps\n\n"
    "Tree-structured Parzen Estimator (TPE) searches for the best hyperparameters.\n\n"
    "| Hyperparameter | Search Space | Why |\n"
    "|----------------|-------------|-----|\n"
    "| `learning_rate` | 1e-5 → 1e-2 (log) | Most impactful RL param |\n"
    "| `gamma` | 0.90 → 0.999 | Discount: near-sighted vs far-sighted |\n"
    "| `batch_size` | 32, 64, 128, 256 | Gradient noise vs compute |\n\n"
    "**Algorithm**: `top2_algos[0]` (DQN from NB1)  \n"
    "**30 trials × ~3.7 min each ≈ 55-65 min**  \n"
    "AMD production standard: 30+ trials minimum for reliable HPO."
))
cells.append(cc(
    "import sys\n"
    "from src.agent import run_optuna_hpo\n\n"
    "def fp(*a, **k): print(*a, **k); sys.stdout.flush()\n\n"
    "best_algo = top2_algos[0]  # DQN from NB1\n"
    "n_trials  = 30\n"
    "hpo_steps = 100_000\n\n"
    "fp(f'Optuna HPO: algo={best_algo}, trials={n_trials}, steps/trial={hpo_steps:,}')\n"
    "fp(f'Est. time: {n_trials} × ~3.7 min = ~{n_trials * 3.7:.0f} min')\n"
    "fp(f'Started:   {_time.strftime(\"%H:%M:%S\")}')\n"
    "fp('=' * 60)\n\n"
    "t_hpo = _time.time()\n"
    "hpo_result = run_optuna_hpo(\n"
    "    env_cls=TestFlowEnv,\n"
    "    env_kwargs={'n_tests': 200, 'cost_budget': 50.0, 'time_budget': 120.0},\n"
    "    algo=best_algo,\n"
    "    n_trials=n_trials,\n"
    "    timesteps=hpo_steps,\n"
    ")\n"
    "hpo_elapsed = (_time.time() - t_hpo) / 60\n\n"
    "fp(f'\\n{\"=\"*60}')\n"
    "fp(f'OPTUNA COMPLETE  |  {hpo_elapsed:.1f} min')\n"
    "fp(f'  Best reward: {hpo_result[\"best_value\"]:+.2f}')\n"
    "fp(f'  Best params:')\n"
    "for k, v in hpo_result['best_params'].items():\n"
    "    fp(f'    {k}: {v}')\n"
    "fp(f'  Total trials completed: {hpo_result[\"n_trials\"]}')\n"
))

# Save NB2 outputs
cells.append(mc(
    "## Save NB2 Outputs\n\n"
    "Serialises all stage-2 + HPO results plus copies of stage-1 data.\n\n"
    "> **After this cell completes**: Output tab → Create Dataset → **`rl-stage2-results`**\n"
    "> \n"
    "> Then open NB3 and add BOTH:\n"
    "> - `rl-stage1-results` (from NB1)\n"
    "> - `rl-stage2-results` (from this notebook)"
))
cells.append(cc(
    "import json, sys\n"
    "def fp(*a, **k): print(*a, **k); sys.stdout.flush()\n\n"
    "save_data2 = {\n"
    "    'baselines':        baseline_results,\n"
    "    'stage1':           stage1_results,\n"
    "    'stage2':           stage2_results,\n"
    "    'hpo':              hpo_result,\n"
    "    'top2_algos':       top2_algos,\n"
    "    'best_algo':        best_algo,\n"
    "    'best_base_reward': float(best_bl),\n"
    "    'best_params':      hpo_result['best_params'],\n"
    "}\n\n"
    "with open(SAVE_DIR2 / 'stage2_results.json', 'w') as f:\n"
    "    json.dump(save_data2, f, indent=2, default=str)\n\n"
    "fp('=== NB2 ARTIFACTS ===')\n"
    "total = 0\n"
    "for ff in sorted(SAVE_DIR2.rglob('*')):\n"
    "    if ff.is_file() and ff.suffix in ['.json', '.zip']:\n"
    "        fp(f'  {str(ff.relative_to(SAVE_DIR2)):60s} {ff.stat().st_size/1e3:6.0f} KB')\n"
    "        total += ff.stat().st_size\n"
    "fp(f'\\nTotal: {total/1e6:.1f} MB')\n"
    "fp('\\nNB2 complete!')\n"
    "fp('Next: Output tab → Create Dataset → rl-stage2-results')\n"
    "fp('Then open NB3 and add both rl-stage1-results + rl-stage2-results as inputs')\n"
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
