"""Build NB1: RL Test Flow — Steps 1-5 (Setup → Stage-1 Training)."""
import json
from pathlib import Path

def cc(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def mc(src): return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────
cells.append(mc(
    "# RL Test Flow Optimization — Notebook 1 of 3: Setup + Stage-1\n\n"
    "**GPU**: Kaggle T4 x2  |  **Runtime target**: <10 hours  |  **Steps**: 1–5\n\n"
    "| Notebook | Contents | Est. Time |\n"
    "|----------|----------|----------|\n"
    "| **NB1 (this)** | Install, Data, Env, Baselines, Stage-1 (4 algos × 200K) | ~6-8 h |\n"
    "| NB2 | Stage-2 (top-2 × 3 seeds × 500K) + Optuna (30 trials × 100K) | ~7-9 h |\n"
    "| NB3 | Final (1M) + Curriculum + All Plots + Export | ~4-5 h |\n\n"
    "## Why 3 Notebooks?\n"
    "- Kaggle GPU sessions time out after **12 hours**\n"
    "- Full pipeline = ~20+ hours of compute\n"
    "- Each notebook saves models + results to `/kaggle/working/` for the next\n\n"
    "> **After NB1 completes**: Go to Output tab → Create Dataset → name it `rl-stage1-results`\n"
    "> Then add that dataset as input to NB2.\n\n"
    "## Problem: Semiconductor Test Flow Optimization\n"
    "A chip must pass through 200 tests (voltage, timing, thermal, functional, analog).\n"
    "Running all tests is expensive. An RL agent learns the **optimal test ordering** to\n"
    "maximize defect detection while minimizing cost — industry-critical for AMD, Micron, Infineon.\n\n"
    "## Algorithms Compared\n"
    "| Algorithm | Type | Key Feature |\n"
    "|-----------|------|-------------|\n"
    "| A2C | On-policy | Fast convergence, synchronous |\n"
    "| DQN | Off-policy | Experience replay, stable |\n"
    "| PPO | On-policy | Clipped surrogate objective |\n"
    "| MaskablePPO | On-policy | Action masking for invalid tests |\n\n"
    "> **Production scale note**: This runs 100K chips × 200 tests (Kaggle T4).\n"
    "> Production: 1M chips × 1000 tests on AMD MI300X / NVIDIA A100."
))

# ── Step 1: Install ────────────────────────────────────────────────────────
cells.append(mc("## Step 1: Environment Setup"))
cells.append(cc(
    "import subprocess, sys, os\n"
    "subprocess.run(['pip', 'install', '-q',\n"
    "    'stable-baselines3[extra]', 'sb3-contrib', 'gymnasium',\n"
    "    'optuna', 'mlflow', 'pyarrow', 'matplotlib', 'seaborn',\n"
    "    'pandas', 'numpy', 'torch'], check=True)\n\n"
    "# Always fresh clone → picks up latest code fixes\n"
    "!rm -rf rl-test-flow-optimization\n"
    "!git clone https://github.com/rajendarmuddasani/rl-test-flow-optimization.git\n"
    "os.chdir('rl-test-flow-optimization')\n"
    "sys.path.insert(0, '.')\n\n"
    "import torch\n"
    "print(f'PyTorch:        {torch.__version__}')\n"
    "print(f'CUDA available: {torch.cuda.is_available()}')\n"
    "if torch.cuda.is_available():\n"
    "    print(f'GPU:            {torch.cuda.get_device_name(0)}')\n"
    "    print(f'VRAM:           {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')\n"
    "else:\n"
    "    raise RuntimeError('No GPU! Go to Settings > Accelerator > GPU T4 x2')\n\n"
    "import stable_baselines3 as sb3, sb3_contrib, optuna\n"
    "print(f'SB3:            {sb3.__version__}')\n"
    "print(f'sb3-contrib:    {sb3_contrib.__version__}')\n"
    "print(f'Optuna:         {optuna.__version__}')\n"
    "print('\\nAll dependencies installed ✓')\n"
))

# ── Step 2: Generate dataset ───────────────────────────────────────────────
cells.append(mc(
    "## Step 2: Generate Test Results Dataset\n\n"
    "100K chips × 200 tests (Kaggle scale). 70% defect rate across 14 defect types.\n"
    "Production scale: 1M chips × 1000 tests on AMD EPYC cluster.\n"
    "> **Note**: The RL environment generates chip profiles on-the-fly from the test catalog.\n"
    "> This dataset is used for statistical validation and defect rate verification."
))
cells.append(cc(
    "from generate_test_data import generate_dataset\n"
    "import json\n\n"
    "summary = generate_dataset(\n"
    "    output_dir='data',\n"
    "    n_chips=100_000,\n"
    "    n_tests=200,\n"
    "    defect_rate=0.70,\n"
    "    seed=42,\n"
    "    chunk_size=25_000,\n"
    "    fmt='parquet',\n"
    ")\n"
    "print('Dataset Summary:')\n"
    "print(json.dumps(summary, indent=2))\n"
))

# ── Step 3: Environment validation ────────────────────────────────────────
cells.append(mc(
    "## Step 3: Environment Validation\n\n"
    "Verify the Gymnasium environment scales correctly at 10, 50, and 200 tests.\n"
    "**Vectorised hot paths**: `_obs()` and `action_masks()` use numpy ops (no Python loops) for 20-50x speedup."
))
cells.append(cc(
    "import numpy as np, time\n"
    "from src.environment import TestFlowEnv, DEFECT_CATEGORIES, CATEGORY_GROUPS\n\n"
    "print('Environment Validation')\n"
    "print('=' * 60)\n\n"
    "for n in [10, 50, 200]:\n"
    "    env = TestFlowEnv(n_tests=n, cost_budget=50.0, time_budget=120.0)\n"
    "    obs, _ = env.reset(seed=42)\n"
    "    masks = env.action_masks()\n"
    "    print(f'n_tests={n:4d}: obs={obs.shape[0]}d  actions={env.action_space.n}  valid={int(masks.sum())}')\n\n"
    "# Benchmark environment speed\n"
    "print('\\n=== Environment Speed Benchmark ===')\n"
    "env200 = TestFlowEnv(n_tests=200, cost_budget=50.0, time_budget=120.0)\n"
    "obs, _ = env200.reset(seed=42)\n"
    "N_BENCH = 10_000\n"
    "t0 = time.perf_counter()\n"
    "for _ in range(N_BENCH):\n"
    "    mask = env200.action_masks()\n"
    "    valid = np.where(mask)[0]\n"
    "    action = int(np.random.choice(valid))\n"
    "    obs, r, done, trunc, info = env200.step(action)\n"
    "    if done or trunc:\n"
    "        obs, _ = env200.reset()\n"
    "elapsed = time.perf_counter() - t0\n"
    "print(f'{N_BENCH:,} steps in {elapsed:.2f}s  →  {N_BENCH/elapsed:.0f} steps/sec')\n"
    "print(f'Estimated time for 200K training steps: {200_000/(N_BENCH/elapsed)/60:.1f} min')\n"
    "print('\\nEnvironment checks passed ✓')\n"
))

# ── Step 4: Baselines ─────────────────────────────────────────────────────
cells.append(mc(
    "## Step 4: Baseline Evaluation\n\n"
    "Three heuristic policies evaluated over 500 episodes. RL agents must beat these to be useful.\n\n"
    "| Baseline | Strategy |\n"
    "|----------|----------|\n"
    "| Random | Uniformly random from valid tests |\n"
    "| Greedy Coverage | Always highest-coverage test |\n"
    "| Cost Efficient | Best coverage/cost ratio up to 40% budget |"
))
cells.append(cc(
    "import pandas as pd\n"
    "from src.agent import BASELINES, evaluate_policy\n\n"
    "env = TestFlowEnv(n_tests=200, cost_budget=50.0, time_budget=120.0)\n\n"
    "print('Baseline Evaluation (500 episodes each)')\n"
    "print('=' * 70)\n"
    "print(f'{\"Policy\":20s} {\"Reward\":>10s} {\"Accuracy\":>10s} {\"Cost\":>10s} {\"Tests\":>10s}')\n"
    "print('-' * 70)\n\n"
    "baseline_results = {}\n"
    "for name, policy_fn in BASELINES.items():\n"
    "    metrics = evaluate_policy(env, policy_fn, n_episodes=500)\n"
    "    baseline_results[name] = metrics\n"
    "    print(f'{name:20s} {metrics[\"mean_reward\"]:>+10.2f} {metrics[\"accuracy\"]:>10.3f} '\n"
    "          f'{metrics[\"mean_cost\"]:>10.2f} {metrics[\"mean_tests\"]:>10.1f}')\n\n"
    "print('=' * 70)\n"
    "baseline_df = pd.DataFrame(baseline_results).T\n"
    "best_bl = baseline_df['mean_reward'].max()\n"
    "print(f'Best baseline: {baseline_df[\"mean_reward\"].idxmax()} (reward={best_bl:+.2f})')\n"
    "print(f'\\nRL must beat: {best_bl:+.2f} reward')\n"
))

# ── Step 5: Stage-1 training ───────────────────────────────────────────────
cells.append(mc(
    "## Step 5: Stage-1 — Train All 4 Algorithms (200K steps each)\n\n"
    "**Execution order**: A2C → DQN → PPO → MaskablePPO (fastest to slowest).\n"
    "Progress prints every 10K steps with elapsed time and ETA.\n\n"
    "| Algorithm | Key Setting | Expected Time |\n"
    "|-----------|------------|---------------|\n"
    "| A2C | n_steps=5, fast updates | ~1-2h |\n"
    "| DQN | buffer=100K, replay | ~1-2h |\n"
    "| PPO | n_steps=2048, clipped | ~1-2h |\n"
    "| MaskablePPO | action masking | ~2-3h |\n\n"
    "> Checkpoints saved to `/kaggle/working/rl_stage1/` every 5K steps via EvalCallback."
))
cells.append(cc(
    "import time as _time, json\n"
    "import pandas as pd\n"
    "from pathlib import Path\n"
    "from src.agent import ALGO_REGISTRY, evaluate_trained_model\n\n"
    "STAGE1_STEPS = 200_000\n"
    "SAVE_DIR = Path('/kaggle/working/rl_stage1')\n"
    "SAVE_DIR.mkdir(parents=True, exist_ok=True)\n\n"
    "stage1_results = {}\n\n"
    "print(f'Stage-1: Training 4 algorithms × {STAGE1_STEPS:,} steps')\n"
    "print('=' * 70)\n\n"
    "for algo_name, train_fn in ALGO_REGISTRY.items():\n"
    "    print(f'\\n>>> {algo_name.upper()}')\n"
    "    env = TestFlowEnv(n_tests=200, cost_budget=50.0, time_budget=120.0)\n"
    "    t0 = _time.time()\n"
    "    model = train_fn(\n"
    "        env,\n"
    "        total_timesteps=STAGE1_STEPS,\n"
    "        output_dir=str(SAVE_DIR / algo_name),\n"
    "    )\n"
    "    train_time = _time.time() - t0\n"
    "    metrics = evaluate_trained_model(env, model, n_episodes=500)\n"
    "    metrics['train_time_sec'] = round(train_time, 1)\n"
    "    stage1_results[algo_name] = metrics\n"
    "    print(f'  DONE: reward={metrics[\"mean_reward\"]:+.2f} | '\n"
    "        f'accuracy={metrics[\"accuracy\"]:.3f} | '\n"
    "        f'cost={metrics[\"mean_cost\"]:.2f} | '\n"
    "        f'time={train_time/60:.1f}m')\n\n"
    "# Save results\n"
    "stage1_df = pd.DataFrame(stage1_results).T\n"
    "ranked = stage1_df.sort_values('mean_reward', ascending=False)\n"
    "top2_algos = list(ranked.index[:2])\n\n"
    "print(f'\\n{\"=\"*70}')\n"
    "print('STAGE-1 SUMMARY')\n"
    "print(stage1_df.to_string())\n"
    "print(f'\\nTop-2 for Stage-2: {top2_algos}')\n\n"
    "for algo, row in stage1_df.iterrows():\n"
    "    imp = row['mean_reward'] - best_bl\n"
    "    pct = (imp / abs(best_bl)) * 100 if best_bl != 0 else 0\n"
    "    print(f'  {algo} vs best baseline: {imp:+.2f} ({pct:+.1f}%)')\n"
))

# ── Save all outputs ───────────────────────────────────────────────────────
cells.append(mc(
    "## Save Outputs for NB2\n\n"
    "All models already saved to `/kaggle/working/rl_stage1/` by EvalCallback.\n"
    "Now save the metrics JSON and create the summary.\n\n"
    "> **After this cell**: Output tab → Create Dataset → name it **`rl-stage1-results`**\n"
    "> Add it as input to NB2."
))
cells.append(cc(
    "import json, shutil\n"
    "from pathlib import Path\n\n"
    "save_data = {\n"
    "    'baselines': baseline_results,\n"
    "    'stage1':    stage1_results,\n"
    "    'top2_algos': top2_algos,\n"
    "    'best_base_reward': float(best_bl),\n"
    "}\n\n"
    "with open(SAVE_DIR / 'stage1_results.json', 'w') as f:\n"
    "    json.dump(save_data, f, indent=2)\n\n"
    "print('=== NB1 ARTIFACTS ===')\n"
    "for f in sorted(SAVE_DIR.rglob('*')):\n"
    "    if f.is_file():\n"
    "        print(f'  {str(f.relative_to(SAVE_DIR)):60s} {f.stat().st_size/1e3:6.0f} KB')\n\n"
    "total = sum(f.stat().st_size for f in SAVE_DIR.rglob('*') if f.is_file())\n"
    "print(f'\\nTotal: {total/1e6:.1f} MB')\n"
    "print('\\nNB1 complete!')\n"
    "print('Next: Output tab → Create Dataset → name it rl-stage1-results')\n"
    "print('Then open NB2 and add rl-stage1-results as input dataset')\n"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path('/Users/rajendarmuddasani/AIML/58_End2End/projects/rl-test-flow-optimization/notebooks/train_rl_kaggle_nb1.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
with open(out) as f:
    loaded = json.load(f)
print(f"NB1: {out.stat().st_size:,} bytes | {len(loaded['cells'])} cells | valid JSON ✓")
